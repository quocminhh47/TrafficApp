# modules/model_utils.py

from modules.data_loader import load_slice
import numpy as np
import pandas as pd


# -------------------------------------------------
# Time features: sin/cos hour & day-of-week
# -------------------------------------------------
def time_feats(dt):
    """
    Nhận vào: Series / DatetimeIndex / list datetime
    Trả về: np.ndarray shape (N, 4):
      [sin(hour), cos(hour), sin(dow), cos(dow)]
    """
    dt = pd.to_datetime(dt)

    # Nếu là DatetimeIndex → convert sang Series để dùng .dt
    if isinstance(dt, pd.DatetimeIndex):
        dt = pd.Series(dt)

    # Nếu là list/ndarray → cũng convert sang Series
    if not isinstance(dt, pd.Series):
        dt = pd.Series(dt)

    hour = dt.dt.hour
    dow = dt.dt.dayofweek

    feats = np.c_[
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * dow / 7),
        np.cos(2 * np.pi * dow / 7),
    ].astype(np.float32)

    return feats


# -------------------------------------------------
# Baseline fallback (khi GRU không dùng được)
# -------------------------------------------------
def _baseline_forecast(route_id, base_date, horizon, hist_df: pd.DataFrame | None = None):
    """
    Fallback đơn giản khi GRU không chạy được.

    Logic:
      - Nếu có hist_df: lấy mean(Vehicles) làm giá trị constant.
      - Nếu không có gì: PredictedVehicles = 0.
    """
    base_dt = pd.to_datetime(base_date)
    next_hours = pd.date_range(base_dt, periods=horizon, freq="H")

    if hist_df is not None and not hist_df.empty:
        v = pd.to_numeric(hist_df.get("Vehicles", pd.Series([])), errors="coerce").dropna()
        if not v.empty:
            val = float(v.mean())
        else:
            val = 0.0
    else:
        val = 0.0

    df_fc = pd.DataFrame(
        {
            "DateTime": next_hours,
            "RouteId": route_id,
            "PredictedVehicles": val,
        }
    )
    return df_fc, "Baseline"


# -------------------------------------------------
# GRU: dùng history LOOKBACK giờ trước base_date
# -------------------------------------------------
def forecast_gru(
    route_id,
    base_date,
    model,       # GRU model (traffic_seq.keras) đã load ở app / model_manager
    meta: dict,  # seq_meta.json
    scaler,      # vehicles_scaler.pkl
    routes_model,  # list routes trong meta
    rid2idx,       # dict route → index (có thể không dùng)
    df_hist: pd.DataFrame,
):
    """
    Dự báo 24h bằng GRU, nếu thiếu history / lỗi → fallback Baseline.

    df_hist: lịch sử đã được app load sẵn (từ parquet), gồm các cột:
             DateTime, RouteId, Vehicles (ít nhất)
    """
    LOOKBACK = int(meta.get("LOOKBACK", 168))
    HORIZON = int(meta.get("HORIZON", 24))
    routes = list(meta.get("routes", routes_model))
    n_routes = len(routes)

    base_dt = pd.to_datetime(base_date)

    # ---- Lọc đúng route & resample 1h ----
    if df_hist is None or df_hist.empty:
        print(f"[GRU] No df_hist passed for {route_id} → fallback Baseline.")
        return _baseline_forecast(route_id, base_date, HORIZON, hist_df=None)

    g = df_hist[df_hist["RouteId"].astype(str) == str(route_id)].copy()
    if g.empty:
        print(f"[GRU] No history rows for {route_id} in df_hist → fallback Baseline.")
        return _baseline_forecast(route_id, base_date, HORIZON, hist_df=None)

    g["DateTime"] = pd.to_datetime(g["DateTime"], errors="coerce")
    g["Vehicles"] = pd.to_numeric(g["Vehicles"], errors="coerce")
    g = g.dropna(subset=["DateTime", "Vehicles"])

    if g.empty:
        print(f"[GRU] History became empty after cleaning for {route_id} → fallback Baseline.")
        return _baseline_forecast(route_id, base_date, HORIZON, hist_df=None)

    g = (
        g.set_index("DateTime")
        .resample("1h")["Vehicles"]       # dùng "1h" để tránh FutureWarning
        .mean()
        .dropna()
        .reset_index()
        .sort_values("DateTime")
    )

    # ---- Lấy đúng LOOKBACKh trước base_dt ----
    g_hist = g[g["DateTime"] < base_dt].tail(LOOKBACK)

    print(
        f"[GRU] {route_id} hist rows={len(g_hist)} from "
        f"{g_hist['DateTime'].min()} → {g_hist['DateTime'].max()}"
    )

    if len(g_hist) < LOOKBACK:
        # Không đủ history → dùng Baseline trên history hiện có
        print(
            f"[GRU] Route {route_id}: only {len(g_hist)}h (<{LOOKBACK}h) → fallback Baseline."
        )
        return _baseline_forecast(route_id, base_date, HORIZON, hist_df=g_hist)

    # ---- Chuẩn bị input cho GRU giống lúc train ----
    v_raw = g_hist["Vehicles"].values.astype(float)
    v_scaled = scaler.transform(v_raw.reshape(-1, 1)).reshape(-1)  # (LOOKBACK,)

    dt_hist = g_hist["DateTime"]
    tf_feats = time_feats(dt_hist)  # (LOOKBACK, 4)

    onehot = np.zeros((LOOKBACK, n_routes), dtype=np.float32)
    try:
        rid_idx = routes.index(str(route_id))
        onehot[:, rid_idx] = 1.0
    except ValueError:
        print(f"[GRU] route {route_id} not in meta.routes → fallback Baseline.")
        return _baseline_forecast(route_id, base_date, HORIZON, hist_df=g_hist)

    X = np.concatenate(
        [v_scaled.reshape(-1, 1), tf_feats, onehot], axis=1
    )  # (LOOKBACK, 1+4+n_routes)
    X = X[np.newaxis, ...]  # (1, LOOKBACK, features)

    # ---- Predict bằng GRU ----
    if model is None:
        print(f"[GRU] model is None for {route_id} → fallback Baseline.")
        return _baseline_forecast(route_id, base_date, HORIZON, hist_df=g_hist)

    try:
        y_scaled = model.predict(X, verbose=0).reshape(-1, 1)  # (HORIZON, 1)
        y = scaler.inverse_transform(y_scaled).reshape(-1)     # (HORIZON,)
    except Exception as e:
        print(f"[GRU] ❌ predict error for {route_id}: {e} → fallback Baseline.")
        return _baseline_forecast(route_id, base_date, HORIZON, hist_df=g_hist)

    next_hours = pd.date_range(base_dt, periods=HORIZON, freq="H")

    df_fc = pd.DataFrame(
        {
            "DateTime": next_hours,
            "RouteId": route_id,
            "PredictedVehicles": y,
        }
    )
    return df_fc, "GRU"


def _forecast_seq_model(
    route_id: str,
    base_date,
    seq_model,
    model_name: str,          # "GRU" hoặc "RNN"
    meta: dict,
    scaler,
    routes_model: list[str],
    rid2idx: dict[str, int],
    df_hist: pd.DataFrame,
):
    """
    Core logic dự báo HORIZON giờ sau base_date bằng 1 model sequence bất kỳ (GRU/RNN).
    """
    if seq_model is None:
        print(f"[{model_name}] ❌ seq_model is None → fallback MLP.")
        from modules.model_utils import forecast_mlp  # tránh import vòng
        return forecast_mlp(route_id, base_date, scaler, routes_model, horizon=meta.get("HORIZON", 24))

    LOOKBACK = int(meta.get("LOOKBACK", 168))
    HORIZON = int(meta.get("HORIZON", 24))

    if df_hist is None or df_hist.empty:
        print(f"[{model_name}] ❌ empty history → fallback MLP.")
        from modules.model_utils import forecast_mlp
        return forecast_mlp(route_id, base_date, scaler, routes_model, horizon=HORIZON)

    df = df_hist.copy()
    df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
    df = df.dropna(subset=["DateTime"])

    # Nếu df_hist chứa nhiều route thì filter
    if "RouteId" in df.columns:
        df = df[df["RouteId"].astype(str) == str(route_id)]

    if df.empty:
        print(f"[{model_name}] ❌ no history for route {route_id} → fallback MLP.")
        from modules.model_utils import forecast_mlp
        return forecast_mlp(route_id, base_date, scaler, routes_model, horizon=HORIZON)

    base_dt = pd.to_datetime(base_date).normalize()
    hist_end = base_dt
    hist_start = hist_end - pd.Timedelta(hours=LOOKBACK)

    df = df[(df["DateTime"] >= hist_start) & (df["DateTime"] < hist_end)]
    if len(df) < LOOKBACK:
        print(f"[{model_name}] ❌ not enough history len={len(df)} < {LOOKBACK} → fallback MLP.")
        from modules.model_utils import forecast_mlp
        return forecast_mlp(route_id, base_date, scaler, routes_model, horizon=HORIZON)

    df = df.sort_values("DateTime")
    veh = df["Vehicles"].astype(float).values.reshape(-1, 1)
    veh_scaled = scaler.transform(veh)   # (T, 1)

    tf_time = time_feats(df["DateTime"])  # (T, 4)

    n_routes = len(routes_model)
    onehot = np.zeros((len(df), n_routes), dtype=np.float32)
    if route_id in rid2idx:
        j = rid2idx[route_id]
        onehot[:, j] = 1.0

    feats = np.concatenate([veh_scaled, tf_time, onehot], axis=1)  # (T, 1+4+n_routes)

    # Lấy đúng LOOKBACK cuối
    feats = feats[-LOOKBACK:, :]
    X = feats.reshape(1, LOOKBACK, -1)

    try:
        y_scaled = seq_model.predict(X, verbose=0).reshape(-1)   # (HORIZON,)
        y = scaler.inverse_transform(y_scaled.reshape(-1, 1)).reshape(-1)
    except Exception as e:
        print(f"[{model_name}] ❌ predict error: {e} → fallback MLP.")
        from modules.model_utils import forecast_mlp
        return forecast_mlp(route_id, base_date, scaler, routes_model, horizon=HORIZON)

    next_hours = pd.date_range(base_dt, periods=HORIZON, freq="H")

    df_fc = pd.DataFrame(
        {
            "DateTime": next_hours,
            "RouteId": route_id,
            "PredictedVehicles": y,
        }
    )
    return df_fc, model_name


def forecast_week_after_last_point(route_id, city, zone, ctx, n_days=7):
    """
    Dùng GRU để forecast n_days (mặc định 7) sau NGÀY CUỐI CÙNG trong dữ liệu thật.

    - Không dịch dữ liệu, forecast trên timeline thật (2012–2018).
    - Trả về:
        df_fc_raw: DataFrame forecast trên timeline thật (ngay sau năm 2018)
        anchor_day_raw: ngày cuối cùng trong dữ liệu (normalize, 00:00)
    """
    # 1) Load toàn bộ series của route
    df_full = load_slice(
        city=city,
        zone=zone,
        routes=[route_id],
        start_dt=None,
        end_dt=None,
    )
    if df_full.empty:
        return pd.DataFrame(), None

    df_full["DateTime"] = pd.to_datetime(df_full["DateTime"], errors="coerce")
    df_full = df_full.dropna(subset=["DateTime", "Vehicles"])
    df_full = df_full.sort_values("DateTime")

    last_dt = df_full["DateTime"].max()
    anchor_day_raw = last_dt.normalize()  # ví dụ: 2018-10-31 00:00

    # 2) History tổng hợp (ban đầu = dữ liệu thật)
    hist = df_full.copy()

    all_fc = []

    for k in range(1, n_days + 1):
        # base_date = đầu ngày thứ k sau anchor_day
        base_date = anchor_day_raw + pd.Timedelta(days=k)

        # history LOOKBACK giờ trước base_date
        hist_start = base_date - pd.Timedelta(hours=ctx.lookback)
        df_hist = hist[
            (hist["DateTime"] >= hist_start) &
            (hist["DateTime"] < base_date)
        ]

        if len(df_hist) < ctx.lookback:
            # không đủ history thì dừng
            print(f"[forecast_week] Route {route_id}: thiếu history cho ngày {base_date}, dừng.")
            break

        df_fc_day, model_used = forecast_gru(
            route_id=route_id,
            base_date=base_date,
            model=ctx.gru_model,
            meta=ctx.meta,
            scaler=ctx.scaler,
            routes_model=ctx.routes_model,
            rid2idx=ctx.rid2idx,
            df_hist=df_hist,
        )

        all_fc.append(df_fc_day)

        # append prediction vào hist để ngày sau dùng luôn cả data forecast
        tmp = df_fc_day.rename(columns={"PredictedVehicles": "Vehicles"})
        hist = pd.concat(
            [hist, tmp[["DateTime", "Vehicles", "RouteId"]]],
            ignore_index=True,
        )

    if not all_fc:
        return pd.DataFrame(), anchor_day_raw

    df_fc_raw = pd.concat(all_fc, ignore_index=True)
    return df_fc_raw, anchor_day_raw


def shift_forecast_to_today(df_fc_raw, anchor_day_raw, target_today=None):
    """
    Dịch cột DateTime của forecast sang 'hôm nay' mong muốn.

    Ý tưởng:
    - anchor_day_raw = ngày cuối cùng trong data thật (VD: 2018-10-31).
    - target_today = ngày "hôm nay" trên UI (VD: 2025-11-18).
    - Sau khi shift:
        - anchor_day_raw sẽ map về target_today,
        - nên các forecast (ngày sau anchor_day_raw) sẽ map về target_today+1, target_today+2, ...

    """
    if df_fc_raw is None or df_fc_raw.empty:
        return df_fc_raw

    if target_today is None:
        target_today = pd.Timestamp.today().normalize()

    if anchor_day_raw is None:
        # không biết anchor, thôi không dịch
        return df_fc_raw

    delta = target_today - anchor_day_raw

    df_shifted = df_fc_raw.copy()
    df_shifted["DateTime"] = df_shifted["DateTime"] + delta
    return df_shifted

# -------------------------------------------------
# LSTM: dùng history LOOKBACK giờ trước base_date
# -------------------------------------------------
def forecast_lstm(
    route_id,
    base_date,
    model,       # LSTM model (traffic_lstm.keras) đã load
    meta: dict,  # seq_meta.json
    scaler,      # vehicles_scaler.pkl
    routes_model,  # list routes trong meta
    rid2idx,       # dict route → index (có thể không dùng)
    df_hist: pd.DataFrame,
):
    """
    Dự báo 24h bằng LSTM, nếu thiếu history / lỗi → fallback Baseline.

    df_hist: lịch sử đã được app load sẵn (từ parquet), gồm các cột:
             DateTime, RouteId, Vehicles (ít nhất)
    """
    LOOKBACK = int(meta.get("LOOKBACK", 168))
    HORIZON = int(meta.get("HORIZON", 24))
    routes = list(meta.get("routes", routes_model))
    n_routes = len(routes)

    base_dt = pd.to_datetime(base_date)

    # ---- Lọc đúng route & resample 1h ----
    if df_hist is None or df_hist.empty:
        print(f"[LSTM] No df_hist passed for {route_id} → fallback Baseline.")
        return _baseline_forecast(route_id, base_date, HORIZON, hist_df=None)

    g = df_hist[df_hist["RouteId"].astype(str) == str(route_id)].copy()
    if g.empty:
        print(f"[LSTM] No history rows for {route_id} in df_hist → fallback Baseline.")
        return _baseline_forecast(route_id, base_date, HORIZON, hist_df=None)

    g["DateTime"] = pd.to_datetime(g["DateTime"], errors="coerce")
    g["Vehicles"] = pd.to_numeric(g["Vehicles"], errors="coerce")
    g = g.dropna(subset=["DateTime", "Vehicles"])

    if g.empty:
        print(f"[LSTM] History became empty after cleaning for {route_id} → fallback Baseline.")
        return _baseline_forecast(route_id, base_date, HORIZON, hist_df=None)

    g = (
        g.set_index("DateTime")
        .resample("1h")["Vehicles"]       # dùng "1h" để tránh FutureWarning
        .mean()
        .dropna()
        .reset_index()
        .sort_values("DateTime")
    )

    # ---- Lấy đúng LOOKBACKh trước base_dt ----
    g_hist = g[g["DateTime"] < base_dt].tail(LOOKBACK)

    print(
        f"[LSTM] {route_id} hist rows={len(g_hist)} from "
        f"{g_hist['DateTime'].min()} → {g_hist['DateTime'].max()}"
    )

    if len(g_hist) < LOOKBACK:
        # Không đủ history → dùng Baseline trên history hiện có
        print(
            f"[LSTM] Route {route_id}: only {len(g_hist)}h (<{LOOKBACK}h) → fallback Baseline."
        )
        return _baseline_forecast(route_id, base_date, HORIZON, hist_df=g_hist)

    # ---- Chuẩn bị input cho LSTM giống lúc train ----
    v_raw = g_hist["Vehicles"].values.astype(float)
    v_scaled = scaler.transform(v_raw.reshape(-1, 1)).reshape(-1)  # (LOOKBACK,)

    dt_hist = g_hist["DateTime"]
    tf_feats = time_feats(dt_hist)  # (LOOKBACK, 4)

    onehot = np.zeros((LOOKBACK, n_routes), dtype=np.float32)
    try:
        rid_idx = routes.index(str(route_id))
        onehot[:, rid_idx] = 1.0
    except ValueError:
        print(f"[LSTM] route {route_id} not in meta.routes → fallback Baseline.")
        return _baseline_forecast(route_id, base_date, HORIZON, hist_df=g_hist)

    X = np.concatenate(
        [v_scaled.reshape(-1, 1), tf_feats, onehot], axis=1
    )  # (LOOKBACK, 1+4+n_routes)
    X = X[np.newaxis, ...]  # (1, LOOKBACK, features)

    # ---- Predict bằng LSTM ----
    if model is None:
        print(f"[LSTM] model is None for {route_id} → fallback Baseline.")
        return _baseline_forecast(route_id, base_date, HORIZON, hist_df=g_hist)

    try:
        y_scaled = model.predict(X, verbose=0).reshape(-1, 1)  # (HORIZON, 1)
        y = scaler.inverse_transform(y_scaled).reshape(-1)     # (HORIZON,)
    except Exception as e:
        print(f"[LSTM] ❌ predict error for {route_id}: {e} → fallback Baseline.")
        return _baseline_forecast(route_id, base_date, HORIZON, hist_df=g_hist)

    next_hours = pd.date_range(base_dt, periods=HORIZON, freq="H")

    df_fc = pd.DataFrame(
        {
            "DateTime": next_hours,
            "RouteId": route_id,
            "PredictedVehicles": y,
        }
    )
    return df_fc, "LSTM"