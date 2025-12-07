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
# Baseline fallback (khi GRU/RNN không dùng được)
# -------------------------------------------------
def _baseline_forecast(route_id, base_date, horizon, hist_df: pd.DataFrame | None = None):
    """
    Fallback đơn giản khi seq model không chạy được.

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
    Dự báo HORIZON giờ bằng GRU, nếu thiếu history / lỗi → fallback Baseline.

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
        # đảm bảo không âm vì là lưu lượng xe
        y = np.maximum(y, 0.0)

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


# -------------------------------------------------
# RNN: dùng history LOOKBACK giờ trước base_date
# -------------------------------------------------
def forecast_rnn(
    route_id,
    base_date,
    model,       # RNN model (traffic_rnn_seq.keras) đã load ở app / model_manager
    meta: dict,  # seq_meta.json
    scaler,      # vehicles_scaler.pkl
    routes_model,  # list routes trong meta
    rid2idx,       # dict route → index (có thể không dùng)
    df_hist: pd.DataFrame,
):
    """
    Dự báo HORIZON giờ bằng RNN, pipeline giống GRU.
    Nếu thiếu history / lỗi → fallback Baseline.
    """
    LOOKBACK = int(meta.get("LOOKBACK", 168))
    HORIZON = int(meta.get("HORIZON", 24))
    routes = list(meta.get("routes", routes_model))
    n_routes = len(routes)

    base_dt = pd.to_datetime(base_date)

    # ---- Lọc đúng route & resample 1h ----
    if df_hist is None or df_hist.empty:
        print(f"[RNN] No df_hist passed for {route_id} → fallback Baseline.")
        return _baseline_forecast(route_id, base_date, HORIZON, hist_df=None)

    g = df_hist[df_hist["RouteId"].astype(str) == str(route_id)].copy()
    if g.empty:
        print(f"[RNN] No history rows for {route_id} in df_hist → fallback Baseline.")
        return _baseline_forecast(route_id, base_date, HORIZON, hist_df=None)

    g["DateTime"] = pd.to_datetime(g["DateTime"], errors="coerce")
    g["Vehicles"] = pd.to_numeric(g["Vehicles"], errors="coerce")
    g = g.dropna(subset=["DateTime", "Vehicles"])

    if g.empty:
        print(f"[RNN] History became empty after cleaning for {route_id} → fallback Baseline.")
        return _baseline_forecast(route_id, base_date, HORIZON, hist_df=None)

    g = (
        g.set_index("DateTime")
        .resample("1h")["Vehicles"]
        .mean()
        .dropna()
        .reset_index()
        .sort_values("DateTime")
    )

    # ---- Lấy đúng LOOKBACKh trước base_dt ----
    g_hist = g[g["DateTime"] < base_dt].tail(LOOKBACK)

    print(
        f"[RNN] {route_id} hist rows={len(g_hist)} from "
        f"{g_hist['DateTime'].min()} → {g_hist['DateTime'].max()}"
    )

    if len(g_hist) < LOOKBACK:
        print(
            f"[RNN] Route {route_id}: only {len(g_hist)}h (<{LOOKBACK}h) → fallback Baseline."
        )
        return _baseline_forecast(route_id, base_date, HORIZON, hist_df=g_hist)

    # ---- Chuẩn bị input cho RNN giống lúc train ----
    v_raw = g_hist["Vehicles"].values.astype(float)
    v_scaled = scaler.transform(v_raw.reshape(-1, 1)).reshape(-1)

    dt_hist = g_hist["DateTime"]
    tf_feats = time_feats(dt_hist)

    onehot = np.zeros((LOOKBACK, n_routes), dtype=np.float32)
    try:
        rid_idx = routes.index(str(route_id))
        onehot[:, rid_idx] = 1.0
    except ValueError:
        print(f"[RNN] route {route_id} not in meta.routes → fallback Baseline.")
        return _baseline_forecast(route_id, base_date, HORIZON, hist_df=g_hist)

    X = np.concatenate(
        [v_scaled.reshape(-1, 1), tf_feats, onehot], axis=1
    )
    X = X[np.newaxis, ...]

    if model is None:
        print(f"[RNN] model is None for {route_id} → fallback Baseline.")
        return _baseline_forecast(route_id, base_date, HORIZON, hist_df=g_hist)

    try:
        y_scaled = model.predict(X, verbose=0).reshape(-1, 1)
        y = scaler.inverse_transform(y_scaled).reshape(-1)
        # đảm bảo không âm vì là lưu lượng xe
        y = np.maximum(y, 0.0)

    except Exception as e:
        print(f"[RNN] ❌ predict error for {route_id}: {e} → fallback Baseline.")
        return _baseline_forecast(route_id, base_date, HORIZON, hist_df=g_hist)

    next_hours = pd.date_range(base_dt, periods=HORIZON, freq="H")

    df_fc = pd.DataFrame(
        {
            "DateTime": next_hours,
            "RouteId": route_id,
            "PredictedVehicles": y,
        }
    )
    return df_fc, "RNN"


# -------------------------------------------------
# Forecast tuần sau NGÀY CUỐI CÙNG trong data thật
# -------------------------------------------------
def forecast_week_after_last_point(route_id, city, zone, ctx, n_days=7, model_type="GRU"):
    """
    Dùng seq model (GRU/RNN) để forecast n_days (mặc định 7)
    sau NGÀY CUỐI CÙNG trong dữ liệu thật.

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

    model_type_norm = (model_type or "GRU").upper()
    use_rnn = model_type_norm == "RNN" and getattr(ctx, "rnn_model", None) is not None

    if model_type_norm == "RNN" and not use_rnn:
        print("[forecast_week] RNN được chọn nhưng ctx.rnn_model is None → dùng GRU.")

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

        if use_rnn:
            df_fc_day, model_used = forecast_rnn(
                route_id=route_id,
                base_date=base_date,
                model=ctx.rnn_model,
                meta=ctx.meta,
                scaler=ctx.scaler,
                routes_model=ctx.routes_model,
                rid2idx=ctx.rid2idx,
                df_hist=df_hist,
            )
        else:
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


def shift_forecast_to_today(
    df_fc_raw,
    anchor_day_raw,
    target_today=None,
    *,
    drop_past_hours: bool = True,
):
    """
    Dịch cột DateTime của forecast sang 'hôm nay' mong muốn.

    Ý tưởng mới:
    - anchor_day_raw = ngày cuối cùng trong data thật (VD: 2018-10-31).
    - Dữ liệu forecast bắt đầu từ anchor_day_raw + 1 ngày (00:00) → map về *hôm nay*.
    - target_today = ngày "hôm nay" trên UI (VD: 2025-11-18).
    - Sau khi shift:
        - anchor_day_raw + 1 → target_today (đầu ngày)
        - anchor_day_raw + 2 → target_today + 1, ...
    - Nếu drop_past_hours=True: bỏ các mốc thời gian đã qua (chỉ giữ từ hiện tại → 7 ngày tới).
    """
    if df_fc_raw is None or df_fc_raw.empty:
        return df_fc_raw

    if target_today is None:
        target_today = pd.Timestamp.today().normalize()

    if anchor_day_raw is None:
        # không biết anchor, thôi không dịch
        return df_fc_raw

    # Forecast luôn bắt đầu từ ngày sau anchor → map ngày đó về target_today
    base_forecast_day = pd.to_datetime(anchor_day_raw).normalize() + pd.Timedelta(days=1)
    delta = target_today - base_forecast_day

    df_shifted = df_fc_raw.copy()
    df_shifted["DateTime"] = pd.to_datetime(df_shifted["DateTime"], errors="coerce")
    df_shifted = df_shifted.dropna(subset=["DateTime"])
    df_shifted["DateTime"] = df_shifted["DateTime"] + delta

    if drop_past_hours:
        now_floor = pd.Timestamp.now().floor("H")
        df_shifted = df_shifted[df_shifted["DateTime"] >= now_floor]

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
        # đảm bảo không âm vì là lưu lượng xe
        y = np.maximum(y, 0.0)

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