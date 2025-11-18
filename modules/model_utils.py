# modules/model_utils.py
# Tiện ích dự báo cho GRU + baseline

from pathlib import Path  # nếu không dùng có thể xoá luôn
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
