# modules/model_utils.py
# Tiện ích dự báo cho MLP + GRU

from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
import tensorflow as tf


MODEL_DIR = Path("model")
MLP_PATH = MODEL_DIR / "traffic_mlp.h5"        # mô hình MLP fallback
SEQ_MODEL_PATH = MODEL_DIR / "traffic_seq.keras"  # GRU được app load sẵn (không dùng ở đây)


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
# Lazy load MLP
# -------------------------------------------------
@lru_cache(maxsize=1)
def _load_mlp_model():
    if not MLP_PATH.exists():
        raise FileNotFoundError(f"MLP model not found at {MLP_PATH}")
    print(f"[MLP] ✅ loaded {MLP_PATH}")
    model = tf.keras.models.load_model(MLP_PATH)
    return model


# -------------------------------------------------
# Baseline: constant (cực kỳ đơn giản, chỉ fallback cuối)
# -------------------------------------------------
def forecast_baseline(route_id, base_date, horizon=24, const_value=100.0):
    """
    Baseline: dự báo hằng số (vd 100 Vehicles) cho horizon giờ.
    Dùng khi MLP/GRU đều fail.
    """
    base_dt = pd.to_datetime(base_date)
    next_hours = pd.date_range(base_dt, periods=horizon, freq="H")

    return pd.DataFrame(
        {
            "DateTime": next_hours,
            "RouteId": route_id,
            "PredictedVehicles": np.full(horizon, const_value, dtype=float),
        }
    ), "Baseline"


# -------------------------------------------------
# MLP: dùng time_feats + one-hot route
# -------------------------------------------------
# -------------------------------------------------
# MLP: dùng 4 time features, KHÔNG one-hot route
# -------------------------------------------------
def forecast_mlp(route_id, base_date, scaler, routes=None, horizon=24):
    """
    Dự báo bằng MLP:
    - Input cho MLP: 4 time features (sin/cos(hour), sin/cos(dow))
    - KHÔNG dùng one-hot route (vì hiện tại chỉ có 1 route I-94-WB)
    - scaler: vehicles_scaler.pkl (StandardScaler trên Vehicles)
    - horizon: số giờ muốn dự đoán (mặc định = 24)

    Trả về: (df_forecast, "MLP")
      df_forecast có cột: DateTime, RouteId, PredictedVehicles
    """
    try:
        mlp = _load_mlp_model()
    except Exception as e:
        print(f"[MLP] ❌ load error: {e} → fallback Baseline.")
        return forecast_baseline(route_id, base_date, horizon=horizon)

    base_dt = pd.to_datetime(base_date)
    next_hours = pd.date_range(base_dt, periods=horizon, freq="H")

    # 4 time features
    feats = time_feats(next_hours)  # (horizon, 4)

    try:
        # MLP output là Vehicles đã được scale
        y_scaled = mlp.predict(feats, verbose=0).reshape(-1, 1)  # (horizon, 1)
        # inverse_scale về Vehicles thực tế
        y = scaler.inverse_transform(y_scaled).reshape(-1)       # (horizon,)
    except Exception as e:
        print(f"[MLP] ❌ predict error: {e} → fallback Baseline.")
        return forecast_baseline(route_id, base_date, horizon=horizon)

    df_fc = pd.DataFrame(
        {
            "DateTime": next_hours,
            "RouteId": route_id,
            "PredictedVehicles": y,
        }
    )
    return df_fc, "MLP"

# -------------------------------------------------
# GRU: dùng history LOOKBACK giờ trước base_date
# -------------------------------------------------
def forecast_gru(
    route_id,
    base_date,
    model,       # GRU model (traffic_seq.keras) đã load ở app
    meta: dict,  # seq_meta.json
    scaler,      # vehicles_scaler.pkl
    routes_model,  # list routes trong meta
    rid2idx,       # dict route → index (có thể không dùng)
    df_hist: pd.DataFrame,
):
    """
    Dự báo 24h bằng GRU, nếu thiếu history / lỗi → fallback MLP, rồi Baseline.

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
        print(f"[GRU] No df_hist passed for {route_id} → fallback MLP.")
        return forecast_mlp(route_id, base_date, scaler, routes_model, horizon=HORIZON)

    g = df_hist[df_hist["RouteId"].astype(str) == str(route_id)].copy()
    if g.empty:
        print(f"[GRU] No history rows for {route_id} in df_hist → fallback MLP.")
        return forecast_mlp(route_id, base_date, scaler, routes_model, horizon=HORIZON)

    g["DateTime"] = pd.to_datetime(g["DateTime"], errors="coerce")
    g["Vehicles"] = pd.to_numeric(g["Vehicles"], errors="coerce")
    g = g.dropna(subset=["DateTime", "Vehicles"])

    if g.empty:
        print(f"[GRU] History became empty after cleaning for {route_id} → fallback MLP.")
        return forecast_mlp(route_id, base_date, scaler, routes_model, horizon=HORIZON)

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
        # Không đủ history → dùng MLP
        print(
            f"[GRU] Route {route_id}: only {len(g_hist)}h (<{LOOKBACK}h) → fallback MLP."
        )
        return forecast_mlp(route_id, base_date, scaler, routes_model, horizon=HORIZON)

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
        print(f"[GRU] route {route_id} not in meta.routes → fallback MLP.")
        return forecast_mlp(route_id, base_date, scaler, routes_model, horizon=HORIZON)

    X = np.concatenate(
        [v_scaled.reshape(-1, 1), tf_feats, onehot], axis=1
    )  # (LOOKBACK, 1+4+n_routes)
    X = X[np.newaxis, ...]  # (1, LOOKBACK, features)

    # ---- Predict bằng GRU ----
    try:
        y_scaled = model.predict(X, verbose=0).reshape(-1, 1)  # (HORIZON, 1)
        y = scaler.inverse_transform(y_scaled).reshape(-1)     # (HORIZON,)
    except Exception as e:
        print(f"[GRU] ❌ predict error for {route_id}: {e} → fallback MLP.")
        return forecast_mlp(route_id, base_date, scaler, routes_model, horizon=HORIZON)

    next_hours = pd.date_range(base_dt, periods=HORIZON, freq="H")

    df_fc = pd.DataFrame(
        {
            "DateTime": next_hours,
            "RouteId": route_id,
            "PredictedVehicles": y,
        }
    )
    return df_fc, "GRU"
