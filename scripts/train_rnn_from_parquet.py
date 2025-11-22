# scripts/train_rnn_from_parquet.py
"""
Train RNN seq2seq cho traffic volume (đa tuyến) từ parquet,
dùng chung scaler + meta (LOOKBACK, HORIZON, routes) với GRU hiện tại nếu có.

Usage (ví dụ):

# I-94 (Minneapolis, zone I94)
python scripts/train_rnn_from_parquet.py --city Minneapolis --zone I94

# Fremont Bridge (Seattle)
python scripts/train_rnn_from_parquet.py --city Seattle --zone FremontBridge

Có thể override LOOKBACK/HORIZON nếu meta chưa tồn tại:
python scripts/train_rnn_from_parquet.py --city Seattle --zone FremontBridge --lookback 168 --horizon 24
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
import os
import sys

# Thêm project root (TrafficApp) vào sys.path
# __file__  => .../TrafficApp/scripts/train_rnn_from_parquet.py
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from modules.data_loader import load_slice
from modules.model_utils import time_feats

MODEL_ROOT = Path("model")

#
# def time_feats(dt: pd.DatetimeIndex) -> np.ndarray:
#     """Sin/cos hour-of-day + day-of-week, shape (N, 4)."""
#     hour = dt.hour.values
#     dow = dt.dayofweek.values
#
#     feats = np.c_[
#         np.sin(2 * np.pi * hour / 24),
#         np.cos(2 * np.pi * hour / 24),
#         np.sin(2 * np.pi * dow / 7),
#         np.cos(2 * np.pi * dow / 7),
#     ].astype(np.float32)
#     return feats


def build_rnn_model(lookback: int, n_feats: int, horizon: int) -> tf.keras.Model:
    """SimpleRNN seq2seq tương tự GRU nhưng dùng SimpleRNN layer."""
    inputs = layers.Input(shape=(lookback, n_feats))

    x = layers.SimpleRNN(64, return_sequences=True)(inputs)
    x = layers.SimpleRNN(64)(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(horizon)(x)

    model = models.Model(inputs, outputs, name="traffic_rnn_seq")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="mse",
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True, help="Tên city, ví dụ 'Minneapolis' hoặc 'Seattle'")
    parser.add_argument("--zone", required=True, help="Zone, ví dụ 'I94' hoặc 'FremontBridge'")
    parser.add_argument("--lookback", type=int, default=168, help="Số giờ history dùng cho mỗi sample")
    parser.add_argument("--horizon", type=int, default=24, help="Số giờ dự báo mỗi sample")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    city = args.city
    zone = args.zone

    model_dir = MODEL_ROOT / city / zone
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Training RNN for city={city}, zone={zone}")
    print(f"[INFO] Model dir: {model_dir}")

    # --------------------------------------------------
    # 1) Load full history cho city/zone này
    # --------------------------------------------------
    df = load_slice(
        city=city,
        zone=zone,
        routes=None,         # lấy tất cả RouteId
        start_dt=None,
        end_dt=None,
    )

    if df.empty:
        raise RuntimeError(f"[ERROR] Không có dữ liệu cho city={city}, zone={zone}")

    df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
    df = df.dropna(subset=["DateTime"])
    df["RouteId"] = df["RouteId"].astype(str)

    all_routes = sorted(df["RouteId"].unique().tolist())
    print(f"[INFO] Found routes: {all_routes}")

    # --------------------------------------------------
    # 2) Meta (LOOKBACK, HORIZON, routes)
    #    - Nếu đã có seq_meta.json (GRU đã train) → reuse
    #    - Nếu chưa → tạo mới.
    # --------------------------------------------------
    meta_path = model_dir / "seq_meta.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        lookback = int(meta.get("LOOKBACK", args.lookback))
        horizon = int(meta.get("HORIZON", args.horizon))
        routes_meta = meta.get("routes", all_routes)
        print(f"[INFO] Using existing meta: LOOKBACK={lookback}, HORIZON={horizon}, routes={routes_meta}")
        routes = routes_meta
    else:
        lookback = int(args.lookback)
        horizon = int(args.horizon)
        routes = all_routes
        meta = {
            "LOOKBACK": lookback,
            "HORIZON": horizon,
            "routes": routes,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"[INFO] Created new meta at {meta_path}: {meta}")

    rid2idx = {rid: i for i, rid in enumerate(routes)}

    # --------------------------------------------------
    # 3) Scaler cho Vehicles (chung với GRU/MLP nếu đã tồn tại)
    # --------------------------------------------------
    scaler_path = model_dir / "vehicles_scaler.pkl"
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        print(f"[INFO] Loaded existing scaler from {scaler_path}")
    else:
        vehicles_all = df["Vehicles"].astype(float).values.reshape(-1, 1)
        scaler = StandardScaler()
        scaler.fit(vehicles_all)
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        print(f"[INFO] Fitted and saved new scaler to {scaler_path}")

    # --------------------------------------------------
    # 4) Resample 1H cho từng route, build sequences
    # --------------------------------------------------
    samples_X = []
    samples_y = []

    for rid in routes:
        g = df[df["RouteId"] == rid].copy()
        if g.empty:
            continue

        g = (
            g.set_index("DateTime")
             .resample("1H")["Vehicles"]
             .mean()
             .dropna()
             .reset_index()
        )
        if len(g) < lookback + horizon + 1:
            print(f"[WARN] Route {rid}: không đủ dữ liệu (len={len(g)}) → skip")
            continue

        veh = g["Vehicles"].astype(float).values.reshape(-1, 1)
        veh_scaled = scaler.transform(veh)   # (T, 1)
        dt_index = g["DateTime"]
        tf_time = time_feats(dt_index)       # (T, 4)

        onehot = np.zeros((len(g), len(routes)), dtype=np.float32)
        j = rid2idx[rid]
        onehot[:, j] = 1.0

        feats = np.concatenate([veh_scaled, tf_time, onehot], axis=1)  # (T, 1+4+n_routes)
        T = feats.shape[0]

        for t in range(lookback, T - horizon):
            X_win = feats[t - lookback:t, :]  # (lookback, n_feats)
            # y: scaled Vehicles cho horizon tiếp theo
            y_win_scaled = scaler.transform(veh[t:t + horizon]).reshape(-1)  # (horizon,)

            samples_X.append(X_win)
            samples_y.append(y_win_scaled)

    if not samples_X:
        raise RuntimeError("[ERROR] Không build được sample nào. Kiểm tra lại LOOKBACK/HORIZON và data length.")

    X = np.stack(samples_X, axis=0)  # (N, lookback, n_feats)
    y = np.stack(samples_y, axis=0)  # (N, horizon)

    print(f"[INFO] Built training set: X={X.shape}, y={y.shape}")
    n_feats = X.shape[-1]

    # --------------------------------------------------
    # 5) Build & train RNN
    # --------------------------------------------------
    model = build_rnn_model(lookback, n_feats, horizon)
    model.summary()

    model.fit(
        X,
        y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.1,
        verbose=1,
    )

    out_path = model_dir / "traffic_rnn_seq.keras"
    model.save(out_path)
    print(f"[INFO] Saved RNN model to {out_path}")


if __name__ == "__main__":
    main()
