#!/usr/bin/env python3
# scripts/train_lstm_from_parquet.py
#
# Huấn luyện model LSTM dự báo traffic 24h (HORIZON) dựa trên 168h (LOOKBACK).
# Multi-route: 1 model dùng chung cho tất cả RouteId trong city/zone.
#
# Output cho mỗi "family" (I94, Seattle_FremontBridge, ...):
#   model/<FAMILY>/traffic_lstm.keras
#   model/<FAMILY>/lstm_meta.json
#   model/<FAMILY>/vehicles_scaler.pkl (chỉ tạo nếu chưa tồn tại)
#
# Lưu ý:
#   - Nếu đã có GRU cho family này, script sẽ dùng chung scaler
#     từ vehicles_scaler.pkl hiện có, và KHÔNG overwrite file đó.

import os
import sys
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

# ----------------- PATH & IMPORT PROJECT MODULES -----------------

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from modules.data_loader import load_slice  # type: ignore


# ----------------- CONFIG DEFAULT -----------------
DEFAULT_LOOKBACK = 168
DEFAULT_HORIZON = 24
DEFAULT_EPOCHS = 30


# ----------------- HELPER: time_feats -----------------
def time_feats(dt) -> np.ndarray:
    # đảm bảo dt là DatetimeIndex
    if not isinstance(dt, pd.DatetimeIndex):
        dt = pd.DatetimeIndex(dt)

    hour = dt.hour.values
    dow = dt.dayofweek.values

    hour_sin = np.sin(2 * np.pi * hour / 24.0)
    hour_cos = np.cos(2 * np.pi * hour / 24.0)
    dow_sin = np.sin(2 * np.pi * dow / 7.0)
    dow_cos = np.cos(2 * np.pi * dow / 7.0)

    return np.c_[hour_sin, hour_cos, dow_sin, dow_cos].astype(np.float32)



# ----------------- HELPER: xác định model_dir cho family -----------------
def get_model_dir_for_family(city: str, zone: str | None) -> Path:
    """
    Tạo ra thư mục model cho 1 family, theo logic tương tự model_manager._detect_model_dir
    nhưng dành riêng cho TRAIN (không check tồn tại file model).
    """
    base = Path(ROOT_DIR) / "model"

    city_str = (city or "").strip()
    zone_str = (zone or "").strip() if zone else None

    # Special-case I94 như GRU
    if city_str.lower() == "minneapolis" or (zone_str and zone_str.upper() == "I94"):
        family = "I94"
    elif zone_str and zone_str != "(All)":
        family = f"{city_str}_{zone_str}".replace(" ", "_")
    elif city_str:
        family = city_str.replace(" ", "_")
    else:
        family = "default"

    model_dir = base / family
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"[LSTM-TRAIN] Using model_dir = {model_dir}")
    return model_dir


# ----------------- BUILD DATASET -----------------
def build_dataset_for_routes(
    df_all: pd.DataFrame,
    routes: List[str],
    lookback: int,
    horizon: int,
    scaler: StandardScaler,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Tạo X, y cho tất cả route.

    df_all: DataFrame gồm DateTime, RouteId, Vehicles.
    routes: list các RouteId cần train.
    scaler: đã được fit (với Vehicles).
    """
    X_list = []
    y_list = []

    n_routes = len(routes)
    route2idx = {rid: idx for idx, rid in enumerate(routes)}

    for rid in routes:
        g = df_all[df_all["RouteId"].astype(str) == str(rid)].copy()
        if g.empty:
            print(f"[LSTM-TRAIN] Route {rid}: no rows -> skip")
            continue

        g["DateTime"] = pd.to_datetime(g["DateTime"], errors="coerce")
        g["Vehicles"] = pd.to_numeric(g["Vehicles"], errors="coerce")
        g = g.dropna(subset=["DateTime", "Vehicles"])
        if g.empty:
            print(f"[LSTM-TRAIN] Route {rid}: empty after cleaning -> skip")
            continue

        # Resample hourly
        g = (
            g.set_index("DateTime")["Vehicles"]
            .resample("1H")
            .mean()
            .dropna()
            .reset_index()
            .sort_values("DateTime")
        )

        if len(g) < lookback + horizon:
            print(
                f"[LSTM-TRAIN] Route {rid}: not enough hourly samples ({len(g)}) "
                f"for LOOKBACK={lookback}, HORIZON={horizon}"
            )
            continue

        v_raw = g["Vehicles"].values.astype(float)
        v_scaled_all = scaler.transform(v_raw.reshape(-1, 1)).reshape(-1)  # (T,)
        dt_all = g["DateTime"]

        # Duyệt các cửa sổ slide
        for i in range(lookback, len(g) - horizon):
            past_slice = slice(i - lookback, i)
            fut_slice = slice(i, i + horizon)

            v_hist = v_scaled_all[past_slice]  # (lookback,)
            dt_hist = dt_all.iloc[past_slice]

            # time features cho lịch sử
            tf_hist = time_feats(pd.DatetimeIndex(dt_hist))

            # one-hot cho route
            onehot = np.zeros((lookback, n_routes), dtype=np.float32)
            rid_idx = route2idx[rid]
            onehot[:, rid_idx] = 1.0

            # build feature: [v_hist, tf_hist(4), onehot(n_routes)]
            X_i = np.concatenate(
                [
                    v_hist.reshape(-1, 1),
                    tf_hist,
                    onehot,
                ],
                axis=1,
            )  # (lookback, 1 + 4 + n_routes)

            # target: horizon bước phía trước (đã scaled)
            y_i = v_scaled_all[fut_slice]  # (horizon,)

            X_list.append(X_i)
            y_list.append(y_i)

    if not X_list:
        raise RuntimeError("[LSTM-TRAIN] Không tạo được sample nào (X_list empty).")

    X = np.stack(X_list, axis=0).astype(np.float32)  # (N, lookback, n_feats)
    y = np.stack(y_list, axis=0).astype(np.float32)  # (N, horizon)
    print(f"[LSTM-TRAIN] Dataset X shape = {X.shape}, y shape = {y.shape}")
    return X, y, routes


# ----------------- MAIN TRAINING LOGIC -----------------
def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True, help="City name (ví dụ: Minneapolis)")
    parser.add_argument(
        "--zone", required=True, help="Zone name (ví dụ: I94, FremontBridge, ...)"
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=DEFAULT_LOOKBACK,
        help=f"LOOKBACK (default={DEFAULT_LOOKBACK})",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_HORIZON,
        help=f"HORIZON (default={DEFAULT_HORIZON})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Số epoch train LSTM (default={DEFAULT_EPOCHS})",
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size (default=128)"
    )
    args = parser.parse_args()

    city = args.city
    zone = None if args.zone == "(All)" else args.zone
    lookback = int(args.lookback)
    horizon = int(args.horizon)

    print(
        f"[LSTM-TRAIN] city={city}, zone={zone}, "
        f"LOOKBACK={lookback}, HORIZON={horizon}, EPOCHS={args.epochs}"
    )

    # ---- 1) Load full data (tất cả route) ----
    df_full = load_slice(
        city=city,
        zone=zone,
        routes=None,  # tất cả route thuộc city/zone
        start_dt=None,
        end_dt=None,
    )

    if df_full is None or df_full.empty:
        raise RuntimeError("[LSTM-TRAIN] df_full empty – không có dữ liệu để train.")

    df_full = df_full.copy()
    if "DateTime" not in df_full.columns or "Vehicles" not in df_full.columns:
        raise RuntimeError("[LSTM-TRAIN] df_full thiếu DateTime hoặc Vehicles")

    # Lấy list routes distinct
    routes = sorted(df_full["RouteId"].astype(str).unique().tolist())
    print(f"[LSTM-TRAIN] Found {len(routes)} routes: {routes}")

    # ---- 2) Model dir cho family ----
    model_dir = get_model_dir_for_family(city, zone)

    # ---- 3) Chuẩn bị scaler Vehicles ----
    scaler_path = model_dir / "vehicles_scaler.pkl"
    if scaler_path.exists():
        print(f"[LSTM-TRAIN] Reusing existing scaler from {scaler_path}")
        scaler: StandardScaler = joblib.load(scaler_path)
    else:
        print("[LSTM-TRAIN] Fitting new StandardScaler for Vehicles...")
        df_clean = df_full.copy()
        df_clean["Vehicles"] = pd.to_numeric(df_clean["Vehicles"], errors="coerce")
        df_clean = df_clean.dropna(subset=["Vehicles"])
        if df_clean.empty:
            raise RuntimeError("[LSTM-TRAIN] Không có Vehicles để fit scaler.")
        scaler = StandardScaler()
        scaler.fit(df_clean["Vehicles"].values.reshape(-1, 1))
        joblib.dump(scaler, scaler_path)
        print(f"[LSTM-TRAIN] Saved new scaler to {scaler_path}")

    # ---- 4) Build dataset X, y ----
    X, y, routes_final = build_dataset_for_routes(
        df_all=df_full,
        routes=routes,
        lookback=lookback,
        horizon=horizon,
        scaler=scaler,
    )

    n_samples, _, n_feats = X.shape
    print(f"[LSTM-TRAIN] n_samples={n_samples}, n_feats={n_feats}")

    # ---- 5) Build LSTM model ----
    inputs = tf.keras.Input(shape=(lookback, n_feats))
    x = layers.LSTM(64, return_sequences=True)(inputs)
    x = layers.LSTM(64)(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(horizon)(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="traffic_lstm_seq2seq")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="mse",
    )

    model.summary(print_fn=lambda s: print("[LSTM-TRAIN] " + s))

    # ---- 6) Train ----
    cb_early = callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )

    history = model.fit(
        X,
        y,
        epochs=args.epochs,          # default = 30
        batch_size=args.batch_size,
        validation_split=0.1,
        shuffle=True,
        callbacks=[cb_early],
        verbose=1,
    )

    print("[LSTM-TRAIN] Training done.")

    # ---- 7) Save model & meta ----
    model_path = model_dir / "traffic_lstm.keras"
    model.save(model_path)
    print(f"[LSTM-TRAIN] Saved LSTM model to {model_path}")

    meta = {
        "LOOKBACK": lookback,
        "HORIZON": horizon,
        "routes": routes_final,
        "n_features": n_feats,
    }
    meta_path = model_dir / "lstm_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[LSTM-TRAIN] Saved LSTM meta to {meta_path}")

    print("[LSTM-TRAIN] ✅ DONE")


if __name__ == "__main__":
    main()
