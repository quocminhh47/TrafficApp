#!/usr/bin/env python3
# scripts/train_lstm_from_parquet.py
# Huấn luyện model LSTM dự báo traffic 24h dựa trên 168h lịch sử.

import os
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
import joblib


# ---------------------------
# CONFIG
# ---------------------------
DATA_ROOT = Path("data/processed_ds")
MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True, parents=True)

LOOKBACK = 168   # 7 ngày
HORIZON = 24     # 24 giờ tiếp theo

SCALER_PATH = MODEL_DIR / "vehicles_scaler.pkl"
SEQ_MODEL_PATH = MODEL_DIR / "traffic_lstm.keras"
SEQ_META_PATH = MODEL_DIR / "lstm_meta.json"


def time_feats(dt: pd.Series) -> np.ndarray:
    """Sin/cos cho hour & day-of-week."""
    dt = pd.to_datetime(dt)
    hour = dt.dt.hour
    dow = dt.dt.dayofweek
    feats = np.c_[
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * dow / 7),
        np.cos(2 * np.pi * dow / 7),
    ].astype(np.float32)
    return feats


def load_all_parquet() -> pd.DataFrame:
    """Load tất cả parquet trong data/processed_ds/**."""
    if not DATA_ROOT.is_dir():
        raise FileNotFoundError(f"{DATA_ROOT} không tồn tại. Hãy chạy prepare_from_raw trước.")

    frames = []
    for root, dirs, files in os.walk(DATA_ROOT):
        for fname in files:
            if not fname.endswith(".parquet"):
                continue
            fpath = Path(root) / fname
            try:
                df = pd.read_parquet(fpath)
                if not {"DateTime", "RouteId", "Vehicles"} <= set(df.columns):
                    continue
                df = df[["DateTime", "RouteId", "Vehicles"]].copy()
                df["DateTime"] = pd.to_datetime(df["DateTime"], utc=True, errors="coerce")
                df = df.dropna(subset=["DateTime", "Vehicles"])
                frames.append(df)
            except Exception as e:
                print(f"[WARN] skip {fpath}: {e}")

    if not frames:
        raise RuntimeError("Không load được parquet nào từ data/processed_ds/**.parquet")

    all_df = pd.concat(frames, ignore_index=True)
    all_df["RouteId"] = all_df["RouteId"].astype(str)
    return all_df


def main():
    print("[INFO] Loading parquet...")
    df = load_all_parquet()

    routes = sorted(df["RouteId"].dropna().unique().tolist())
    print(f"[INFO] {len(routes)} routes: {routes}")

    # ---------------------------
    # Load / fit scaler
    # ---------------------------
    if SCALER_PATH.exists():
        print(f"[INFO] Load existing scaler from {SCALER_PATH}")
        scaler: StandardScaler = joblib.load(SCALER_PATH)
    else:
        print("[INFO] vehicles_scaler.pkl không tồn tại → fit mới.")
        scaler = StandardScaler()
        scaler.fit(df["Vehicles"].values.reshape(-1, 1))
        joblib.dump(scaler, SCALER_PATH)
        print(f"[INFO] Saved new scaler to {SCALER_PATH}")

    # ---------------------------
    # Build sequences X, y
    # ---------------------------
    X_list = []
    y_list = []

    for rid in routes:
        g = df[df["RouteId"] == rid].copy()
        if g.empty:
            continue

        g["DateTime"] = pd.to_datetime(g["DateTime"], utc=True, errors="coerce")
        g = (
            g.set_index("DateTime")
            .resample("1h")["Vehicles"]
            .mean()
            .dropna()
            .reset_index()
            .sort_values("DateTime")
        )

        if len(g) < LOOKBACK + HORIZON:
            print(f"[INFO] Route {rid}: not enough data ({len(g)}h) → skip")
            continue

        v_raw = g["Vehicles"].values.astype(float)
        v_scaled = scaler.transform(v_raw.reshape(-1, 1)).reshape(-1)
        dt = g["DateTime"]

        for i in range(LOOKBACK, len(g) - HORIZON):
            past_slice = slice(i - LOOKBACK, i)
            fut_slice = slice(i, i + HORIZON)

            past_v_scaled = v_scaled[past_slice]
            past_dt = dt.iloc[past_slice]

            time_features = time_feats(past_dt)  # (LOOKBACK, 4)

            # one-hot route
            #onehot = np.zeros((LOOKBACK, len(routes)), dtype=np.float32)
            #route_index = routes.index(rid)
            #onehot[:, route_index] = 1.0
            route_index = routes.index(rid)
            onehot = np.zeros((1, len(routes)), dtype=np.float32)
            onehot[0, route_index] = 1.0
            onehot = np.repeat(onehot, LOOKBACK, axis=0)

            features = np.concatenate(
                [past_v_scaled.reshape(-1, 1), time_features, onehot],
                axis=1
            )  # (LOOKBACK, 1 + 4 + n_routes)

            X_list.append(features)
            y_list.append(v_scaled[fut_slice])

        print(f"[SEQ] Route {rid}: built {len(X_list)} sequences so far.")

    if not X_list:
        raise RuntimeError("Không tạo được sequence nào cho LSTM.")

    X = np.stack(X_list).astype(np.float32)
    y = np.stack(y_list).astype(np.float32)

    num_features = X.shape[2]
    print(f"[INFO] X shape = {X.shape}  (samples, LOOKBACK, features={num_features})")
    print(f"[INFO] y shape = {y.shape}  (samples, HORIZON={HORIZON})")

    # ---------------------------
    # Train / Val split
    # ---------------------------
    n = X.shape[0]
    n_train = int(n * 0.8)
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:], y[n_train:]

    # ---------------------------
    # Build LSTM model
    # ---------------------------
    inp = layers.Input(shape=(LOOKBACK, num_features))

    x = layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.1, kernel_regularizer=tf.keras.regularizers.l2(1e-6))(inp)
    x = layers.LayerNormalization()(x)
    x = layers.LSTM(64, dropout=0.2, recurrent_dropout=0.1)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.1)(x)

    out = layers.Dense(HORIZON, activation="linear")(x)

    model = models.Model(inp, out, name="traffic_lstm")

    opt = optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)
    model.compile(optimizer=opt, loss="mse")

    model.summary()

    # ---------------------------
    # Callbacks
    # ---------------------------
    cb = [
        callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5
        ),
        callbacks.ModelCheckpoint(
            filepath=str(SEQ_MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    # ---------------------------
    # Train
    # ---------------------------
    print("[INFO] Training LSTM...")
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=256,
        verbose=2,
        callbacks=cb,
    )

    # Save final model
    model.save(str(SEQ_MODEL_PATH))
    print(f"[INFO] Saved LSTM model to {SEQ_MODEL_PATH}")

    # Save meta
    meta = {
        "LOOKBACK": LOOKBACK,
        "HORIZON": HORIZON,
        "routes": routes,
        "num_features": num_features,
    }
    with open(SEQ_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[INFO] Saved meta to {SEQ_META_PATH}")
    print("[DONE] LSTM training complete.")


if __name__ == "__main__":
    main()
