# scripts/train_mlp_from_parquet.py
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

from modules.data_loader import load_all

MODEL_DIR = Path("model")
CITY = "i94"
ZONE = "I94_main"


def time_feats(dt: pd.Series):
    dt = pd.to_datetime(dt)
    hour = dt.dt.hour
    dow = dt.dt.dayofweek
    return np.c_[
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * dow / 7),
        np.cos(2 * np.pi * dow / 7),
    ].astype(np.float32)


def main():
    df = load_all(CITY, ZONE)
    if df.empty:
        raise RuntimeError("No processed data found. Run ingest_metro_i94.py first.")

    scaler_path = MODEL_DIR / "vehicles_scaler.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError("vehicles_scaler.pkl not found. Run prep_seq_light.py first.")

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df = df.sort_values("DateTime")

    X = time_feats(df["DateTime"])
    y_scaled = scaler.transform(df[["Vehicles"]]).astype(np.float32)

    inputs = layers.Input(shape=(X.shape[1],))
    x = layers.Dense(32, activation="relu")(inputs)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(1)(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")

    print("[INFO] Training MLP fallback model...")
    model.fit(X, y_scaled, epochs=10, batch_size=128, validation_split=0.1, verbose=1)

    out_path = MODEL_DIR / "traffic_mlp.h5"
    model.save(out_path)
    print(f"[INFO] Saved MLP model to {out_path}")


if __name__ == "__main__":
    main()
