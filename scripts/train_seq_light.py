# scripts/train_seq_light.py
from pathlib import Path
import json

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

MODEL_DIR = Path("model")


def main():
    data_path = MODEL_DIR / "seq_train.npz"
    meta_path = MODEL_DIR / "seq_meta.json"

    if not data_path.exists() or not meta_path.exists():
        raise FileNotFoundError("seq_train.npz or seq_meta.json not found. Run prep_seq_light.py first.")

    print("[INFO] Loading training data...")
    npz = np.load(data_path)
    X = npz["X"]
    y = npz["y"]

    with open(meta_path, "r") as f:
        meta = json.load(f)

    LOOKBACK = int(meta["LOOKBACK"])
    HORIZON = int(meta["HORIZON"])

    print(f"[INFO] X shape={X.shape}, y shape={y.shape}, LOOKBACK={LOOKBACK}, HORIZON={HORIZON}")

    n_features = X.shape[-1]

    inputs = layers.Input(shape=(LOOKBACK, n_features))
    x = layers.GRU(64, return_sequences=False)(inputs)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(HORIZON)(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")

    model.summary()

    print("[INFO] Training GRU model...")
    model.fit(
        X,
        y,
        epochs=15,
        batch_size=64,
        validation_split=0.1,
        verbose=1,
    )

    out_path = MODEL_DIR / "traffic_seq.keras"
    model.save(out_path)
    print(f"[INFO] Saved GRU model to {out_path}")


if __name__ == "__main__":
    main()
