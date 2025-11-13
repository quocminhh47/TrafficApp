#!/usr/bin/env python
# scripts/train_mlp_light.py — fallback MLP model
import os, glob, pickle, json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

DATA_ROOT = Path("data/processed_ds")
MODEL_DIR = Path("model")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def time_feats(dt: pd.Series):
    dt = pd.to_datetime(dt)
    hour, dow = dt.dt.hour, dt.dt.dayofweek
    return np.c_[np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24),
                 np.sin(2*np.pi*dow/7), np.cos(2*np.pi*dow/7)].astype(np.float32)

frames = []
for f in glob.glob(str(DATA_ROOT / "*" / "*" / "*.parquet")):
    try:
        df = pd.read_parquet(f)
        if {"DateTime","Vehicles","RouteId"}.issubset(df.columns):
            df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
            df["Vehicles"] = pd.to_numeric(df["Vehicles"], errors="coerce")
            df = df.dropna(subset=["DateTime","Vehicles","RouteId"])
            frames.append(df[["DateTime","Vehicles","RouteId"]])
    except Exception as e:
        print(f"[WARN] skip {f}: {e}")
df = pd.concat(frames, ignore_index=True)
routes = sorted(df["RouteId"].astype(str).unique())
rid2idx = {r:i for i,r in enumerate(routes)}

feats = time_feats(df["DateTime"])
onehot = np.zeros((len(df), len(routes)), np.float32)
for r, i in rid2idx.items():
    onehot[df["RouteId"] == r, i] = 1.0
X = np.concatenate([feats, onehot], axis=1)

scaler = MinMaxScaler()
y = scaler.fit_transform(df[["Vehicles"]])

model = Sequential([
    Input(shape=(X.shape[1],)),
    Dense(64, activation="relu"),
    Dense(32, activation="relu"),
    Dense(1)
])
model.compile(optimizer=Adam(1e-3), loss="mse", metrics=["mae"])
model.fit(X, y, validation_split=0.1, epochs=15, batch_size=512, verbose=1)

model.save(MODEL_DIR / "traffic_mlp.h5")
pickle.dump(rid2idx, open(MODEL_DIR / "encoder.pkl", "wb"))
pickle.dump(scaler, open(MODEL_DIR / "vehicles_scaler.pkl", "wb"))
json.dump({"routes": routes}, open(MODEL_DIR / "mlp_meta.json", "w"), indent=2)

print("[DONE] MLP model retrained.")
