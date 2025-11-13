#!/usr/bin/env python
# scripts/train_seq_light.py — train GRU/LSTM sequence model
import os, json, pickle, glob
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam

DATA_ROOT = Path("data/processed_ds")
MODEL_DIR = Path("model")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

LOOKBACK = 168  # 7 ngày
HORIZON = 24

def time_feats(dt: pd.Series):
    dt = pd.to_datetime(dt)
    hour, dow = dt.dt.hour, dt.dt.dayofweek
    return np.c_[np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24),
                 np.sin(2*np.pi*dow/7), np.cos(2*np.pi*dow/7)].astype(np.float32)

def load_data():
    frames = []
    for f in glob.glob(str(DATA_ROOT / "*" / "*" / "*.parquet")):
        try:
            df = pd.read_parquet(f)
            if {"DateTime","Vehicles","RouteId"}.issubset(df.columns):
                df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce", utc=True)
                df["Vehicles"] = pd.to_numeric(df["Vehicles"], errors="coerce")
                df = df.dropna(subset=["DateTime","Vehicles","RouteId"])
                frames.append(df[["DateTime","Vehicles","RouteId"]])
        except Exception as e:
            print(f"[WARN] skip {f}: {e}")
    df = pd.concat(frames, ignore_index=True)
    df["RouteId"] = df["RouteId"].astype(str)
    return df

print("[INFO] Loading data...")
df = load_data()
routes = sorted(df["RouteId"].unique())
rid2idx = {r:i for i,r in enumerate(routes)}
print(f"[INFO] {len(routes)} routes loaded")

scaler = MinMaxScaler()
df["Vehicles"] = scaler.fit_transform(df[["Vehicles"]])

# --- Tạo chuỗi ---
X_list, Y_list = [], []
for rid, g in df.groupby("RouteId"):
    g = (
        g.set_index("DateTime")
        .resample("1h")["Vehicles"]
        .mean()
        .dropna()
        .reset_index()
    )
    g["RouteId"] = rid
    g["Vehicles"] = pd.to_numeric(g["Vehicles"], errors="coerce")
    g = g.dropna(subset=["Vehicles"])

    if len(g) < LOOKBACK + HORIZON: continue
    v = g["Vehicles"].values
    t = time_feats(g["DateTime"])
    one = np.zeros((len(g), len(routes)), np.float32)
    one[:, rid2idx[rid]] = 1
    arr = np.c_[v, t, one]
    for i in range(len(arr) - LOOKBACK - HORIZON):
        X_list.append(arr[i:i+LOOKBACK])
        Y_list.append(v[i+LOOKBACK:i+LOOKBACK+HORIZON])
X, y = np.stack(X_list), np.stack(Y_list)
print(f"[INFO] Seq dataset: X={X.shape}, y={y.shape}")

# --- Model (GRU hoặc LSTM) ---
model = Sequential([
    Input(shape=(LOOKBACK, X.shape[-1])),
    GRU(64, return_sequences=False),
    Dense(HORIZON)
])
# (Nếu muốn LSTM: thay GRU bằng LSTM)
model.compile(optimizer=Adam(1e-3), loss="mse")
model.summary()

model.fit(X, y, validation_split=0.1, epochs=10, batch_size=64, verbose=1)

model.save(MODEL_DIR / "traffic_seq.keras")
pickle.dump(scaler, open(MODEL_DIR / "vehicles_scaler.pkl", "wb"))
json.dump({"LOOKBACK": LOOKBACK, "HORIZON": HORIZON, "routes": routes},
          open(MODEL_DIR / "seq_meta.json", "w"), indent=2)
print("[DONE] Saved GRU model → model/traffic_seq.keras")
