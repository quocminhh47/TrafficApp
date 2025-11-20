# scripts/prep_seq_light.py
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from modules.data_loader import load_all

MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)

LOOKBACK = 168  # 7 ngày
HORIZON = 24    # 24h forecast
CITY = "Minneapolis"
ZONE = "I94"


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
    print("[INFO] Loading all processed data for training...")
    df = load_all(CITY, ZONE)
    if df.empty:
        raise RuntimeError("No data found in processed_ds. Run ingest_metro_i94.py first.")

    routes = sorted(df["RouteId"].astype(str).unique().tolist())
    print(f"[INFO] {len(routes)} routes: {routes}")

    # Fit scaler trên toàn bộ Vehicles
    scaler = StandardScaler()
    df["Vehicles"] = df["Vehicles"].astype(float)
    scaler.fit(df[["Vehicles"]])

    # Chuẩn hóa theo route & resample 1h
    X_list, y_list = [], []

    for rid in routes:
        g = df[df["RouteId"].astype(str) == str(rid)].copy()
        g = (
            g.set_index("DateTime")
            .resample("1h")["Vehicles"]
            .mean()
            .interpolate(limit=3)
            .dropna()
            .reset_index()
        )
        if len(g) < LOOKBACK + HORIZON:
            print(f"[WARN] Route {rid} too short: {len(g)} rows, skipping.")
            continue

        v_scaled = scaler.transform(g[["Vehicles"]]).astype(np.float32)
        t_feat = time_feats(g["DateTime"])
        rid_one = np.zeros((len(routes),), dtype=np.float32)
        rid_idx = routes.index(rid)
        rid_one[rid_idx] = 1.0
        rid_feat = np.tile(rid_one, (len(g), 1))

        feat_all = np.concatenate([v_scaled, t_feat, rid_feat], axis=1)

        # Tạo sequences
        values = g["Vehicles"].values.astype(np.float32)
        for i in range(len(g) - LOOKBACK - HORIZON + 1):
            x_win = feat_all[i : i + LOOKBACK, :]
            y_win = values[i + LOOKBACK : i + LOOKBACK + HORIZON]
            X_list.append(x_win)
            y_list.append(y_win)

    if not X_list:
        raise RuntimeError("No sequences generated. Check data length and LOOKBACK/HORIZON.")

    X = np.stack(X_list, axis=0)
    y = np.stack(y_list, axis=0)

    print(f"[INFO] X shape: {X.shape}, y shape: {y.shape}")

    # Save npz
    np.savez_compressed(MODEL_DIR / "seq_train.npz", X=X, y=y)

    # Save meta
    meta = {"LOOKBACK": LOOKBACK, "HORIZON": HORIZON, "routes": routes}
    with open(MODEL_DIR / "seq_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Save scaler
    import pickle

    with open(MODEL_DIR / "vehicles_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print("[INFO] Saved seq_train.npz, seq_meta.json, vehicles_scaler.pkl")


if __name__ == "__main__":
    main()
