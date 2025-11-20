# scripts/train_gru_i94_from_parquet.py

from pathlib import Path
import json

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input, GRU, Dense
from tensorflow.keras.models import Model

from modules.data_loader import load_slice, list_routes
from modules.model_utils import time_feats


# ======================= CONFIG =======================

CITY = "Minneapolis"
ZONE = "I94"           # thư mục data/processed_ds/Minneapolis/I94/...
LOOKBACK = 168         # 7 ngày x 24h
HORIZON = 24           # dự báo 24h

# ======================================================


def build_scaler(df_all: pd.DataFrame) -> StandardScaler:
    v = df_all["Vehicles"].astype(np.float32).values.reshape(-1, 1)
    scaler = StandardScaler()
    scaler.fit(v)
    return scaler


def build_sequences_for_route(
    g: pd.DataFrame,
    rid: str,
    scaler: StandardScaler,
    lookback: int,
    horizon: int,
    rid2idx: dict,
    n_routes: int,
):
    v = g["Vehicles"].astype(np.float32).values.reshape(-1, 1)
    v_scaled = scaler.transform(v).flatten()  # (N,)

    dt_index = g["DateTime"].to_numpy()
    tf = time_feats(pd.to_datetime(dt_index))  # (N, 4)

    ohe = np.zeros((len(g), n_routes), dtype=np.float32)
    ohe[:, rid2idx[rid]] = 1.0

    feats = np.concatenate(
        [
            v_scaled.reshape(-1, 1),
            tf,
            ohe,
        ],
        axis=1,
    )  # (N, 1+4+n_routes)

    X_list, y_list = [], []
    N = len(feats)
    min_len = lookback + horizon
    if N < min_len:
        print(f"[INFO] Route {rid}: quá ít điểm ({N}), cần >= {min_len}, bỏ qua.")
        return [], []

    for i in range(N - lookback - horizon + 1):
        x_window = feats[i : i + lookback]
        y_window = v_scaled[i + lookback : i + lookback + horizon]

        X_list.append(x_window)
        y_list.append(y_window)

    print(
        f"[OK] Route {rid}: {N} điểm → {len(X_list)} samples "
        f"(lookback={lookback}, horizon={horizon})"
    )
    return X_list, y_list


def build_gru_model(lookback: int, n_feats: int, horizon: int) -> Model:
    inp = Input(shape=(lookback, n_feats))
    x = GRU(64, return_sequences=True)(inp)
    x = GRU(64)(x)
    x = Dense(64, activation="relu")(x)
    out = Dense(horizon)(x)
    model = Model(inp, out, name="traffic_gru_i94_seq2seq")
    model.compile(optimizer="adam", loss="mse")
    return model


def main():
    # 1) Lấy routes từ parquet I94
    routes = list_routes(CITY, ZONE)
    if not routes:
        raise SystemExit(
            f"Không tìm thấy route nào cho {CITY}/{ZONE} trong data/processed_ds."
        )

    print(f"[INFO] Routes I94: {routes}")
    rid2idx = {rid: idx for idx, rid in enumerate(routes)}
    n_routes = len(routes)

    # 2) Load toàn bộ dữ liệu I94
    df_all = load_slice(
        city=CITY,
        zone=ZONE,
        routes=routes,
        start_dt=None,
        end_dt=None,
    )
    if df_all.empty:
        raise SystemExit(
            f"[ERROR] Không load được dữ liệu I94 từ parquet. Kiểm tra lại paths."
        )

    df_all["DateTime"] = pd.to_datetime(df_all["DateTime"], errors="coerce")
    df_all = df_all.dropna(subset=["DateTime", "Vehicles", "RouteId"])
    df_all["RouteId"] = df_all["RouteId"].astype(str)

    # 3) Resample 1H cho từng route
    df_all_resampled = []
    for rid in routes:
        g = df_all[df_all["RouteId"] == rid].copy()
        if g.empty:
            print(f"[WARN] Route {rid} không có dữ liệu, bỏ qua.")
            continue

        g = (
            g.set_index("DateTime")
             .resample("1H")["Vehicles"]
             .mean()
             .dropna()
             .reset_index()
             .sort_values("DateTime")
        )
        g["RouteId"] = rid
        df_all_resampled.append(g)

    if not df_all_resampled:
        raise SystemExit("[ERROR] Không resample được route nào cho I94.")

    df_all_hourly = pd.concat(df_all_resampled, ignore_index=True)
    print(
        f"[INFO] Tổng số dòng hourly I94 (gộp tất cả routes): "
        f"{len(df_all_hourly)}"
    )

    # 4) Fit scaler riêng cho family I94
    scaler = build_scaler(df_all_hourly)
    print("[INFO] Đã fit scaler I94.")

    # 5) Build toàn bộ sample X, y (multi-route)
    all_X, all_y = [], []
    for rid in routes:
        g = df_all_hourly[df_all_hourly["RouteId"] == rid].copy()
        if g.empty:
            continue
        X_list, y_list = build_sequences_for_route(
            g=g,
            rid=rid,
            scaler=scaler,
            lookback=LOOKBACK,
            horizon=HORIZON,
            rid2idx=rid2idx,
            n_routes=n_routes,
        )
        all_X.extend(X_list)
        all_y.extend(y_list)

    if not all_X:
        raise SystemExit("[ERROR] Không tạo được sample nào cho GRU I94.")

    X = np.stack(all_X, axis=0)
    y = np.stack(all_y, axis=0)
    n_feats = X.shape[-1]

    print(f"[INFO] X.shape={X.shape}, y.shape={y.shape}, n_feats={n_feats}")

    # 6) Build & train GRU
    model = build_gru_model(LOOKBACK, n_feats, HORIZON)
    model.summary(print_fn=lambda s: print("[MODEL]", s))

    model.fit(
        X,
        y,
        epochs=30,
        batch_size=32,
        validation_split=0.1,
        verbose=1,
    )
    print("[INFO] Train GRU I94 xong.")

    # 7) Save vào thư mục model family I94
    family_dir = Path("model") / "I94"
    family_dir.mkdir(parents=True, exist_ok=True)

    model_path = family_dir / "traffic_seq.keras"
    scaler_path = family_dir / "vehicles_scaler.pkl"
    meta_path = family_dir / "seq_meta.json"

    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    meta = {
        "LOOKBACK": LOOKBACK,
        "HORIZON": HORIZON,
        "routes": routes,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[INFO] Đã lưu GRU I94 vào {model_path}")
    print(f"[INFO] Đã lưu scaler I94 vào {scaler_path}")
    print(f"[INFO] Đã lưu meta I94 vào {meta_path}")
    print("[DONE] I94 family đã sẵn sàng.")


if __name__ == "__main__":
    main()
