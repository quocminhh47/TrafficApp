# scripts/train_lstm_hcmc_congestion_multi.py

import os
import math
import unicodedata
import numpy as np
import pandas as pd

from typing import List, Tuple, Dict

from sklearn.metrics import accuracy_score, f1_score, classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# =========================
# CẤU HÌNH
# =========================

CSV_PATH = "../data/raw/hcmc/train.csv"  # giống GRU
LOOKBACK = 16
TRAIN_RATIO = 0.8
MIN_TIMESTEPS = 80
MIN_POS_RATIO = 0.05
MIN_NEG_RATIO = 0.05

CONGESTED_LOS = {"D", "E", "F"}
MODEL_DIR = "../model/hcmc"  # lưu chung thư mục, nhưng file tên khác (lstm_...)


# =========================
# HÀM TIỆN ÍCH (copy từ GRU)
# =========================

def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower().replace(" ", "_")
    for ch in ["/", "\\", ",", ".", ":", ";", "(", ")", "'", '"']:
        text = text.replace(ch, "")
    return text


def build_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    period_num = df["period"].str.extract(r"period_(\d+)_(\d+)", expand=True).astype(int)
    df["hour"] = period_num[0]
    df["minute"] = period_num[1]
    df["DateTime"] = (
        df["date"]
        + pd.to_timedelta(df["hour"], unit="h")
        + pd.to_timedelta(df["minute"], unit="m")
    )
    return df


def get_eligible_streets(df: pd.DataFrame) -> List[str]:
    street_groups = []
    for street, g in df.groupby("street_name"):

        def is_congested(group: pd.Series) -> int:
            ratio_congested = (group.isin(CONGESTED_LOS)).mean()
            return int(ratio_congested >= 0.5)

        s = (
            g.groupby("DateTime")["LOS"]
            .apply(is_congested)
            .sort_index()
            .astype(int)
        )

        if len(s) < MIN_TIMESTEPS:
            continue

        ratio_pos = (s == 1).mean()
        ratio_neg = (s == 0).mean()
        if ratio_pos < MIN_POS_RATIO or ratio_neg < MIN_NEG_RATIO:
            continue

        street_groups.append((street, len(s), ratio_pos, ratio_neg))

    street_groups.sort(key=lambda x: x[1], reverse=True)

    print("===== CÁC TUYẾN ĐỦ ĐIỀU KIỆN (sau filter) =====")
    for street, n, r_pos, r_neg in street_groups:
        print(
            f"- {street:30s} | timesteps={n:4d} | "
            f"tắc={r_pos*100:5.1f}% | không tắc={r_neg*100:5.1f}%"
        )

    return [s[0] for s in street_groups]


def build_congestion_series(df: pd.DataFrame, street_name: str) -> pd.Series:
    df_st = df[df["street_name"] == street_name].copy()
    if df_st.empty:
        raise ValueError(f"Không có dữ liệu cho street_name='{street_name}'")

    def is_congested(group: pd.Series) -> int:
        ratio_congested = (group.isin(CONGESTED_LOS)).mean()
        return int(ratio_congested >= 0.5)

    s = (
        df_st.groupby("DateTime")["LOS"]
        .apply(is_congested)
        .sort_index()
        .astype(int)
    )

    print(f"[INFO] Tuyến '{street_name}': {len(s)} mốc thời gian (sau group)")
    print(s.value_counts().rename(index={0: "Không tắc (0)", 1: "Tắc (1)"}))
    return s


def build_feature_matrix(series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    s = series.sort_index()
    times = s.index

    y = s.values.astype(np.float32)

    total_minutes = times.hour * 60 + times.minute
    sin_t = np.sin(2 * math.pi * total_minutes / (24 * 60))
    cos_t = np.cos(2 * math.pi * total_minutes / (24 * 60))

    weekday = times.weekday
    sin_w = np.sin(2 * math.pi * weekday / 7.0)
    cos_w = np.cos(2 * math.pi * weekday / 7.0)

    F = np.stack([y, sin_t, cos_t, sin_w, cos_w], axis=1)  # (T, 5)
    return F, y


def build_sequences(F: np.ndarray, y: np.ndarray, lookback: int):
    T = len(y)
    X_list = []
    y_list = []

    for t in range(lookback, T):
        seq = F[t - lookback : t]  # (lookback, n_feat)
        X_list.append(seq)
        y_list.append(y[t])

    X = np.stack(X_list)
    y_target = np.array(y_list)
    return X, y_target


# =========================
# LSTM MODEL
# =========================

def build_lstm_model(input_shape):
    """
    Thay vì GRU, dùng LSTM.
    input_shape ~ (lookback, n_feat)
    """
    model = Sequential()
    model.add(LSTM(32, input_shape=input_shape, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )
    return model


def train_for_street(df: pd.DataFrame, street_name: str) -> Dict:
    print("\n" + "=" * 80)
    print(f"[TRAIN LSTM] Tuyến: {street_name}")
    print("=" * 80)

    s = build_congestion_series(df, street_name)

    F, y = build_feature_matrix(s)
    print(f"[INFO] Feature matrix: F.shape = {F.shape}, y.shape = {y.shape}")

    if len(y) <= LOOKBACK + 10:
        print(
            f"[WARN] Số mốc thời gian ({len(y)}) quá ít so với LOOKBACK={LOOKBACK}. Bỏ qua."
        )
        return {
            "street_name": street_name,
            "timesteps": len(y),
            "status": "skipped_too_short",
        }

    X, y_target = build_sequences(F, y, LOOKBACK)
    print(f"[INFO] Sequence data: X.shape = {X.shape}, y_target.shape = {y_target.shape}")

    N = X.shape[0]
    split_idx = int(N * TRAIN_RATIO)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y_target[:split_idx], y_target[split_idx:]

    print(f"[INFO] Train samples = {len(X_train)}, Test samples = {len(X_test)}")

    if len(X_test) < 10:
        print("[WARN] Test set quá nhỏ (<10). Bỏ qua.")
        return {
            "street_name": street_name,
            "timesteps": len(y),
            "status": "skipped_small_test",
        }

    model = build_lstm_model(input_shape=X_train.shape[1:])

    es = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=200,
        batch_size=16,
        callbacks=[es],
        verbose=1,
    )

    y_prob = model.predict(X_test).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n===== KẾT QUẢ TEST (LSTM) =====")
    print(f"Accuracy : {acc:.3f}")
    print(f"F1-score : {f1:.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=3))

    os.makedirs(MODEL_DIR, exist_ok=True)
    slug = slugify(street_name)
    model_path = os.path.join(MODEL_DIR, f"lstm_congestion_{slug}.keras")
    model.save(model_path)
    print(f"[INFO] Đã lưu model LSTM vào: {model_path}")

    return {
        "street_name": street_name,
        "slug": slug,
        "timesteps": len(y),
        "n_train_seq": len(X_train),
        "n_test_seq": len(X_test),
        "acc": acc,
        "f1": f1,
        "status": "trained",
        "model_path": model_path,
    }


# =========================
# MAIN
# =========================

def main():
    print(f"[INFO] Đọc dữ liệu từ: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    df = build_datetime(df)

    eligible_streets = get_eligible_streets(df)
    if not eligible_streets:
        print("[WARN] Không có tuyến nào đủ điều kiện theo tiêu chí hiện tại.")
        return

    print("\n[INFO] Sẽ train LSTM cho các tuyến sau:")
    for st_name in eligible_streets:
        print("  -", st_name)

    all_results = []
    for street_name in eligible_streets:
        res = train_for_street(df, street_name)
        all_results.append(res)

    os.makedirs(MODEL_DIR, exist_ok=True)
    df_metrics = pd.DataFrame(all_results)
    metrics_path = os.path.join(MODEL_DIR, "lstm_congestion_metrics.csv")
    df_metrics.to_csv(metrics_path, index=False)
    print(f"\n[INFO] Đã lưu tổng hợp metrics LSTM tại: {metrics_path}")
    print(df_metrics)


if __name__ == "__main__":
    main()
