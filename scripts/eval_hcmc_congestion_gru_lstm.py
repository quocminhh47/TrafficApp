# scripts/eval_hcmc_congestion_gru_lstm.py

import os
import math
import unicodedata
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model  # type: ignore

# ==== CẤU HÌNH ====

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

CSV_PATH = os.path.join(BASE_DIR, "data", "raw", "hcmc", "train.csv")

MODEL_DIR = os.path.join(BASE_DIR, "model", "hcmc")

OUTPUT_DIR = os.path.join(BASE_DIR, "data", "hcmc_eval")
DETAIL_DIR = os.path.join(OUTPUT_DIR, "details")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DETAIL_DIR, exist_ok=True)

LOOKBACK = 16          # cùng logic train
TRAIN_RATIO = 0.8      # chia theo thời gian
CONGESTED_LOS = {"D", "E", "F"}


# ==== HÀM TIỆN ÍCH GIỐNG TRAIN GRU ====


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
    # period dạng "period_7_0", "period_16_30"
    period_num = df["period"].astype(str).str.extract(r"period_(\d+)_(\d+)", expand=True).astype(int)
    df["hour"] = period_num[0]
    df["minute"] = period_num[1]
    df["DateTime"] = (
        df["date"]
        + pd.to_timedelta(df["hour"], unit="h")
        + pd.to_timedelta(df["minute"], unit="m")
    )
    return df


def build_congestion_series(df: pd.DataFrame, street_name: str) -> pd.Series:
    """
    Giống train: group theo DateTime, tính is_congested (0/1) từ LOS.
    """
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
    return s


def build_feature_matrix(series: pd.Series) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
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
    return F, y, times


def build_sequences(
    F: np.ndarray,
    y: np.ndarray,
    t_index: pd.DatetimeIndex,
    lookback: int,
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    T = len(y)
    X_list, y_list, idx_list = [], [], []

    for t in range(lookback, T):
        seq = F[t - lookback : t]
        X_list.append(seq)
        y_list.append(y[t])
        idx_list.append(t_index[t])

    if not X_list:
        return (
            np.zeros((0, lookback, F.shape[1]), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            pd.DatetimeIndex([]),
        )

    X = np.stack(X_list)
    y_target = np.array(y_list, dtype=np.float32)
    idx = pd.DatetimeIndex(idx_list)
    return X, y_target, idx


def time_series_train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    idx: pd.DatetimeIndex,
    ratio: float = TRAIN_RATIO,
):
    N = X.shape[0]
    if N == 0:
        return X, X, y, y, idx, idx

    split_idx = max(1, int(N * ratio))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    idx_train, idx_test = idx[:split_idx], idx[split_idx:]
    return X_train, X_test, y_train, y_test, idx_train, idx_test


def compute_common_metrics_offline(y_true, y_pred_proba) -> Dict[str, float]:
    """
    Đánh giá cho bài toán binary (0/1, dự báo xác suất):
      - MSE, RMSE, MAE, SMAPE (%)
      - Accuracy (%) sau khi threshold 0.5.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred_proba, dtype=float)

    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not np.any(mask):
        return {k: np.nan for k in ["MSE", "RMSE", "MAE", "SMAPE", "Accuracy"]}

    y_true = y_true[mask]
    y_pred = y_pred[mask]

    diff = y_pred - y_true
    mse = float(np.mean(diff**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))

    denom = np.abs(y_true) + np.abs(y_pred)
    smape = float(
        np.mean(2.0 * np.abs(diff) / (denom + 1e-8)) * 100.0
    )

    y_bin = (y_pred >= 0.5).astype(float)
    acc = float(np.mean(y_bin == y_true) * 100.0)

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "SMAPE": smape,
        "Accuracy": acc,
    }


def find_street_slug_mapping(df: pd.DataFrame) -> List[Tuple[str, str]]:
    """
    Trả về list (street_name, slug) cho mọi tuyến xuất hiện trong train.csv.
    slug = slugify(street_name)
    """
    streets = df["street_name"].dropna().unique().tolist()
    mapping = [(st, slugify(st)) for st in streets]
    return mapping


def eval_one_model_for_route(
    street_name: str,
    slug: str,
    model_type: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    idx_test: pd.DatetimeIndex,
) -> Dict:
    """
    Evaluate 1 model (GRU hoặc LSTM) cho 1 tuyến.
    """
    if model_type not in {"GRU", "LSTM"}:
        raise ValueError(f"model_type không hỗ trợ: {model_type}")

    if model_type == "GRU":
        model_path = os.path.join(MODEL_DIR, f"gru_congestion_{slug}.keras")
    else:
        model_path = os.path.join(MODEL_DIR, f"lstm_congestion_{slug}.keras")

    if not os.path.exists(model_path):
        print(f"  -> BỎ QUA {model_type}: không tìm thấy model: {model_path}")
        return {
            "street_name": street_name,
            "slug": slug,
            "model_type": model_type,
            "timesteps": len(y_test),
            "n_seq_total": 0,
            "n_seq_test": 0,
            "MSE": np.nan,
            "RMSE": np.nan,
            "MAE": np.nan,
            "SMAPE": np.nan,
            "Accuracy": np.nan,
        }

    print(f"  -> Đánh giá {model_type} với model_path={model_path}")
    model = load_model(model_path)

    if len(X_test) == 0:
        print("    (Test set rỗng)")
        return {
            "street_name": street_name,
            "slug": slug,
            "model_type": model_type,
            "timesteps": len(y_test),
            "n_seq_total": 0,
            "n_seq_test": 0,
            "MSE": np.nan,
            "RMSE": np.nan,
            "MAE": np.nan,
            "SMAPE": np.nan,
            "Accuracy": np.nan,
        }

    y_proba = model.predict(X_test, verbose=0).ravel()
    metrics = compute_common_metrics_offline(y_true=y_test, y_pred_proba=y_proba)

    # Lưu chi tiết actual vs predict để vẽ biểu đồ sau này
    df_detail = pd.DataFrame(
        {
            "DateTime": idx_test,
            "Actual_is_congested": y_test,
            "Predicted_prob": y_proba,
        }
    )
    detail_path = os.path.join(DETAIL_DIR, f"eval_detail_{slug}_{model_type}.csv")
    df_detail.to_csv(detail_path, index=False)
    print(f"    -> Đã lưu chi tiết: {detail_path}")

    return {
        "street_name": street_name,
        "slug": slug,
        "model_type": model_type,
        "timesteps": len(y_test),
        "n_seq_total": len(X_test) + int(len(X_test) / (1 - TRAIN_RATIO) * TRAIN_RATIO),  # gần đúng
        "n_seq_test": len(X_test),
        "MSE": metrics["MSE"],
        "RMSE": metrics["RMSE"],
        "MAE": metrics["MAE"],
        "SMAPE": metrics["SMAPE"],
        "Accuracy": metrics["Accuracy"],
    }


def main():
    print(f"[INFO] Đọc dữ liệu HCMC từ: {CSV_PATH}")
    df_raw = pd.read_csv(CSV_PATH)
    df = build_datetime(df_raw)

    mapping = find_street_slug_mapping(df)
    print(f"[INFO] Tìm thấy {len(mapping)} tuyến trong train.csv")

    results = []

    for street_name, slug in mapping:
        print("\n" + "=" * 80)
        print(f"[ROUTE] {street_name} (slug={slug})")
        print("=" * 80)

        try:
            s = build_congestion_series(df, street_name)
        except Exception as ex:
            print(f"  -> BỎ QUA: lỗi build_congestion_series: {ex}")
            continue

        if len(s) <= LOOKBACK + 10:
            print(
                f"  -> BỎ QUA: quá ít điểm ({len(s)}) so với LOOKBACK={LOOKBACK}"
            )
            continue

        F, y, t_index = build_feature_matrix(s)
        X, y_target, idx = build_sequences(F, y, t_index, LOOKBACK)

        X_train, X_test, y_train, y_test, idx_train, idx_test = time_series_train_test_split(
            X, y_target, idx, ratio=TRAIN_RATIO
        )
        print(
            f"  Seq tổng={len(X)}, Train={len(X_train)}, Test={len(X_test)}"
        )

        # Đánh giá GRU (nếu có)
        res_gru = eval_one_model_for_route(
            street_name, slug, "GRU", X_test, y_test, idx_test
        )
        results.append(res_gru)

        # Đánh giá LSTM (nếu có)
        res_lstm = eval_one_model_for_route(
            street_name, slug, "LSTM", X_test, y_test, idx_test
        )
        results.append(res_lstm)

    df_res = pd.DataFrame(results)
    out_summary = os.path.join(OUTPUT_DIR, "hcmc_eval_summary.csv")
    df_res.to_csv(out_summary, index=False)
    print(f"\n[INFO] Đã lưu summary HCMC (GRU + LSTM): {out_summary}")
    print(df_res)


if __name__ == "__main__":
    main()
