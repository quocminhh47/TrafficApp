# scripts/eval_hcmc_congestion_all_routes.py
#
# Đánh giá các mô hình congestion HCMC cho TẤT CẢ tuyến,
# xuất ra:
#   - data/hcmc_eval/hcmc_eval_summary.csv  (MSE, RMSE, MAE, SMAPE, Accuracy)
#   - data/hcmc_eval/eval_detail_<slug>_<model_type>.csv (actual vs predict)
#
# Hiện tại: dùng cho GRU congestion (binary_prob) với raw train.csv:
#   _id,segment_id,date,weekday,period,LOS,...,street_name,...
#
# Sau này: có thể extend sang LSTM/GRNN/ARIMA/SARIMA bằng cách thêm
# nhiều model_type + model_path vào train_routes_summary.csv.

import os
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from tensorflow import keras  # type: ignore

# ========= CẤU HÌNH PATH =========

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# Raw history HCMC
TRAIN_CSV = os.path.join(BASE_DIR, "data", "raw", "hcmc", "train.csv")

# File summary tuyến + model (từ bảng street_name,slug,timesteps,...,model_path)
ROUTE_SUMMARY_CSV = os.path.join(
    BASE_DIR, "data", "raw", "hcmc", "train_routes_summary.csv"
)

# Nơi lưu kết quả đánh giá
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "hcmc_eval")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========= CẤU HÌNH =========

THRESHOLD = 0.5            # ngưỡng phân loại kẹt / không kẹt
TEST_RATIO = 0.2           # 20% cuối làm test
DEFAULT_MODEL_TYPE = "GRU"  # nếu file summary không có cột model_type
PERIOD_MINUTES = 30        # giả định mỗi period = 30 phút


@dataclass
class RouteConfig:
    street_name: str
    slug: str
    timesteps: int
    model_path: str
    model_type: str = DEFAULT_MODEL_TYPE


def load_route_configs() -> List[RouteConfig]:
    """
    Đọc file summary và trả về list RouteConfig cho các tuyến status='trained'.

    YÊU CẦU file train_routes_summary.csv có ít nhất các cột:
      - street_name
      - slug
      - timesteps
      - model_path
      - (tuỳ chọn) model_type
      - (tuỳ chọn) status
    """
    if not os.path.exists(ROUTE_SUMMARY_CSV):
        raise FileNotFoundError(f"Không tìm thấy ROUTE_SUMMARY_CSV: {ROUTE_SUMMARY_CSV}")

    df = pd.read_csv(ROUTE_SUMMARY_CSV)

    if "model_path" not in df.columns:
        raise KeyError(
            f"File {ROUTE_SUMMARY_CSV} không có cột 'model_path'. "
            "Hãy đảm bảo file summary có cột: street_name, slug, timesteps, model_path"
        )

    # Lọc tuyến đã train xong nếu có cột status
    if "status" in df.columns:
        df = df[df["status"] == "trained"].copy()

    configs: List[RouteConfig] = []

    for _, row in df.iterrows():
        model_path = str(row["model_path"])
        if not os.path.isabs(model_path):
            model_path = os.path.join(BASE_DIR, model_path)

        model_type = (
            str(row["model_type"])
            if "model_type" in df.columns
            else DEFAULT_MODEL_TYPE
        )

        cfg = RouteConfig(
            street_name=str(row["street_name"]),
            slug=str(row["slug"]),
            timesteps=int(row["timesteps"]),
            model_path=model_path,
            model_type=model_type,
        )
        configs.append(cfg)

    return configs


def load_series_for_route_by_street(street_name: str) -> pd.Series:
    """
    Đọc train.csv, lọc theo street_name, trả về Series:
      index = DateTime, value = is_congested (0/1).

    train.csv có các cột:
      - date        (ngày)
      - period      (chỉ số khung trong ngày, int)
      - LOS         (mức độ phục vụ)
      - street_name (tên đường)
    """
    if not os.path.exists(TRAIN_CSV):
        raise FileNotFoundError(f"Không tìm thấy TRAIN_CSV: {TRAIN_CSV}")

    df = pd.read_csv(TRAIN_CSV)

    required_cols = ["date", "period", "LOS", "street_name"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"train.csv thiếu các cột: {missing}")

    # Lọc theo street_name
    df_r = df[df["street_name"] == street_name].copy()
    if df_r.empty:
        raise ValueError(f"Không tìm thấy dữ liệu cho street_name='{street_name}' trong train.csv")

    # Parse date
    df_r["date"] = pd.to_datetime(df_r["date"], errors="coerce")
    df_r = df_r.dropna(subset=["date"])

    # Đảm bảo period là số
    df_r["period"] = pd.to_numeric(df_r["period"], errors="coerce")
    df_r = df_r.dropna(subset=["period"])
    df_r["period"] = df_r["period"].astype(int)

    # Tạo DateTime: giả định mỗi period = PERIOD_MINUTES phút từ đầu ngày
    # nếu period bắt đầu từ 1, có thể dùng (period - 1); nếu từ 0 thì dùng period
    df_r["DateTime"] = df_r["date"] + pd.to_timedelta(
        (df_r["period"] - 1) * PERIOD_MINUTES, unit="m"
    )

    # Tạo nhãn is_congested từ LOS
    # - nếu LOS là số: congested nếu >= 4
    # - nếu LOS là chữ (A..F): congested nếu D/E/F
    if np.issubdtype(df_r["LOS"].dtype, np.number):
        df_r["is_congested"] = (df_r["LOS"] >= 4).astype("float32")
    else:
        los_str = df_r["LOS"].astype(str).str.upper()
        df_r["is_congested"] = los_str.isin(["D", "E", "F"]).astype("float32")

    df_r = df_r.sort_values("DateTime")

    s = pd.Series(df_r["is_congested"].values, index=df_r["DateTime"])
    s = s.astype("float32")
    return s


def build_sequences(
    series: pd.Series,
    timesteps: int,
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Tạo sequences cho các mô hình chuỗi thời gian:

      X shape: (num_seq, timesteps, 1)
      y shape: (num_seq,)
      idx: thời điểm được dự đoán (DateTime của y)
    """
    values = series.values.astype("float32")
    times = series.index.to_numpy()

    xs, ys, idx = [], [], []
    for i in range(len(values) - timesteps):
        xs.append(values[i : i + timesteps])
        ys.append(values[i + timesteps])
        idx.append(times[i + timesteps])

    if not xs:
        return (
            np.zeros((0, timesteps, 1), dtype="float32"),
            np.zeros((0,), dtype="float32"),
            pd.DatetimeIndex([]),
        )

    X = np.array(xs)[..., np.newaxis]  # (N, T, 1)
    y = np.array(ys).astype("float32")
    idx = pd.DatetimeIndex(idx)
    return X, y, idx


def time_series_train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    idx: pd.DatetimeIndex,
    test_ratio: float = TEST_RATIO,
):
    """
    Chia train/test theo thời gian, không shuffle:
      - train: 0 .. n_train-1
      - test : n_train .. n-1
    """
    n = len(X)
    if n == 0:
        return X, X, y, y, idx, idx

    n_test = max(1, int(n * test_ratio))
    n_train = n - n_test
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    idx_train, idx_test = idx[:n_train], idx[n_train:]
    return X_train, X_test, y_train, y_test, idx_train, idx_test


def compute_common_metrics_offline(
    y_true,
    y_pred,
    *,
    task: str = "binary_prob",
    acc_tolerance: float = 0.2,
    threshold: float = 0.5,
) -> dict:
    """
    Bộ chỉ số chung: MSE / RMSE / MAE / SMAPE / Accuracy.

    - Với HCMC hiện tại:
        task="binary_prob", y_true={0,1}, y_pred=xác suất, threshold -> Accuracy.
    - Sau này có thể dùng cho regression:
        task="regression", y_true,y_pred là giá trị liên tục.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not np.any(mask):
        return {
            "MSE": np.nan,
            "RMSE": np.nan,
            "MAE": np.nan,
            "SMAPE": np.nan,
            "Accuracy": np.nan,
        }

    y_true = y_true[mask]
    y_pred = y_pred[mask]

    diff = y_pred - y_true

    # --- Sai số cơ bản ---
    mse = float(np.mean(diff**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))

    # --- SMAPE ---
    denom = np.abs(y_true) + np.abs(y_pred)
    smape = float(
        np.mean(
            2.0 * np.abs(diff) / (denom + 1e-8)
        )
        * 100.0
    )

    # --- Accuracy ---
    if task == "binary_prob":
        y_bin = (y_pred >= threshold).astype(float)
        acc = float(np.mean(y_bin == y_true) * 100.0)
    else:
        rel_err = np.abs(diff) / (np.abs(y_true) + 1e-8)
        acc = float(np.mean(rel_err <= acc_tolerance) * 100.0)

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "SMAPE": smape,
        "Accuracy": acc,
    }


def eval_one_route(cfg: RouteConfig) -> dict:
    """
    Đánh giá model congestion cho 1 tuyến.
    Trả về dict chứa các metric để gộp thành DataFrame summary.
    Đồng thời lưu chi tiết actual vs predict ra CSV.
    """
    print(f"\n=== Đánh giá tuyến: {cfg.street_name} ({cfg.slug}) [{cfg.model_type}] ===")
    print(f"Model path: {cfg.model_path}")
    print(f"Timesteps: {cfg.timesteps}")

    if not os.path.exists(cfg.model_path):
        print(f"  -> BỎ QUA: Không tìm thấy model: {cfg.model_path}")
        return {
            "street_name": cfg.street_name,
            "slug": cfg.slug,
            "model_type": cfg.model_type,
            "timesteps": cfg.timesteps,
            "n_seq_total": 0,
            "n_seq_test": 0,
            "MSE": np.nan,
            "RMSE": np.nan,
            "MAE": np.nan,
            "SMAPE": np.nan,
            "Accuracy": np.nan,
        }

    # 1) Lịch sử theo street_name
    s = load_series_for_route_by_street(cfg.street_name)
    print(f"  Series length (raw points): {len(s)}")

    # 2) Sequence
    X, y, idx = build_sequences(s, cfg.timesteps)
    print(f"  Sequences total: {len(X)}")

    if len(X) < 10:
        print("  -> BỎ QUA: Quá ít sequence để đánh giá.")
        return {
            "street_name": cfg.street_name,
            "slug": cfg.slug,
            "model_type": cfg.model_type,
            "timesteps": cfg.timesteps,
            "n_seq_total": len(X),
            "n_seq_test": 0,
            "MSE": np.nan,
            "RMSE": np.nan,
            "MAE": np.nan,
            "SMAPE": np.nan,
            "Accuracy": np.nan,
        }

    # 3) Train/test split theo thời gian
    X_train, X_test, y_train, y_test, idx_train, idx_test = time_series_train_test_split(
        X, y, idx, test_ratio=TEST_RATIO
    )
    print(f"  Train seq: {len(X_train)}, Test seq: {len(X_test)}")

    if len(X_test) == 0:
        print("  -> Không có mẫu test (do DATA quá ngắn).")
        return {
            "street_name": cfg.street_name,
            "slug": cfg.slug,
            "model_type": cfg.model_type,
            "timesteps": cfg.timesteps,
            "n_seq_total": len(X),
            "n_seq_test": 0,
            "MSE": np.nan,
            "RMSE": np.nan,
            "MAE": np.nan,
            "SMAPE": np.nan,
            "Accuracy": np.nan,
        }

    # 4) Load model & predict
    model = keras.models.load_model(cfg.model_path)
    y_proba = model.predict(X_test, verbose=0).reshape(-1)

    # 5) Tính metric chung
    metrics = compute_common_metrics_offline(
        y_true=y_test,
        y_pred=y_proba,
        task="binary_prob",
        threshold=THRESHOLD,
    )

    print("  ---- Metrics (test) ----")
    print(f"  MSE      : {metrics['MSE']:.4f}")
    print(f"  RMSE     : {metrics['RMSE']:.4f}")
    print(f"  MAE      : {metrics['MAE']:.4f}")
    print(f"  SMAPE    : {metrics['SMAPE']:.2f} %")
    print(f"  Accuracy : {metrics['Accuracy']:.1f} %")

    # 6) Lưu chi tiết actual vs predicted
    y_bin = (y_proba >= THRESHOLD).astype("int32")
    df_detail = pd.DataFrame(
        {
            "DateTime": idx_test,
            "Actual_is_congested": y_test,
            "Predicted_prob": y_proba,
            "Predicted_label": y_bin,
        }
    )
    out_detail = os.path.join(
        OUTPUT_DIR,
        f"eval_detail_{cfg.slug}_{cfg.model_type}.csv",
    )
    df_detail.to_csv(out_detail, index=False)
    print(f"  -> Đã lưu chi tiết actual vs predict: {out_detail}")

    # 7) Trả về summary 1 dòng
    return {
        "street_name": cfg.street_name,
        "slug": cfg.slug,
        "model_type": cfg.model_type,
        "timesteps": cfg.timesteps,
        "n_seq_total": len(X),
        "n_seq_test": len(X_test),
        "MSE": metrics["MSE"],
        "RMSE": metrics["RMSE"],
        "MAE": metrics["MAE"],
        "SMAPE": metrics["SMAPE"],
        "Accuracy": metrics["Accuracy"],
    }


def main():
    configs = load_route_configs()
    print(f"Đang đánh giá {len(configs)} tuyến HCMC (status='trained')")

    results = []
    for cfg in configs:
        res = eval_one_route(cfg)
        results.append(res)

    df_res = pd.DataFrame(results)
    out_summary = os.path.join(OUTPUT_DIR, "hcmc_eval_summary.csv")
    df_res.to_csv(out_summary, index=False)
    print(f"\nĐã lưu summary metrics cho tất cả tuyến vào: {out_summary}")


if __name__ == "__main__":
    main()
