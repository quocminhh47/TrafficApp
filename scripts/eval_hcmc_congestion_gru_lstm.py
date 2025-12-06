# scripts/eval_hcmc_congestion_selected_routes.py

import os
import json
import math
import unicodedata
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model  # type: ignore

# =====================
# CẤU HÌNH PATH
# =====================

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

TRAIN_CSV = os.path.join(BASE_DIR, "data", "raw", "hcmc", "train.csv")
ROUTES_GEO_JSON = os.path.join(BASE_DIR, "data", "routes_geo.json")

MODEL_DIR = os.path.join(BASE_DIR, "model", "hcmc")

OUTPUT_DIR = os.path.join(BASE_DIR, "data", "hcmc_eval")
DETAIL_DIR = os.path.join(OUTPUT_DIR, "details")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DETAIL_DIR, exist_ok=True)

# =====================
# THAM SỐ
# =====================

LOOKBACK = 16
TRAIN_RATIO = 0.8
CONGESTED_LOS = {"D", "E", "F"}

import unicodedata
import re

def slugify(text: str) -> str:
    """
    Chuẩn hóa tên đường thành slug an toàn:
      - Đ/đ/Ð/ð -> D/d
      - bỏ dấu tiếng Việt
      - chỉ giữ [a-z0-9_], các ký tự khác -> '_'
      - gộp nhiều '_' liên tiếp thành 1, bỏ '_' đầu/cuối
    Ví dụ:
      'Nguyễn Ðình Chiểu' -> 'nguyen_dinh_chieu'
      'Lê Đức Thọ'        -> 'le_duc_tho'
      'Phan Đăng Lưu'     -> 'phan_dang_luu'
    """
    if text is None:
        return ""

    # 1) Chuẩn hóa riêng ký tự Đ/đ/Ð/ð trước khi đổi sang ASCII
    text = (
        text.replace("Đ", "D")
            .replace("đ", "d")
            .replace("Ð", "D")
            .replace("ð", "d")
    )

    # 2) Loại bỏ dấu tiếng Việt
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))

    # 3) Lower case
    text = text.lower()

    # 4) Mọi thứ không phải chữ/số -> '_'
    text = re.sub(r"[^a-z0-9]+", "_", text)

    # 5) Gộp nhiều '_' liên tiếp, bỏ '_' đầu/cuối
    text = re.sub(r"_+", "_", text).strip("_")

    return text



def load_hcmc_route_ids_from_geo() -> Dict[str, str]:
    """
    Đọc routes_geo.json và lấy các route HCMC:
      key   = route_id (slug, ví dụ 'ly_thuong_kiet')
      value = name hiển thị (ví dụ 'Lý Thường Kiệt (HCMC)')
    """
    if not os.path.exists(ROUTES_GEO_JSON):
        raise FileNotFoundError(f"Không thấy routes_geo.json tại {ROUTES_GEO_JSON}")

    with open(ROUTES_GEO_JSON, "r", encoding="utf-8") as f:
        geo = json.load(f)

    mapping: Dict[str, str] = {}
    for rec in geo:
        if rec.get("city") == "HoChiMinh":
            rid = rec.get("route_id")
            name = rec.get("name", rid)
            if rid:
                mapping[str(rid)] = str(name)

    print("[INFO] Các route HCMC sẽ đánh giá (theo routes_geo.json):")
    for rid, name in mapping.items():
        print(f"  - {rid}  =>  {name}")
    return mapping


def build_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    # period dạng "period_7_0", "period_16_30", ...
    period_num = df["period"].astype(str).str.extract(r"period_(\d+)_(\d+)", expand=True).astype(int)
    df["hour"] = period_num[0]
    df["minute"] = period_num[1]
    df["DateTime"] = (
        df["date"]
        + pd.to_timedelta(df["hour"], unit="h")
        + pd.to_timedelta(df["minute"], unit="m")
    )
    return df


def build_congestion_series_for_slug(df: pd.DataFrame, slug: str) -> pd.Series:
    """
    Gom tất cả street_name sao cho slugify(street_name) == slug,
    rồi group theo DateTime để tạo chuỗi is_congested (0/1).
    """
    df = df.copy()
    # tính thêm cột slug_from_name
    df["slug_from_name"] = df["street_name"].astype(str).apply(slugify)
    df_sub = df[df["slug_from_name"] == slug].copy()

    if df_sub.empty:
        raise ValueError(f"[WARN] Không tìm thấy dữ liệu nào có slug='{slug}' trong train.csv")

    def is_congested(group: pd.Series) -> int:
        ratio_congested = (group.isin(CONGESTED_LOS)).mean()
        return int(ratio_congested >= 0.5)

    s = (
        df_sub.groupby("DateTime")["LOS"]
        .apply(is_congested)
        .sort_index()
        .astype(int)
    )

    print(f"[INFO] slug='{slug}': {len(s)} mốc thời gian (sau khi gộp tất cả street_name cùng slug)")
    print(s.value_counts().rename(index={0: "Không tắc (0)", 1: "Tắc (1)"}))
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
    smape = float(np.mean(2.0 * np.abs(diff) / (denom + 1e-8)) * 100.0)

    y_bin = (y_pred >= 0.5).astype(float)
    acc = float(np.mean(y_bin == y_true) * 100.0)

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "SMAPE": smape,
        "Accuracy": acc,
    }


def eval_one_model_for_route(
    display_name: str,
    slug: str,
    model_type: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    idx_test: pd.DatetimeIndex,
) -> Dict:
    if model_type == "GRU":
        model_path = os.path.join(MODEL_DIR, f"gru_congestion_{slug}.keras")
    elif model_type == "LSTM":
        model_path = os.path.join(MODEL_DIR, f"lstm_congestion_{slug}.keras")
    else:
        raise ValueError(f"model_type không hỗ trợ: {model_type}")

    if not os.path.exists(model_path):
        print(f"  -> BỎ QUA {model_type}: không tìm thấy model: {model_path}")
        return {
            "street_name": display_name,
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
            "street_name": display_name,
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
        "street_name": display_name,
        "slug": slug,
        "model_type": model_type,
        "timesteps": len(y_test),
        "n_seq_total": len(X_test) + int(len(X_test) / (1 - TRAIN_RATIO) * TRAIN_RATIO),
        "n_seq_test": len(X_test),
        "MSE": metrics["MSE"],
        "RMSE": metrics["RMSE"],
        "MAE": metrics["MAE"],
        "SMAPE": metrics["SMAPE"],
        "Accuracy": metrics["Accuracy"],
    }


def main():
    print(f"[INFO] Đọc train.csv từ: {TRAIN_CSV}")
    df_raw = pd.read_csv(TRAIN_CSV)
    df = build_datetime(df_raw)

    hcmc_routes = load_hcmc_route_ids_from_geo()  # slug -> display_name

    results = []

    for slug, display_name in hcmc_routes.items():
        print("\n" + "=" * 80)
        print(f"[ROUTE] {display_name} (slug={slug})")
        print("=" * 80)

        try:
            s = build_congestion_series_for_slug(df, slug)
        except Exception as ex:
            print(f"  -> BỎ QUA: lỗi build_congestion_series_for_slug: {ex}")
            continue

        if len(s) <= LOOKBACK + 10:
            print(f"  -> BỎ QUA: quá ít điểm ({len(s)}) so với LOOKBACK={LOOKBACK}")
            continue

        F, y, t_index = build_feature_matrix(s)
        X, y_target, idx = build_sequences(F, y, t_index, LOOKBACK)

        X_train, X_test, y_train, y_test, idx_train, idx_test = time_series_train_test_split(
            X, y_target, idx, ratio=TRAIN_RATIO
        )
        print(f"  Seq tổng={len(X)}, Train={len(X_train)}, Test={len(X_test)}")

        # GRU
        res_gru = eval_one_model_for_route(
            display_name, slug, "GRU", X_test, y_test, idx_test
        )
        results.append(res_gru)

        # LSTM
        res_lstm = eval_one_model_for_route(
            display_name, slug, "LSTM", X_test, y_test, idx_test
        )
        results.append(res_lstm)

    df_res = pd.DataFrame(results)
    out_summary = os.path.join(OUTPUT_DIR, "hcmc_eval_summary.csv")
    df_res.to_csv(out_summary, index=False)
    print(f"\n[INFO] Đã lưu summary HCMC (GRU + LSTM, chỉ các route HCMC trong map): {out_summary}")
    print(df_res)


if __name__ == "__main__":
    main()
