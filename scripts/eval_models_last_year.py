#!/usr/bin/env python
"""
Đánh giá GRU / RNN / ARIMA trên 1 năm gần nhất cho 1 route,
lưu lại:
  - <route_id>_metrics_last_year.parquet  (metrics theo ngày)
  - <route_id>_top2_last_year.json       (summary + top-2 model)

Cách chạy (ví dụ):

  conda activate TrafficTrain

  # I-94
  python scripts/eval_models_last_year.py \
      --city Minneapolis \
      --zone I94 \
      --route "I-94-WB"

  # Fremont Total
  python scripts/eval_models_last_year.py \
      --city Seattle \
      --zone FremontBridge \
      --route "Fremont Bridge Total"
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# --- Thêm project root vào sys.path ---
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# --- Import module trong project ---
from modules.data_loader import load_slice
from modules.model_manager import load_model_context
from modules.model_utils import forecast_gru, forecast_rnn

try:
    # Nếu bạn đã tạo modules.arima_utils với forecast_arima_for_day
    from modules.arima_utils import forecast_arima_for_day

    HAS_ARIMA = True
except Exception:
    HAS_ARIMA = False


# =========================
# 1) Hàm tính metrics
# =========================
def compute_error_metrics(
    actual: np.ndarray,
    pred: np.ndarray,
) -> Optional[Dict[str, float]]:
    """Tính MSE, RMSE, MAE, MAPE, SMAPE, R2 cho 2 vector."""
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)

    mask = ~np.isnan(actual) & ~np.isnan(pred)
    actual = actual[mask]
    pred = pred[mask]

    if actual.size == 0:
        return None

    mse = float(np.mean((actual - pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(actual - pred)))

    # MAPE (bỏ những điểm actual = 0)
    mask_nz = actual != 0
    if np.any(mask_nz):
        mape = float(
            np.mean(
                np.abs(
                    (actual[mask_nz] - pred[mask_nz]) / actual[mask_nz]
                )
            )
            * 100.0
        )
    else:
        mape = float("nan")

    # SMAPE
    denom = np.abs(actual) + np.abs(pred)
    smape = float(
        np.mean(
            2.0 * np.abs(pred - actual)
            / np.where(denom == 0, 1.0, denom)
        )
        * 100.0
    )

    # R2
    ss_res = float(np.sum((actual - pred) ** 2))
    ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot != 0 else float("nan")

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "smape": smape,
        "r2": r2,
    }


# =========================
# 2) Evaluate 1 ngày
# =========================
def evaluate_one_day(
    city: str,
    zone: Optional[str],
    route_id: str,
    ctx,
    df_full_route: pd.DataFrame,
    day_start: pd.Timestamp,
    day_end: pd.Timestamp,
    include_arima: bool = True,
) -> List[Dict]:
    """
    Đánh giá GRU / RNN / (ARIMA) cho 1 ngày [day_start, day_end).

    Trả về list record metrics (mỗi record = 1 model).
    """
    records: List[Dict] = []

    # --- Actual ---
    df_actual = load_slice(
        city=city,
        zone=zone,
        routes=[route_id],
        start_dt=day_start,
        end_dt=day_end,
    )
    if df_actual.empty:
        return records

    df_actual = df_actual.copy()
    df_actual["DateTime"] = pd.to_datetime(
        df_actual["DateTime"], errors="coerce"
    )
    df_actual = df_actual.dropna(subset=["DateTime"])

    if df_actual.empty:
        return records

    df_actual_hourly = (
        df_actual.set_index("DateTime")["Vehicles"]
        .resample("1H")
        .mean()
        .dropna()
        .reset_index()
        .rename(columns={"Vehicles": "Actual"})
    )

    if df_actual_hourly.empty:
        return records

    # ---- Common context ----
    LOOKBACK = ctx.lookback
    META = ctx.meta
    SCALER = ctx.scaler
    ROUTES_MODEL = ctx.routes_model
    RID2IDX = ctx.rid2idx
    MODEL_GRU = ctx.gru_model
    MODEL_RNN = getattr(ctx, "rnn_model", None)

    # ---- Chuẩn bị history cho GRU/RNN ----
    df_hist = load_slice(
        city=city,
        zone=zone,
        routes=[route_id],
        start_dt=day_start - pd.Timedelta(hours=int(LOOKBACK)),
        end_dt=day_start,
    )

    # ========== GRU ==========
    if MODEL_GRU is not None:
        try:
            df_fc_gru, model_used_gru = forecast_gru(
                route_id=route_id,
                base_date=day_start,
                model=MODEL_GRU,
                meta=META,
                scaler=SCALER,
                routes_model=ROUTES_MODEL,
                rid2idx=RID2IDX,
                df_hist=df_hist,
            )
        except Exception as ex:
            print(f"[WARN] GRU forecast error at {day_start.date()}: {ex}")
            df_fc_gru = None

        if df_fc_gru is not None and not df_fc_gru.empty:
            df_g = df_fc_gru.copy()
            df_g["DateTime"] = pd.to_datetime(
                df_g["DateTime"], errors="coerce"
            )
            df_g = df_g.dropna(subset=["DateTime"])
            df_g = df_g[
                (df_g["DateTime"] >= day_start)
                & (df_g["DateTime"] < day_end)
            ]
            df_g = df_g.rename(
                columns={"PredictedVehicles": "Pred"}
            )

            merged = pd.merge(
                df_actual_hourly,
                df_g[["DateTime", "Pred"]],
                on="DateTime",
                how="inner",
            )
            if not merged.empty:
                mets = compute_error_metrics(
                    merged["Actual"].values, merged["Pred"].values
                )
                if mets is not None:
                    records.append(
                        {
                            "date": day_start.date().isoformat(),
                            "model": "GRU",
                            **mets,
                        }
                    )

    # ========== RNN ==========
    if MODEL_RNN is not None:
        try:
            df_fc_rnn, model_used_rnn = forecast_rnn(
                route_id=route_id,
                base_date=day_start,
                model=MODEL_RNN,
                meta=META,
                scaler=SCALER,
                routes_model=ROUTES_MODEL,
                rid2idx=RID2IDX,
                df_hist=df_hist,
            )
        except Exception as ex:
            print(f"[WARN] RNN forecast error at {day_start.date()}: {ex}")
            df_fc_rnn = None

        if df_fc_rnn is not None and not df_fc_rnn.empty:
            df_r = df_fc_rnn.copy()
            df_r["DateTime"] = pd.to_datetime(
                df_r["DateTime"], errors="coerce"
            )
            df_r = df_r.dropna(subset=["DateTime"])
            df_r = df_r[
                (df_r["DateTime"] >= day_start)
                & (df_r["DateTime"] < day_end)
            ]
            df_r = df_r.rename(
                columns={"PredictedVehicles": "Pred"}
            )

            merged = pd.merge(
                df_actual_hourly,
                df_r[["DateTime", "Pred"]],
                on="DateTime",
                how="inner",
            )
            if not merged.empty:
                mets = compute_error_metrics(
                    merged["Actual"].values, merged["Pred"].values
                )
                if mets is not None:
                    records.append(
                        {
                            "date": day_start.date().isoformat(),
                            "model": "RNN",
                            **mets,
                        }
                    )

    # ========== ARIMA (optional) ==========
    if include_arima and HAS_ARIMA:
        try:
            df_fc_arima, info = forecast_arima_for_day(
                df_full=df_full_route,
                day_start=day_start,
                day_end=day_end,
                value_col="Vehicles",
            )
        except Exception as ex:
            print(f"[WARN] ARIMA forecast error at {day_start.date()}: {ex}")
            df_fc_arima = None

        if df_fc_arima is not None and not df_fc_arima.empty:
            df_a = df_fc_arima.copy()
            df_a["DateTime"] = pd.to_datetime(
                df_a["DateTime"], errors="coerce"
            )
            df_a = df_a.dropna(subset=["DateTime"])
            df_a = df_a[
                (df_a["DateTime"] >= day_start)
                & (df_a["DateTime"] < day_end)
            ]
            df_a = df_a.rename(
                columns={"Pred_ARIMA": "Pred"}
            )

            merged = pd.merge(
                df_actual_hourly,
                df_a[["DateTime", "Pred"]],
                on="DateTime",
                how="inner",
            )
            if not merged.empty:
                mets = compute_error_metrics(
                    merged["Actual"].values, merged["Pred"].values
                )
                if mets is not None:
                    records.append(
                        {
                            "date": day_start.date().isoformat(),
                            "model": "ARIMA",
                            **mets,
                        }
                    )

    return records


# =========================
# 3) Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True, help="City name")
    parser.add_argument(
        "--zone",
        required=True,
        help="Zone name (dùng 'I94', 'FremontBridge', ...)",
    )
    parser.add_argument(
        "--route",
        required=True,
        help='RouteId (ví dụ "I-94-WB", "Fremont Bridge Total")',
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Số ngày gần nhất để evaluate (default: 365)",
    )
    parser.add_argument(
        "--no-arima",
        action="store_true",
        help="Bỏ qua ARIMA trong evaluation",
    )
    args = parser.parse_args()

    city = args.city
    zone = None if args.zone == "(All)" else args.zone
    route_id = args.route
    last_n_days = args.days
    include_arima = (not args.no_arima) and HAS_ARIMA

    print(f"== Evaluate last {last_n_days} days ==")
    print(f"City={city}, Zone={zone}, RouteId={route_id}")
    if include_arima:
        print("Models: GRU, RNN, ARIMA")
    else:
        print("Models: GRU, RNN")

    # --- Load model context (GRU/RNN + meta + scaler) ---
    ctx = load_model_context(city, zone)
    family_name = ctx.family_name  # vd 'I94', 'Seattle_FremontBridge'

    # --- Load full series cho route ---
    df_full = load_slice(
        city=city,
        zone=zone,
        routes=[route_id],
        start_dt=None,
        end_dt=None,
    )
    if df_full.empty:
        print("!! df_full empty, thoát")
        return

    df_full = df_full.copy()
    df_full["DateTime"] = pd.to_datetime(
        df_full["DateTime"], errors="coerce"
    )
    df_full = df_full.dropna(subset=["DateTime"]).sort_values("DateTime")

    min_dt = df_full["DateTime"].min().normalize()
    max_dt = df_full["DateTime"].max().normalize()

    # --- Define eval window: last_n_days trước max_dt ---
    eval_start = max_dt - pd.Timedelta(days=last_n_days - 1)
    if eval_start < min_dt:
        eval_start = min_dt + pd.Timedelta(days=1)

    all_days = pd.date_range(
        start=eval_start,
        end=max_dt - pd.Timedelta(days=1),
        freq="D",
    )

    print(
        f"Evaluate từ {all_days[0].date()} đến {all_days[-1].date()} "
        f"({len(all_days)} ngày)"
    )

    all_records: List[Dict] = []

    for d in all_days:
        day_start = d.normalize()
        day_end = day_start + pd.Timedelta(days=1)
        print(f"  - Day {day_start.date()} ...", end="", flush=True)

        recs = evaluate_one_day(
            city=city,
            zone=zone,
            route_id=route_id,
            ctx=ctx,
            df_full_route=df_full,
            day_start=day_start,
            day_end=day_end,
            include_arima=include_arima,
        )
        if recs:
            all_records.extend(recs)
            print(f" OK ({len(recs)} model)")
        else:
            print(" skip (no data / no preds)")

    if not all_records:
        print("!! Không có record nào, thoát.")
        return

    df_eval = pd.DataFrame(all_records)
    print()
    print("== Sample records ==")
    print(df_eval.head())

    # --- Save metrics per day ---
    model_dir = Path("model") / family_name
    model_dir.mkdir(parents=True, exist_ok=True)

    metrics_daily_path = model_dir / f"{route_id}_metrics_last_year.parquet"
    df_eval.to_parquet(metrics_daily_path, index=False)
    print(f"Saved daily metrics to: {metrics_daily_path}")

    # --- Average per model ---
    agg = (
        df_eval.groupby("model")
        .agg(
            mse_mean=("mse", "mean"),
            rmse_mean=("rmse", "mean"),
            mae_mean=("mae", "mean"),
            mape_mean=("mape", "mean"),
            smape_mean=("smape", "mean"),
            r2_mean=("r2", "mean"),
        )
        .reset_index()
    )

    print()
    print("== Average metrics (last year) ==")
    print(agg)

    # --- Chọn top-2 model theo SMAPE (fallback RMSE) ---
    def score_row(row) -> float:
        smape = row["smape_mean"]
        rmse = row["rmse_mean"]
        if not math.isnan(smape):
            return float(smape)
        return float(rmse)

    agg["score"] = agg.apply(score_row, axis=1)
    agg_sorted = agg.sort_values("score", ascending=True)

    # top_k = min(2, số model có metrics)
    top_k = min(2, len(agg_sorted))
    top_models = agg_sorted["model"].tolist()[:top_k]

    # Prepare JSON summary
    metrics_avg = {}
    for _, row in agg_sorted.iterrows():
        m = row["model"]
        metrics_avg[m] = {
            "mse": float(row["mse_mean"]),
            "rmse": float(row["rmse_mean"]),
            "mae": float(row["mae_mean"]),
            "mape": float(row["mape_mean"]),
            "smape": float(row["smape_mean"]),
            "r2": float(row["r2_mean"]),
            "score": float(row["score"]),
        }

    summary = {
        "city": city,
        "zone": zone,
        "route_id": route_id,
        "family_name": family_name,
        "eval_start_date": str(all_days[0].date()),
        "eval_end_date": str(all_days[-1].date()),
        "n_days_evaluated": int(len(all_days)),
        "top_models": top_models,  # ví dụ ["GRU", "RNN"]
        "metrics_avg": metrics_avg,
    }

    summary_path = model_dir / f"{route_id}_top2_last_year.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print()
    print(f"Saved summary to: {summary_path}")
    print(f"Top models: {top_models}")


if __name__ == "__main__":
    main()
