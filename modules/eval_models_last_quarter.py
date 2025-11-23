#!/usr/bin/env python
"""
Evaluate GRU / RNN / LSTM / ARIMA / SARIMA trên 1 năm gần nhất cho 1 route.

Sinh file:
  model/<family_name>/<route_id>_top2_last_quarter.json

để app.py dùng trong load_top2_summary() cho ensemble forecast.
"""

import os
import sys
import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- init sys.path để import modules.* ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from functools import lru_cache
from modules.data_loader import load_slice
from modules.model_manager import load_model_context
from modules.model_utils import forecast_gru, forecast_rnn, forecast_lstm
from modules.arima_utils import forecast_arima_for_day
from modules.sarima_utils import forecast_sarima_for_day

import joblib
import tensorflow as tf


@lru_cache(maxsize=None)
def load_lstm_artifacts(city: str, zone: str | None):
    """
    Load LSTM model theo city/zone giống flow GRU/RNN.

    Tức là tìm trong:
        model/<City>_<Zone>/
        model/<City>/
        model/

    Và trả về:
        { "model", "meta", "scaler", "routes", "rid2idx" }
    hoặc None nếu không tìm thấy.
    """
    from pathlib import Path
    import json
    import joblib
    from tensorflow.keras.models import load_model

    base = Path("model")
    city_str = (city or "").replace(" ", "_")
    zone_str = (zone or "").replace(" ", "_") if zone else None

    # Build danh sách folder theo thứ tự ưu tiên
    candidates = []

    if zone_str and zone_str != "(All)":
        candidates.append(base / f"{city_str}_{zone_str}")

    candidates.append(base / city_str)
    candidates.append(base)

    for d in candidates:
        meta_path = d / "lstm_meta.json"
        model_path = d / "traffic_lstm.keras"
        scaler_path = d / "vehicles_scaler.pkl"

        if meta_path.exists() and model_path.exists() and scaler_path.exists():
            print(f"[LSTM] Using model dir: {d}")

            meta = json.load(open(meta_path, "r"))
            model = load_model(model_path)
            scaler = joblib.load(scaler_path)

            routes = list(meta.get("routes", []))
            rid2idx = {rid: i for i, rid in enumerate(routes)}

            return {
                "model": model,
                "meta": meta,
                "scaler": scaler,
                "routes": routes,
                "rid2idx": rid2idx,
                "dir": str(d),
            }

    print(f"[LSTM] No valid LSTM model found for city={city}, zone={zone}")
    return None



def compute_metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    mse = mean_squared_error(actual, pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(actual, pred)

    if np.any(actual != 0):
        mape = (
            np.mean(np.abs((actual - pred)[actual != 0] / actual[actual != 0]))
            * 100.0
        )
    else:
        mape = np.nan

    denom = np.abs(actual) + np.abs(pred)
    smape = (
        np.mean(2.0 * np.abs(pred - actual) / np.where(denom == 0, 1.0, denom))
        * 100.0
    )

    r2 = r2_score(actual, pred)

    return {
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "MAPE": float(mape) if not np.isnan(mape) else None,
        "SMAPE": float(smape),
        "R2": float(r2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True, help="City name, ví dụ: Minneapolis")
    parser.add_argument("--zone", required=True, help="Zone, ví dụ: I94 hoặc FremontBridge")
    parser.add_argument("--route", required=True, help="RouteId, ví dụ: I-94-WB")
    parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="Số ngày gần nhất để evaluate (mặc định 90)",
    )
    args = parser.parse_args()

    city = args.city
    zone = None if args.zone == "(All)" else args.zone
    route_id = args.route
    n_days_back = args.days

    print(f"=== Evaluate models for city={city}, zone={zone}, route={route_id} ===")

    # ---- Load full data của route ----
    df_full = load_slice(
        city=city,
        zone=zone,
        routes=[route_id],
        start_dt=None,
        end_dt=None,
    )
    if df_full is None or df_full.empty:
        print("⚠️ Không có dữ liệu cho route này.")
        return

    df_full = df_full.copy()
    df_full["DateTime"] = pd.to_datetime(df_full["DateTime"], errors="coerce")
    df_full = df_full.dropna(subset=["DateTime"])
    df_full = df_full.sort_values("DateTime")

    min_dt = df_full["DateTime"].min().normalize()
    max_dt = df_full["DateTime"].max().normalize()

    start_candidate = max_dt - pd.Timedelta(days=n_days_back - 1)
    start_eval = max(min_dt, start_candidate)

    dates = pd.date_range(start_eval, max_dt, freq="D")
    print(f"Evaluate từ {start_eval.date()} đến {max_dt.date()} ({len(dates)} ngày)")

    # ---- Load model context cho GRU/RNN ----
    ctx = load_model_context(city, zone)
    LOOKBACK = int(ctx.lookback)

    lstm_ctx = load_lstm_artifacts()
    print("[DEBUG] LSTM DIR =", lstm_ctx["dir"] if lstm_ctx else "None")

    # Sẽ lưu list metrics theo ngày cho từng model
    per_model_metrics: dict[str, list[dict]] = {
        "GRU": [],
        "RNN": [],
        "LSTM": [],
        "ARIMA": [],
        "SARIMA": [],
    }

    for day_start in dates:
        day_end = day_start + pd.Timedelta(days=1)

        # Actual
        df_day = df_full[
            (df_full["DateTime"] >= day_start)
            & (df_full["DateTime"] < day_end)
        ].copy()
        if df_day.empty:
            continue

        df_actual_hourly = (
            df_day.set_index("DateTime")["Vehicles"]
            .resample("1H")
            .mean()
            .dropna()
        )
        if df_actual_hourly.empty:
            continue

        actual = df_actual_hourly.values.astype(float)

        # ---- Chuẩn bị history cho seq models ----
        hist_start = day_start - pd.Timedelta(hours=LOOKBACK)
        df_hist = df_full[
            (df_full["DateTime"] >= hist_start)
            & (df_full["DateTime"] < day_start)
        ].copy()

        # === GRU ===
        try:
            df_fc_gru, _ = forecast_gru(
                route_id=route_id,
                base_date=day_start,
                model=ctx.gru_model,
                meta=ctx.meta,
                scaler=ctx.scaler,
                routes_model=ctx.routes_model,
                rid2idx=ctx.rid2idx,
                df_hist=df_hist,
            )
            if df_fc_gru is not None and not df_fc_gru.empty:
                df_gru = df_fc_gru.copy()
                df_gru["DateTime"] = pd.to_datetime(
                    df_gru["DateTime"], errors="coerce"
                )
                df_gru = df_gru.dropna(subset=["DateTime"])
                df_gru = df_gru[
                    (df_gru["DateTime"] >= day_start)
                    & (df_gru["DateTime"] < day_end)
                ]
                df_merge = df_actual_hourly.reset_index().merge(
                    df_gru[["DateTime", "PredictedVehicles"]],
                    on="DateTime",
                    how="inner",
                )
                if not df_merge.empty:
                    pred = df_merge["PredictedVehicles"].values.astype(float)
                    act = df_merge["Vehicles"].values.astype(float)
                    per_model_metrics["GRU"].append(
                        compute_metrics(act, pred)
                    )
        except Exception as ex:
            print(f"[GRU][{day_start.date()}] error: {ex}")

        # === RNN (nếu có) ===
        if getattr(ctx, "rnn_model", None) is not None:
            try:
                df_fc_rnn, _ = forecast_rnn(
                    route_id=route_id,
                    base_date=day_start,
                    model=ctx.rnn_model,
                    meta=ctx.meta,
                    scaler=ctx.scaler,
                    routes_model=ctx.routes_model,
                    rid2idx=ctx.rid2idx,
                    df_hist=df_hist,
                )
                if df_fc_rnn is not None and not df_fc_rnn.empty:
                    df_rnn = df_fc_rnn.copy()
                    df_rnn["DateTime"] = pd.to_datetime(
                        df_rnn["DateTime"], errors="coerce"
                    )
                    df_rnn = df_rnn.dropna(subset=["DateTime"])
                    df_rnn = df_rnn[
                        (df_rnn["DateTime"] >= day_start)
                        & (df_rnn["DateTime"] < day_end)
                    ]
                    df_merge = df_actual_hourly.reset_index().merge(
                        df_rnn[["DateTime", "PredictedVehicles"]],
                        on="DateTime",
                        how="inner",
                    )
                    if not df_merge.empty:
                        pred = df_merge["PredictedVehicles"].values.astype(float)
                        act = df_merge["Vehicles"].values.astype(float)
                        per_model_metrics["RNN"].append(
                            compute_metrics(act, pred)
                        )
            except Exception as ex:
                print(f"[RNN][{day_start.date()}] error: {ex}")

        # === LSTM (nếu có artifacts) ===
        if lstm_ctx is not None:
            try:
                df_fc_lstm, _ = forecast_lstm(
                    route_id=route_id,
                    base_date=day_start,
                    model=lstm_ctx["model"],
                    meta=lstm_ctx["meta"],
                    scaler=lstm_ctx["scaler"],
                    routes_model=lstm_ctx["routes"],
                    rid2idx=lstm_ctx["rid2idx"],
                    df_hist=df_hist,
                )
                if df_fc_lstm is not None and not df_fc_lstm.empty:
                    df_lstm = df_fc_lstm.copy()
                    df_lstm["DateTime"] = pd.to_datetime(
                        df_lstm["DateTime"], errors="coerce"
                    )
                    df_lstm = df_lstm.dropna(subset=["DateTime"])
                    df_lstm = df_lstm[
                        (df_lstm["DateTime"] >= day_start)
                        & (df_lstm["DateTime"] < day_end)
                    ]
                    df_merge = df_actual_hourly.reset_index().merge(
                        df_lstm[["DateTime", "PredictedVehicles"]],
                        on="DateTime",
                        how="inner",
                    )
                    if not df_merge.empty:
                        pred = df_merge["PredictedVehicles"].values.astype(float)
                        act = df_merge["Vehicles"].values.astype(float)
                        per_model_metrics["LSTM"].append(
                            compute_metrics(act, pred)
                        )
            except Exception as ex:
                print(f"[LSTM][{day_start.date()}] error: {ex}")

        # === ARIMA ===
        # try:
        #     df_fc_arima, info_arima = forecast_arima_for_day(
        #         df_full=df_full,
        #         day_start=day_start,
        #         day_end=day_end,
        #         value_col="Vehicles",
        #     )
        #     if df_fc_arima is not None and not df_fc_arima.empty:
        #         df_arima = df_fc_arima.copy()
        #         df_arima["DateTime"] = pd.to_datetime(
        #             df_arima["DateTime"], errors="coerce"
        #         )
        #         df_arima = df_arima.dropna(subset=["DateTime"])
        #         df_merge = df_actual_hourly.reset_index().merge(
        #             df_arima[["DateTime", "Pred_ARIMA"]],
        #             on="DateTime",
        #             how="inner",
        #         )
        #         if not df_merge.empty:
        #             pred = df_merge["Pred_ARIMA"].values.astype(float)
        #             act = df_merge["Vehicles"].values.astype(float)
        #             per_model_metrics["ARIMA"].append(
        #                 compute_metrics(act, pred)
        #             )
        # except Exception as ex:
        #     print(f"[ARIMA][{day_start.date()}] error: {ex}")
        #
        # # === SARIMA ===
        # try:
        #     df_fc_sarima, info_sarima = forecast_sarima_for_day(
        #         df_full=df_full,
        #         day_start=day_start,
        #         day_end=day_end,
        #         value_col="Vehicles",
        #     )
        #     if df_fc_sarima is not None and not df_fc_sarima.empty:
        #         df_sarima = df_fc_sarima.copy()
        #         df_sarima["DateTime"] = pd.to_datetime(
        #             df_sarima["DateTime"], errors="coerce"
        #         )
        #         df_sarima = df_sarima.dropna(subset=["DateTime"])
        #         df_merge = df_actual_hourly.reset_index().merge(
        #             df_sarima[["DateTime", "Pred_SARIMA"]],
        #             on="DateTime",
        #             how="inner",
        #         )
        #         if not df_merge.empty:
        #             pred = df_merge["Pred_SARIMA"].values.astype(float)
        #             act = df_merge["Vehicles"].values.astype(float)
        #             per_model_metrics["SARIMA"].append(
        #                 compute_metrics(act, pred)
        #             )
        # except Exception as ex:
        #     print(f"[SARIMA][{day_start.date()}] error: {ex}")

    # ---- Aggregate metrics theo model ----
    agg_metrics = {}
    for model_name, daily_list in per_model_metrics.items():
        if not daily_list:
            continue
        df_m = pd.DataFrame(daily_list)
        agg = df_m.mean(numeric_only=True).to_dict()
        agg["n_days"] = int(len(daily_list))
        agg_metrics[model_name] = agg

    if not agg_metrics:
        print("⚠️ Không có model nào evaluate được.")
        return

    # Rank model theo MAE tăng dần
    valid_models = [
        (m, v) for m, v in agg_metrics.items() if "MAE" in v and v["MAE"] is not None
    ]
    valid_models.sort(key=lambda x: x[1]["MAE"])

    top_models = [m for m, _ in valid_models[:2]]
    print("Top models theo MAE:", top_models)

    # ---- Save summary JSON ----
    family_name = ctx.family_name
    out_dir = Path(ROOT_DIR) / "model" / family_name
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{route_id}_top2_last_quarter.json"

    summary = {
        "city": city,
        "zone": args.zone,
        "route_id": route_id,
        "family_name": family_name,
        "days_back": n_days_back,
        "start_eval": str(start_eval.date()),
        "end_eval": str(max_dt.date()),
        "top_models": top_models,
        "metrics": agg_metrics,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved summary to {out_path}")


if __name__ == "__main__":
    main()
