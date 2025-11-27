#!/usr/bin/env python
"""
Precompute daily 3-months traffic (Actual vs Models) cho mọi route.

Output cho mỗi route:
    model/<family_name>/<route_id>_daily_3months.parquet

Trong đó có các cột:
    Date
    DailyActual
    Daily_GRU (nếu có)
    Daily_RNN (nếu có)
    Daily_LSTM (nếu có LSTM artifacts)
    Daily_ARIMA (nếu forecast_arima_for_day chạy được)
    Daily_SARIMA (nếu forecast_sarima_for_day chạy được)
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

from modules.data_loader import load_slice, list_cities, list_zones, list_routes
from modules.model_manager import load_model_context
from modules.model_utils import forecast_gru, forecast_rnn, forecast_lstm

# ---- ARIMA / SARIMA (tùy có hay không) ----
try:
    from modules.arima_utils import forecast_arima_for_day

    HAS_ARIMA = True
except Exception:
    forecast_arima_for_day = None
    HAS_ARIMA = False

try:
    from modules.sarima_utils import forecast_sarima_for_day

    HAS_SARIMA = True
except Exception:
    forecast_sarima_for_day = None
    HAS_SARIMA = False


def load_lstm_artifacts_for_family(family_name: str):
    """
    Load LSTM artifacts trong model/<family_name>/:
      - lstm_meta.json
      - traffic_lstm.keras
      - vehicles_scaler.pkl (joblib)

    Nếu thiếu hoặc lỗi -> trả về None (bỏ qua LSTM cho family này).
    """
    import json

    base_dir = Path("model") / family_name
    meta_path = base_dir / "lstm_meta.json"
    model_path = base_dir / "traffic_lstm.keras"
    scaler_path = base_dir / "vehicles_scaler.pkl"

    if not (meta_path.exists() and model_path.exists() and scaler_path.exists()):
        logging.info(
            f"[LSTM] Missing artifacts for family={family_name} in {base_dir} "
            f"(meta={meta_path.exists()}, model={model_path.exists()}, scaler={scaler_path.exists()})"
        )
        return None

    try:
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

        model = load_model(model_path)
        scaler = joblib.load(scaler_path)

        routes = list(meta.get("routes", []))
        rid2idx = {rid: idx for idx, rid in enumerate(routes)}

        logging.info(
            f"[LSTM] Loaded LSTM artifacts for family={family_name} from {base_dir}"
        )

        return {
            "model": model,
            "meta": meta,
            "scaler": scaler,
            "routes": routes,
            "rid2idx": rid2idx,
            "dir": base_dir,
        }
    except Exception as ex:
        logging.warning(
            f"[LSTM] Không load được LSTM artifacts cho family={family_name} ở {base_dir}: {ex}"
        )
        return None


def compute_daily_for_route(
    city: str,
    zone: str,
    ctx,
    lstm_artifacts,
    route_id: str,
    days_back: int,
    overwrite: bool = False,
):
    """
    Tính daily traffic N ngày gần nhất (thông thường 90 ngày) cho 1 route
    và cache ra parquet: model/<family_name>/<route_id>_daily_3months.parquet
    """

    family_name = ctx.family_name
    cache_dir = Path("model") / family_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{route_id}_daily_3months.parquet"

    if cache_path.exists() and not overwrite:
        logging.info(f"[SKIP] Đã có cache: {cache_path}")
        return

    logging.info(
        f"[START] city={city}, zone={zone}, route={route_id}, family={family_name}"
    )

    # 1) Load full data của route
    df_full_route = load_slice(
        city=city,
        zone=zone,
        routes=[route_id],
        start_dt=None,
        end_dt=None,
    )
    if df_full_route is None or df_full_route.empty:
        logging.warning(
            f"[WARN] Không có dữ liệu cho city={city}, zone={zone}, route={route_id}"
        )
        return

    df_full_route = df_full_route.copy()
    df_full_route["DateTime"] = pd.to_datetime(
        df_full_route["DateTime"], errors="coerce"
    )
    df_full_route = df_full_route.dropna(subset=["DateTime"])
    if df_full_route.empty:
        logging.warning(
            f"[WARN] Dữ liệu DateTime không hợp lệ cho route={route_id}, bỏ qua."
        )
        return

    df_full_route = df_full_route.sort_values("DateTime")

    # 2) Xác định N ngày gần nhất dựa trên ngày cuối cùng trong data
    max_dt_norm = df_full_route["DateTime"].max().normalize()
    start_n_days = max_dt_norm - pd.Timedelta(days=days_back)

    df_last = df_full_route[df_full_route["DateTime"] >= start_n_days].copy()
    if df_last.empty:
        logging.warning(
            f"[WARN] Không có dữ liệu trong {days_back} ngày gần nhất cho route={route_id}."
        )
        return

    df_last["Date"] = df_last["DateTime"].dt.normalize()

    # ==== Actual daily ====
    df_daily_actual = (
        df_last.groupby("Date", as_index=False)["Vehicles"]
        .sum()
        .rename(columns={"Vehicles": "DailyActual"})
        .sort_values("Date")
    )
    if df_daily_actual.empty:
        logging.warning(
            f"[WARN] Không gom được DailyActual cho route={route_id}."
        )
        return

    dates = df_daily_actual["Date"].tolist()
    df_eval = df_daily_actual.set_index("Date")

    # Chuẩn bị context cho model
    LOOKBACK = ctx.lookback
    MODEL_GRU = ctx.gru_model
    MODEL_RNN = getattr(ctx, "rnn_model", None)
    META = ctx.meta
    SCALER = ctx.scaler
    ROUTES_MODEL = ctx.routes_model
    RID2IDX = ctx.rid2idx

    # ---- Helper seq model: GRU / RNN / LSTM ----
    def _compute_seq_model_daily(model_name: str):
        records = []

        if model_name == "GRU" and MODEL_GRU is None:
            return None
        if model_name == "RNN" and MODEL_RNN is None:
            return None
        if model_name == "LSTM" and lstm_artifacts is None:
            return None

        for d in dates:
            day_start = pd.Timestamp(d).normalize()
            day_end = day_start + pd.Timedelta(days=1)

            # history trước base_date
            hist_start = day_start - pd.Timedelta(hours=LOOKBACK)
            df_hist = df_full_route[
                (df_full_route["DateTime"] >= hist_start)
                & (df_full_route["DateTime"] < day_start)
            ].copy()
            if df_hist.empty:
                continue

            try:
                if model_name == "GRU":
                    df_fc, _used = forecast_gru(
                        route_id=route_id,
                        base_date=day_start,
                        model=MODEL_GRU,
                        meta=META,
                        scaler=SCALER,
                        routes_model=ROUTES_MODEL,
                        rid2idx=RID2IDX,
                        df_hist=df_hist,
                    )
                elif model_name == "RNN":
                    df_fc, _used = forecast_rnn(
                        route_id=route_id,
                        base_date=day_start,
                        model=MODEL_RNN,
                        meta=META,
                        scaler=SCALER,
                        routes_model=ROUTES_MODEL,
                        rid2idx=RID2IDX,
                        df_hist=df_hist,
                    )
                elif model_name == "LSTM":
                    df_fc, _used = forecast_lstm(
                        route_id=route_id,
                        base_date=day_start,
                        model=lstm_artifacts["model"],
                        meta=lstm_artifacts["meta"],
                        scaler=lstm_artifacts["scaler"],
                        routes_model=lstm_artifacts["routes"],
                        rid2idx=lstm_artifacts["rid2idx"],
                        df_hist=df_hist,
                    )
                else:
                    return None
            except Exception as ex:
                logging.warning(
                    f"[WARN] {model_name} forecast error route={route_id}, "
                    f"date={day_start.date()}: {ex}"
                )
                continue

            if df_fc is None or df_fc.empty:
                continue

            df_day = df_fc.copy()
            df_day["DateTime"] = pd.to_datetime(
                df_day["DateTime"], errors="coerce"
            )
            df_day = df_day.dropna(subset=["DateTime"])
            df_day = df_day[
                (df_day["DateTime"] >= day_start)
                & (df_day["DateTime"] < day_end)
            ]
            if df_day.empty:
                continue

            if "PredictedVehicles" not in df_day.columns:
                continue

            v = float(df_day["PredictedVehicles"].sum())
            records.append({"Date": day_start, "DailyPred": v})

        if not records:
            return None

        df_m = (
            pd.DataFrame(records)
            .groupby("Date", as_index=False)["DailyPred"]
            .mean()
            .rename(columns={"DailyPred": f"Daily_{model_name}"})
        )
        return df_m.set_index("Date")

    # ---- GRU / RNN / LSTM ----
    for m in ["GRU", "RNN", "LSTM"]:
        df_m = _compute_seq_model_daily(m)
        if df_m is not None:
            df_eval = df_eval.join(df_m, how="left")

    # ---- ARIMA ----
    if HAS_ARIMA and forecast_arima_for_day is not None:
        records = []
        for d in dates:
            day_start = pd.Timestamp(d).normalize()
            day_end = day_start + pd.Timedelta(days=1)

            try:
                # signature: forecast_arima_for_day(df_full, day_start, day_end)
                df_fc_arima, _info = forecast_arima_for_day(
                    df_full_route,
                    day_start,
                    day_end,
                )
            except Exception as ex:
                logging.warning(
                    f"[WARN] ARIMA error route={route_id}, date={day_start.date()}: {ex}"
                )
                continue

            if df_fc_arima is None or df_fc_arima.empty:
                continue

            df_a = df_fc_arima.copy()
            df_a["DateTime"] = pd.to_datetime(
                df_a["DateTime"], errors="coerce"
            )
            df_a = df_a.dropna(subset=["DateTime"])
            df_a = df_a[
                (df_a["DateTime"] >= day_start)
                & (df_a["DateTime"] < day_end)
            ]
            if df_a.empty:
                continue

            pred_col = (
                "Pred_ARIMA"
                if "Pred_ARIMA" in df_a.columns
                else "PredictedVehicles"
            )
            if pred_col not in df_a.columns:
                continue

            v = float(df_a[pred_col].sum())
            records.append({"Date": day_start, "DailyPred": v})

        if records:
            df_arima = (
                pd.DataFrame(records)
                .groupby("Date", as_index=False)["DailyPred"]
                .mean()
                .rename(columns={"DailyPred": "Daily_ARIMA"})
            ).set_index("Date")
            df_eval = df_eval.join(df_arima, how="left")

    # ---- SARIMA (nếu có) ----
    if HAS_SARIMA and forecast_sarima_for_day is not None:
        records = []
        for d in dates:
            day_start = pd.Timestamp(d).normalize()
            day_end = day_start + pd.Timedelta(days=1)

            try:
                df_fc_sarima, _info = forecast_sarima_for_day(
                    df_full_route,
                    day_start,
                    day_end,
                )
            except Exception as ex:
                logging.warning(
                    f"[WARN] SARIMA error route={route_id}, date={day_start.date()}: {ex}"
                )
                continue

            if df_fc_sarima is None or df_fc_sarima.empty:
                continue

            df_s = df_fc_sarima.copy()
            df_s["DateTime"] = pd.to_datetime(
                df_s["DateTime"], errors="coerce"
            )
            df_s = df_s.dropna(subset=["DateTime"])
            df_s = df_s[
                (df_s["DateTime"] >= day_start)
                & (df_s["DateTime"] < day_end)
            ]
            if df_s.empty:
                continue

            pred_col = (
                "Pred_SARIMA"
                if "Pred_SARIMA" in df_s.columns
                else "PredictedVehicles"
            )
            if pred_col not in df_s.columns:
                continue

            v = float(df_s[pred_col].sum())
            records.append({"Date": day_start, "DailyPred": v})

        if records:
            df_sarima = (
                pd.DataFrame(records)
                .groupby("Date", as_index=False)["DailyPred"]
                .mean()
                .rename(columns={"DailyPred": "Daily_SARIMA"})
            ).set_index("Date")
            df_eval = df_eval.join(df_sarima, how="left")

    # 3) Ghi cache
    df_out = df_eval.reset_index()
    try:
        df_out.to_parquet(cache_path, index=False)
        logging.info(f"[OK] Ghi cache xong: {cache_path}")
    except Exception as ex:
        logging.error(f"[ERR] Không ghi được {cache_path}: {ex}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Precompute daily 3-months traffic (Actual vs Models) "
            "cho mọi route trong data/processed_ds."
        )
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=30,
        help="Số ngày gần nhất để tính (mặc định: 90).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Nếu set, sẽ ghi đè file cache nếu đã tồn tại.",
    )
    parser.add_argument(
        "--city",
        type=str,
        default=None,
        help="Chỉ chạy cho 1 city (tên thư mục trong data/processed_ds). Nếu bỏ trống thì chạy tất cả.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    cities = list_cities()
    if args.city:
        cities = [c for c in cities if c == args.city]
        if not cities:
            logging.error(f"Không tìm thấy city={args.city} trong data/processed_ds.")
            return

    for city in cities:
        logging.info(f"==== CITY: {city} ====")
        zones = list_zones(city)
        # bỏ qua "(All)" vì route sẽ được xử lý theo từng zone cụ thể
        zones = [z for z in zones if z != "(All)"]

        if not zones:
            logging.warning(f"[WARN] City={city} không có zone cụ thể.")
            continue

        for zone in zones:
            logging.info(f"-- Zone: {city} / {zone} --")
            try:
                ctx = load_model_context(city, zone)
            except FileNotFoundError as ex:
                logging.warning(
                    f"[WARN] Không load được model context cho city={city}, zone={zone}: {ex}"
                )
                continue

            family_name = ctx.family_name
            lstm_artifacts = load_lstm_artifacts_for_family(family_name)

            routes = list_routes(city, zone)
            if not routes:
                logging.warning(
                    f"[WARN] Không có route nào cho city={city}, zone={zone}"
                )
                continue

            for rid in routes:
                compute_daily_for_route(
                    city=city,
                    zone=zone,
                    ctx=ctx,
                    lstm_artifacts=lstm_artifacts,
                    route_id=rid,
                    days_back=args.days_back,
                    overwrite=args.overwrite,
                )


if __name__ == "__main__":
    main()
