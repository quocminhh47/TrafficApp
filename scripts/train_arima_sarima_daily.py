"""
Train nhanh ARIMA/SARIMA trên chuỗi giao thông và lưu dự báo daily ra parquet.

Ví dụ chạy:
    python scripts/train_arima_sarima_daily.py --city hcmc --zone duong --route ly_thuong_kiet \
        --days-back 90 --output outputs/arima_sarima_daily.parquet

Script sẽ lấy dữ liệu thô qua data_loader, gom daily actual cho N ngày gần nhất,
chạy ARIMA và SARIMA cho từng ngày, rồi lưu các cột:
    Date, DailyActual, Daily_ARIMA, Daily_SARIMA
"""
import argparse
from pathlib import Path

import pandas as pd

from modules.data_loader import load_slice

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


def compute_daily_predictions(city: str, zone: str | None, route: str, days_back: int) -> pd.DataFrame | None:
    df_full = load_slice(city=city, zone=zone, routes=[route], start_dt=None, end_dt=None)
    if df_full is None or df_full.empty:
        print("[WARN] Không có dữ liệu đầu vào.")
        return None

    df = df_full.copy()
    df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
    df = df.dropna(subset=["DateTime"])
    if df.empty:
        print("[WARN] Dữ liệu DateTime không hợp lệ.")
        return None

    df = df.sort_values("DateTime")
    max_dt = df["DateTime"].max().normalize()
    start_dt = max_dt - pd.Timedelta(days=days_back)
    df_last = df[df["DateTime"] >= start_dt].copy()
    if df_last.empty:
        print(f"[WARN] Không có dữ liệu trong {days_back} ngày gần nhất.")
        return None

    df_last["Date"] = df_last["DateTime"].dt.normalize()
    df_daily_actual = (
        df_last.groupby("Date", as_index=False)["Vehicles"]
        .sum()
        .rename(columns={"Vehicles": "DailyActual"})
        .sort_values("Date")
    )

    records_arima: list[dict] = []
    records_sarima: list[dict] = []

    for date_val in df_daily_actual["Date"]:
        day_start = pd.Timestamp(date_val)
        day_end = day_start + pd.Timedelta(days=1)

        if HAS_ARIMA and forecast_arima_for_day is not None:
            try:
                df_fc_arima, _ = forecast_arima_for_day(df_full, day_start, day_end)
                if df_fc_arima is not None and not df_fc_arima.empty:
                    v = float(df_fc_arima["Pred_ARIMA"].sum())
                    records_arima.append({"Date": day_start, "Daily_ARIMA": v})
            except Exception as ex:  # pragma: no cover - logging helper
                print(f"[ARIMA] Lỗi {day_start.date()}: {ex}")

        if HAS_SARIMA and forecast_sarima_for_day is not None:
            try:
                df_fc_sarima, _ = forecast_sarima_for_day(df_full, day_start, day_end)
                if df_fc_sarima is not None and not df_fc_sarima.empty:
                    v = float(df_fc_sarima["Pred_SARIMA"].sum())
                    records_sarima.append({"Date": day_start, "Daily_SARIMA": v})
            except Exception as ex:  # pragma: no cover - logging helper
                print(f"[SARIMA] Lỗi {day_start.date()}: {ex}")

    df_out = df_daily_actual.copy()
    if records_arima:
        df_out = df_out.merge(pd.DataFrame(records_arima), on="Date", how="left")
    if records_sarima:
        df_out = df_out.merge(pd.DataFrame(records_sarima), on="Date", how="left")

    return df_out


def main():
    parser = argparse.ArgumentParser(description="Train ARIMA/SARIMA và lưu dự báo daily")
    parser.add_argument("--city", required=True, help="Tên city, ví dụ: hcmc")
    parser.add_argument("--zone", default=None, help="Zone (hoặc để trống cho all)")
    parser.add_argument("--route", required=True, help="route_id cần train")
    parser.add_argument("--days-back", type=int, default=90, help="Số ngày gần nhất để gom daily")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs") / "arima_sarima_daily.parquet",
        help="Đường dẫn file parquet đích",
    )

    args = parser.parse_args()
    df_pred = compute_daily_predictions(args.city, args.zone, args.route, args.days_back)
    if df_pred is None or df_pred.empty:
        print("[INFO] Không có dữ liệu để lưu.")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_pred.to_parquet(args.output, index=False)
    print(f"[DONE] Đã lưu {len(df_pred)} dòng vào {args.output}")


if __name__ == "__main__":
    main()
