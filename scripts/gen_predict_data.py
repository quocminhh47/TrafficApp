# scripts/gen_predict_data.py
from pathlib import Path
import glob
import numpy as np
import pandas as pd

from modules.data_loader import list_zones
from modules.model_utils import forecast_rnn, forecast_gru
from modules.model_manager import load_model_context

DATA_ROOT = Path("data/processed_ds")

STEP_DAYS = 7
MODEL_TYPE = "GRU"

def files_for(city: str, zone: str | None, file_name: str):
    """
    Trả về list path parquet cho 1 city + optional zone.
    """
    file = f"{file_name}_original.parquet"
    if zone in (None, "(All)"):
        pat = DATA_ROOT / city / "**" / file
    else:
        pat = DATA_ROOT / city / zone / "**" / file
    return sorted(glob.glob(str(pat), recursive=True))

def build_forecast_cache(city, zone, route_id, file_name):
    file = f"{file_name}.parquet"

    REAL_PATH = files_for(city, zone, file_name)
    EXT_PATH = DATA_ROOT / city / zone / file

    print(REAL_PATH)
    print(EXT_PATH)

    # load model, scaler, meta
    zone_for_model = None if zone == "(All)" else zone
    try:
        ctx = load_model_context(city, zone_for_model)
    except FileNotFoundError as e:
        if zone == "(All)":
            zones_all = list_zones(city)
            ctx = None
            for z in zones_all:
                if z == "(All)":
                    continue
                try:
                    ctx = load_model_context(city, z)
                    zone_for_model = z
                    break
                except FileNotFoundError:
                    continue

            if ctx is None:
                print.error(str(e))
                return
            else:
                print.info(
                    f"Không có model tổng cho city={city}, zone='(All)'. "
                    f"Đang dùng model của zone='{zone_for_model}'."
                )
        else:
            print.error(str(e))
            return

    # 1. Load data thật
    df_real = pd.read_parquet(REAL_PATH)
    df_real = df_real[
        (df_real["RouteId"] == route_id)
    ]

    if df_real.empty:
        print("❌ Không có data thật")
        return

    # 2. Load forecast cache nếu có
    if EXT_PATH.exists():
        df_ext = pd.read_parquet(EXT_PATH)
        df_ext = df_ext[
            (df_ext["City"] == city) &
            (df_ext["RouteId"] == route_id)
        ]


        hist = pd.concat([df_real, df_ext], ignore_index=True)
    else:
        hist = df_real.copy()

    today = pd.Timestamp.today().normalize()
    #today = pd.Timestamp("2021-01-15 18:30").normalize()
    all_new_fc = []

    # 3. Forecast nối tiếp
    last_ext_day = hist["DateTime"].max().normalize()
    while last_ext_day < today:
        print("▶ Anchor:", last_ext_day)

        remaining_days = (today - last_ext_day).days
        n_days = STEP_DAYS
        if remaining_days <= STEP_DAYS:
            n_days = remaining_days - 1

        df_fc, anchor = forecast_week_from_history(
            hist=hist,
            route_id=route_id,
            ctx=ctx,
            n_days=n_days,
            model_type=MODEL_TYPE,
        )

        if df_fc.empty:
            break

        # gắn metadata
        df_fc["is_forecast"] = True

        # nối vào hist
        tmp = df_fc.rename(columns={"PredictedVehicles": "Vehicles"})
        hist = pd.concat(
            [hist, tmp[["DateTime", "Vehicles", "RouteId"]]],
            ignore_index=True,
        )
        all_new_fc.append(tmp)
        last_ext_day = hist["DateTime"].max().normalize()

    # 4. Save
    if all_new_fc:
        df_base = df_real.copy()
        df_add = pd.concat(all_new_fc, ignore_index=True)

        df_new = (
            pd.concat([df_base, df_add], ignore_index=True)
            .drop_duplicates(subset=["DateTime", "RouteId"], keep="last")
            .sort_values("DateTime")
        )

        mask = df_new["is_forecast"] == True
        noise = np.random.uniform(-0.03, 0.03, mask.sum())
        df_new.loc[mask, "Vehicles"] = (
                df_new.loc[mask, "Vehicles"] * (1 + noise)
        ).clip(lower=0).round().astype(int)

        if EXT_PATH.exists():
            df_old = pd.read_parquet(EXT_PATH)
            df_all = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_all = df_new

        df_all = df_all.drop_duplicates(
            subset=["DateTime", "RouteId"], keep="last"
        )

        df_all.to_parquet(EXT_PATH, index=False)
        print(f"✅ Saved forecast → {EXT_PATH}")

        #df = pd.read_parquet(EXT_PATH)

        #df.to_csv(
        #    "data/traffic_extended.csv",
        #    index=False,
        #    encoding="utf-8-sig",  # mở bằng Excel không lỗi font
        #)

        #print("✅ Đã export traffic_extended.csv")
    else:
        print("Không có forecast mới.")

def forecast_week_from_history(
    hist: pd.DataFrame,
    route_id,
    ctx,
    n_days=7,
    model_type="GRU",
):

    if hist.empty:
        return pd.DataFrame(), None

    hist = hist.copy()
    hist["DateTime"] = pd.to_datetime(hist["DateTime"], errors="coerce")
    hist = hist.dropna(subset=["DateTime", "Vehicles"])
    hist = hist.sort_values("DateTime")

    last_dt = hist["DateTime"].max()
    anchor_day_raw = last_dt.normalize()

    model_type_norm = (model_type or "GRU").upper()
    use_rnn = (
        model_type_norm == "RNN"
        and getattr(ctx, "rnn_model", None) is not None
    )

    if model_type_norm == "RNN" and not use_rnn:
        print("[forecast_week_from_history] RNN selected but ctx.rnn_model is None → fallback GRU")

    all_fc = []

    for k in range(1, n_days + 1):
        base_date = anchor_day_raw + pd.Timedelta(days=k)

        hist_start = base_date - pd.Timedelta(hours=ctx.lookback)
        df_hist = hist[
            (hist["DateTime"] >= hist_start)
            & (hist["DateTime"] < base_date)
        ]

        if len(df_hist) < ctx.lookback:
            print(
                f"[forecast_week_from_history] thiếu history cho {base_date.date()}, dừng."
            )
            break

        if use_rnn:
            df_fc_day, model_used = forecast_rnn(
                route_id=route_id,
                base_date=base_date,
                model=ctx.rnn_model,
                meta=ctx.meta,
                scaler=ctx.scaler,
                routes_model=ctx.routes_model,
                rid2idx=ctx.rid2idx,
                df_hist=df_hist,
            )
        else:
            df_fc_day, model_used = forecast_gru(
                route_id=route_id,
                base_date=base_date,
                model=ctx.gru_model,
                meta=ctx.meta,
                scaler=ctx.scaler,
                routes_model=ctx.routes_model,
                rid2idx=ctx.rid2idx,
                df_hist=df_hist,
            )

        if df_fc_day is None or df_fc_day.empty:
            break

        all_fc.append(df_fc_day)

        # 🔁 append forecast vào history để dùng cho ngày sau
        tmp = df_fc_day.rename(columns={"PredictedVehicles": "Vehicles"})
        hist = pd.concat(
            [hist, tmp[["DateTime", "Vehicles", "RouteId"]]],
            ignore_index=True,
        )

    if not all_fc:
        return pd.DataFrame(), anchor_day_raw

    df_fc_raw = pd.concat(all_fc, ignore_index=True)
    return df_fc_raw, anchor_day_raw

def main():

    build_forecast_cache(
        city="Minneapolis",
        zone="I94",
        route_id="I-94-WB",
        file_name="i94_main"
    )

    build_forecast_cache(
        city="Seattle",
        zone="FremontBridge",
        route_id="Fremont-East",
        file_name="Fremont_East"
    )

    build_forecast_cache(
        city="Seattle",
        zone="FremontBridge",
        route_id="Fremont-Total",
        file_name="Fremont_Total"
    )

    build_forecast_cache(
        city="Seattle",
        zone="FremontBridge",
        route_id="Fremont-West",
        file_name="Fremont_West"
    )


if __name__ == "__main__":
    main()
