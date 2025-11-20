import pandas as pd
from pathlib import Path
import tensorflow as tf
import pickle
import json

from modules.data_loader import load_actual_data
from modules.model_utils import forecast_gru


def main():
    RAW_CITY = "Minneapolis"
    OUT_PATH = Path("outputs/pred_i94.csv")

    # Load raw actual
    df_actual = load_actual_data(RAW_CITY)
    if df_actual.empty:
        print("❌ No raw data available.")
        return

    print(f"[INFO] Raw data loaded: {len(df_actual)} rows")

    # Load model
    MODEL = tf.keras.models.load_model("model/traffic_seq.keras")

    with open("model/vehicles_scaler.pkl", "rb") as f:
        SCALER = pickle.load(f)

    with open("model/seq_meta.json") as f:
        META = json.load(f)

    ROUTES = META["routes"]
    RID2IDX = {r: i for i, r in enumerate(ROUTES)}

    df_pred_list = []

    # Run forecast for each day in raw dataset
    for route in ROUTES:
        days = pd.to_datetime(df_actual["DateTime"]).dt.normalize().unique()

        print(f"[INFO] Forecasting route {route} for {len(days)} days…")

        for d in days:
            hist = df_actual[(df_actual["RouteId"] == route) &
                             (df_actual["DateTime"] < d + pd.Timedelta(days=1))]

            if hist.empty:
                continue

            df_fc, model_used = forecast_gru(
                route, d, MODEL, META, SCALER, ROUTES, RID2IDX, hist
            )

            if not df_fc.empty:
                df_pred_list.append(df_fc)

    if not df_pred_list:
        print("❌ No predictions generated.")
        return

    df_pred = pd.concat(df_pred_list, ignore_index=True)
    df_pred.to_csv(OUT_PATH, index=False)

    print(f"✅ Predictions saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
