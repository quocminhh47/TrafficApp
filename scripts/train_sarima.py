#!/usr/bin/env python3
# Đánh giá mô hình SARIMA dự báo 24h cho từng route
# So sánh benchmark với LSTM

import warnings
warnings.filterwarnings("ignore")

import glob
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

import pmdarima as pm

DATA_ROOT = Path("data/processed_ds")

LOOKBACK = 168
HORIZON = 24


def load_all_parquet():
    pat = DATA_ROOT / "**" / "**" / "**" / "*.parquet"
    files = sorted(glob.glob(str(pat), recursive=True))

    frames = []
    for f in files:
        try:
            df = pd.read_parquet(f)
        except Exception:
            continue

        if not {"DateTime", "RouteId", "Vehicles"} <= set(df.columns):
            continue

        df["DateTime"] = pd.to_datetime(df["DateTime"], utc=True)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# -----------------------------------------------------
# Train SARIMA tự động (Auto-ARIMA)
# -----------------------------------------------------
def train_sarima(series: pd.Series):
    """
    series: pandas Series, index = datetime, 1h frequency
    """
    model = pm.auto_arima(
        series,
        seasonal=True,
        m=24,                   # season daily
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore"
    )
    return model


# -----------------------------------------------------
# Forecast 24h
# -----------------------------------------------------
def forecast_sarima(model, horizon=24):
    fc = model.predict(n_periods=horizon)
    return np.array(fc)


# -----------------------------------------------------
# Evaluate SARIMA trên từng route
# -----------------------------------------------------
def evaluate_sarima(df: pd.DataFrame):
    results = []
    routes = sorted(df["RouteId"].unique())

    for rid in routes:
        print(f"\n[ROUTE] {rid}")

        g = df[df["RouteId"] == rid].copy()
        g = (
            g.set_index("DateTime")
             .resample("1h")["Vehicles"]
             .mean()
             .dropna()
        )

        if len(g) < LOOKBACK + HORIZON:
            print(f" - Dữ liệu không đủ ({len(g)}h). Bỏ qua.")
            continue

        # Chọn đoạn cuối 168h làm train, 24h làm test
        train = g.iloc[-(LOOKBACK + HORIZON):-HORIZON]
        test = g.iloc[-HORIZON:]

        # Huấn luyện SARIMA
        print(" - Train SARIMA...")
        model = train_sarima(train)

        # Dự báo 24h
        pred = forecast_sarima(model, HORIZON)

        # Tránh trường hợp output bị NaN
        if np.isnan(pred).any():
            print(" - NaN trong forecast → skip")
            continue

        # Đánh giá
        rmse = np.sqrt(mean_squared_error(test.values, pred))
        mae = mean_absolute_error(test.values, pred)

        results.append({
            "RouteId": rid,
            "RMSE": rmse,
            "MAE": mae
        })

    return pd.DataFrame(results)

# -----------------------------------------------------
# Main
# -----------------------------------------------------
def main():
    print("[INFO] Loading parquet files...")
    df = load_all_parquet()

    df_eval = evaluate_sarima(df)

    out_path = Path("model/sarima_eval.csv")
    df_eval.to_csv(out_path, index=False)

    print(f"[DONE] SARIMA evaluation saved to {out_path}")


if __name__ == "__main__":
    main()