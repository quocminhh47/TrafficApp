# modules/evaluation.py
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def merge_and_score(df_pred, df_actual):
    if df_pred.empty or df_actual.empty:
        return pd.DataFrame(), {}

    df_pred = df_pred.copy()
    df_actual = df_actual.copy()

    df_pred["DateTime"] = pd.to_datetime(df_pred["DateTime"])
    df_actual["DateTime"] = pd.to_datetime(df_actual["DateTime"])

    df_pred = df_pred.sort_values("DateTime")
    df_actual = df_actual.sort_values("DateTime")

    merged = pd.merge_asof(
        df_pred,
        df_actual,
        on="DateTime",
        by="RouteId",
        direction="nearest",
        tolerance=pd.Timedelta("30min"),
        suffixes=("_pred", "_act"),
    )

    merged = merged.dropna(subset=["PredictedVehicles", "Vehicles"])
    if merged.empty:
        return merged, {}

    y_true = merged["Vehicles"].values
    y_pred = merged["PredictedVehicles"].values

    scores = {
        "MSE": float(mean_squared_error(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }
    return merged, scores
