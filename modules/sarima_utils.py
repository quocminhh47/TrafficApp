# modules/sarima_utils.py
from __future__ import annotations
from typing import Optional, Tuple
import warnings

import numpy as np
import pandas as pd

try:
    import pmdarima as pm  # pip install pmdarima
except ImportError:
    pm = None


def forecast_sarima_for_day(
    df_full: pd.DataFrame,
    day_start: pd.Timestamp,
    day_end: pd.Timestamp,
    value_col: str = "Vehicles",
    seasonal: bool = True,
    m: int = 24,
) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Fit SARIMA (auto_arima) trên toàn bộ history < day_start
    và dự báo hourly cho [day_start, day_end).

    Trả về:
      - df_fc: DataFrame ["DateTime", "Pred_SARIMA"] hoặc None
      - info: string mô tả model / lý do lỗi
    """
    if pm is None:
        return None, "pmdarima chưa được cài – không thể chạy SARIMA"

    if df_full is None or df_full.empty:
        return None, "df_full empty"

    df = df_full.copy()
    if "DateTime" not in df.columns or value_col not in df.columns:
        return None, "df_full thiếu DateTime/value_col"

    df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
    df = df.dropna(subset=["DateTime"])
    if df.empty:
        return None, "Parse DateTime thất bại"

    # Resample 1H
    s = (
        df.set_index("DateTime")[value_col]
        .sort_index()
        .resample("1H")
        .mean()
        .interpolate(limit_direction="both")
    )

    # Train = toàn bộ history trước day_start
    train = s[s.index < day_start]
    if len(train) < 24:
        return None, f"Không đủ history cho SARIMA (len={len(train)})"

    horizon = int((day_end - day_start) / pd.Timedelta(hours=1))
    if horizon <= 0:
        return None, "Horizon <= 0"

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            model = pm.auto_arima(
                train,
                seasonal=seasonal,
                m=m,  # seasonal period, 24h
                stepwise=True,
                suppress_warnings=True,
                error_action="ignore",
            )
            y_fc = model.predict(n_periods=horizon)
    except Exception as ex:
        return None, f"SARIMA fit/predict error: {ex}"

    idx_fc = pd.date_range(start=day_start, periods=horizon, freq="1H")
    df_fc = pd.DataFrame(
        {
            "DateTime": idx_fc,
            "Pred_SARIMA": np.asarray(y_fc, dtype=float),
        }
    )
    return df_fc, "SARIMA(auto)"
