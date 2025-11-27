# modules/sarima_utils.py
from __future__ import annotations
from typing import Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


def forecast_sarima_for_day(
    df_full: pd.DataFrame,
    day_start: pd.Timestamp,
    day_end: pd.Timestamp,
    value_col: str = "Vehicles",
    seasonal: bool = True,
    m: int = 24,
) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Fit SARIMA (SARIMAX với bộ tham số cố định) trên toàn bộ history < day_start
    và dự báo hourly cho [day_start, day_end).

    Trả về:
      - df_fc: DataFrame ["DateTime", "Pred_SARIMA"] hoặc None
      - info: string mô tả model / lý do lỗi
    """
    if df_full is None or df_full.empty:
        return None, "df_full empty"

    df = df_full.copy()
    if "DateTime" not in df.columns or value_col not in df.columns:
        return None, "df_full thiếu DateTime/value_col"

    df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
    df = df.dropna(subset=["DateTime"])
    if df.empty:
        return None, "Parse DateTime thất bại"

    # Resample 1h (fix FutureWarning: dùng '1h' thay vì '1H')
    s = (
        df.set_index("DateTime")[value_col]
        .sort_index()
        .resample("1h")
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

    # Dùng SARIMAX với bộ tham số cố định (nhẹ hơn auto_arima rất nhiều)
    # Bạn có thể chỉnh order/seasonal_order nếu muốn.
    order = (1, 0, 1)
    if seasonal:
        seasonal_order = (1, 1, 1, m)  # m = 24 (daily seasonality) hoặc 168 nếu weekly
    else:
        seasonal_order = (0, 0, 0, 0)

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            model = SARIMAX(
                train,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            res = model.fit(disp=False)
            y_fc = res.forecast(steps=horizon)
    except Exception as ex:
        return None, f"SARIMA fit/predict error: {ex}"

    # Tạo index forecast với freq '1h' (chữ thường)
    idx_fc = pd.date_range(start=day_start, periods=horizon, freq="1h")
    df_fc = pd.DataFrame(
        {
            "DateTime": idx_fc,
            "Pred_SARIMA": np.asarray(y_fc, dtype=float),
        }
    )
    info = f"SARIMA(order={order}, seasonal_order={seasonal_order})"
    return df_fc, info
