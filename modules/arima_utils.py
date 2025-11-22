# modules/arima_utils.py

from __future__ import annotations
from typing import Tuple, Optional
import warnings

import numpy as np
import pandas as pd

try:
    # Cần cài: pip install statsmodels
    from statsmodels.tsa.arima.model import ARIMA
except ImportError:
    ARIMA = None


def forecast_arima_for_day(
    df_full: pd.DataFrame,
    day_start: pd.Timestamp,
    day_end: pd.Timestamp,
    value_col: str = "Vehicles",
    order=(1, 0, 1),
) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Fit ARIMA trên toàn bộ history < day_start và dự báo hourly cho [day_start, day_end).

    Parameters
    ----------
    df_full : DataFrame
        Dữ liệu toàn bộ route (ít nhất gồm cột DateTime + value_col).
    day_start : Timestamp
        Ngày cần forecast (00:00).
    day_end : Timestamp
        Ngày kết thúc (ví dụ day_start + 1 day).
    value_col : str
        Tên cột giá trị (Vehicles).
    order : tuple
        Tham số ARIMA(p,d,q).

    Returns
    -------
    df_fc : DataFrame or None
        Nếu thành công: DataFrame ["DateTime", "Pred_ARIMA"].
        Nếu lỗi: None.
    info : str
        Mô tả model hoặc message lỗi (dùng để log / hiển thị).
    """
    if ARIMA is None:
        return None, "statsmodels chưa được cài – không thể chạy ARIMA"

    if df_full is None or df_full.empty:
        return None, "df_full empty"

    df = df_full.copy()
    if "DateTime" not in df.columns or value_col not in df.columns:
        return None, "df_full thiếu cột DateTime hoặc value_col"

    df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
    df = df.dropna(subset=["DateTime"])
    if df.empty:
        return None, "Parse DateTime thất bại"

    # Resample về chuỗi hourly
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
        return None, f"Không đủ history cho ARIMA (len={len(train)})"

    horizon = int((day_end - day_start) / pd.Timedelta(hours=1))
    if horizon <= 0:
        return None, "Horizon <= 0"

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            model = ARIMA(train, order=order)
            fitted = model.fit()
            fc_res = fitted.get_forecast(steps=horizon)
        y_fc = fc_res.predicted_mean
    except Exception as ex:
        return None, f"ARIMA fit/predict error: {ex}"

    idx_fc = pd.date_range(start=day_start, periods=horizon, freq="1H")
    df_fc = pd.DataFrame(
        {
            "DateTime": idx_fc,
            "Pred_ARIMA": np.asarray(y_fc, dtype=float),
        }
    )
    return df_fc, f"ARIMA{order}"
