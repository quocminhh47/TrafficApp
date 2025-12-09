#!/usr/bin/env python
import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
from pathlib import Path
import joblib
import json
import os

from functools import lru_cache
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from modules.data_loader import load_slice, list_cities, list_zones, list_routes
from modules.geo_routes import load_routes_geo
from map_component import map_routes  # custom map component

from modules.model_utils import (
    forecast_gru,
    forecast_rnn,
    forecast_lstm,
    forecast_week_after_last_point,
    shift_forecast_to_today,
)
from modules.model_manager import load_model_context

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# =========================
# HCMC: cấu hình cho travel-time
# =========================

# Chiều dài xấp xỉ của từng tuyến (km) – bạn có thể chỉnh lại cho sát thực tế hơn
HCMC_ROUTE_LENGTH_KM = {
    "ly_thuong_kiet": 4.3,
    "nguyen_kiem": 3.8,
    "quang_trung": 5.6,
    "nguyen_dinh_chieu": 3.2,
    "le_duc_tho": 7.2,
    "quoc_lo_1a": 51.0,
    "to_hien_thanh": 2.1,
    "truong_chinh": 8.5
}

HCMC_DEFAULT_LENGTH_KM = 4.0          # nếu route chưa có trong dict trên
HCMC_FREE_FLOW_SPEED_KMH = 40.0       # tốc độ "thoáng" mặc định trong nội đô

# =====================================================
# HÀM TÍNH CHỈ SỐ ĐÁNH GIÁ CHUNG CHO UI
# =====================================================

def compute_common_metrics(
    y_true,
    y_pred,
    *,
    task: str = "regression",
    acc_tolerance: float = 0.2,
    threshold: float = 0.5,
) -> dict:
    """
    MSE / RMSE / MAE / SMAPE / Accuracy – dùng cho UI.

    - task="regression": I-94, Fremont, v.v.
        Accuracy = % điểm có sai số tương đối <= acc_tolerance.
    - task="binary_prob": HCMC congestion.
        Accuracy = accuracy nhị phân sau khi threshold.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not np.any(mask):
        return {
            "MSE": np.nan,
            "RMSE": np.nan,
            "MAE": np.nan,
            "SMAPE": np.nan,
            "Accuracy": np.nan,
        }

    y_true = y_true[mask]
    y_pred = y_pred[mask]

    diff = y_pred - y_true
    mse = float(np.mean(diff**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))

    denom = np.abs(y_true) + np.abs(y_pred)
    smape = float(
        np.mean(
            2.0 * np.abs(diff) / (denom + 1e-8)
        )
        * 100.0
    )

    if task == "regression":
        rel_err = np.abs(diff) / (np.abs(y_true) + 1e-8)
        acc = float(np.mean(rel_err <= acc_tolerance) * 100.0)
    elif task == "binary_prob":
        y_bin = (y_pred >= threshold).astype(float)
        acc = float(np.mean(y_bin == y_true) * 100.0)
    else:
        acc = np.nan

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "SMAPE": smape,
        "Accuracy": acc,
    }


def get_hcmc_route_length_km(route_id: str) -> float:
    """Trả về chiều dài tuyến (km), nếu không có thì dùng default."""
    return HCMC_ROUTE_LENGTH_KM.get(route_id, HCMC_DEFAULT_LENGTH_KM)


# ARIMA / SARIMA (optional)
try:
    from modules.arima_utils import forecast_arima_for_day
    HAS_ARIMA = True
except Exception:
    forecast_arima_for_day = None
    HAS_ARIMA = False

try:
    from modules.arima_utils import forecast_sarima_for_day
    HAS_SARIMA = True
except Exception:
    forecast_sarima_for_day = None
    HAS_SARIMA = False


@st.cache_resource
def get_model_context(city: str, zone: str | None):
    """
    Cache ModelContext cho mỗi (city, zone) để tránh load model nhiều lần.
    """
    return load_model_context(city, zone)


@lru_cache(maxsize=None)
def load_lstm_artifacts_for_family(family_name: str):
    """
    Load LSTM artifacts trong:
        model/<family_name>/

    Trả về dict:
      {
        "model", "meta", "scaler",
        "routes", "rid2idx", "dir"
      }
    hoặc None nếu thiếu file.
    """
    base = Path("model")
    model_dir = base / family_name

    meta_path = model_dir / "lstm_meta.json"
    model_path = model_dir / "traffic_lstm.keras"
    scaler_path = model_dir / "vehicles_scaler.pkl"

    if not (meta_path.exists() and model_path.exists() and scaler_path.exists()):
        print(
            f"[LSTM] Missing artifacts in {model_dir}: "
            f"{meta_path.exists()=}, {model_path.exists()=}, {scaler_path.exists()=}"
        )
        return None

    print(f"[LSTM] Using LSTM model dir: {model_dir}")

    with open(meta_path, "r") as f:
        meta = json.load(f)
    from tensorflow.keras.models import load_model
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
        "dir": str(model_dir),
    }


# ======================================================
# HELPER: Forecast 24h cho 1 ngày cụ thể (GRU / RNN / LSTM)
# ======================================================
def forecast_one_day(
    route_id,
    forecast_date: pd.Timestamp,
    city,
    zone,
    ctx,
    seq_model_type: str = "GRU",
):
    """
    Forecast 24h cho 1 ngày cụ thể (00:00 -> 23:00) bằng GRU / RNN / LSTM,
    dựa trên window history LOOKBACK giờ ngay trước forecast_date.
    """
    LOOKBACK = int(ctx.lookback)

    forecast_date = pd.Timestamp(forecast_date).normalize()
    base_date = forecast_date

    # History window cho seq model: [base_date - LOOKBACK, base_date)
    start_dt = base_date - pd.Timedelta(hours=LOOKBACK)
    end_dt = base_date

    df_hist = load_slice(
        city=city,
        zone=None if zone == "(All)" else zone,
        routes=[route_id],
        start_dt=start_dt,
        end_dt=end_dt,
    )

    if df_hist is None or df_hist.empty:
        return pd.DataFrame(), seq_model_type

    # Khởi tạo mặc định để tránh UnboundLocalError
    df_fc = None
    model_used = seq_model_type

    # ---- RNN ----
    if seq_model_type == "RNN" and getattr(ctx, "rnn_model", None) is not None:
        df_fc, model_used = forecast_rnn(
            route_id=route_id,
            base_date=base_date,
            model=ctx.rnn_model,
            meta=ctx.meta,
            scaler=ctx.scaler,
            routes_model=ctx.routes_model,
            rid2idx=ctx.rid2idx,
            df_hist=df_hist,
        )

    elif seq_model_type == "LSTM":
        # LSTM dùng artifacts theo family_name của ctx (I94, Seattle_FremontBridge, ...)
        from modules.model_utils import forecast_lstm  # nếu bạn để trong module riêng
        lstm_ctx = load_lstm_artifacts_for_family(ctx.family_name)

        if lstm_ctx is not None:
            df_fc, model_used = forecast_lstm(
                route_id=route_id,
                base_date=base_date,
                model=lstm_ctx["model"],
                meta=lstm_ctx["meta"],
                scaler=lstm_ctx["scaler"],
                routes_model=lstm_ctx["routes"],
                rid2idx=lstm_ctx["rid2idx"],
                df_hist=df_hist,
            )
        else:
            # Không có LSTM → trả về rỗng, phía trên sẽ bỏ qua
            df_fc, model_used = pd.DataFrame(), "LSTM_missing"

    else:
        # GRU default
        df_fc, model_used = forecast_gru(
            route_id=route_id,
            base_date=base_date,
            model=ctx.gru_model,
            meta=ctx.meta,
            scaler=ctx.scaler,
            routes_model=ctx.routes_model,
            rid2idx=ctx.rid2idx,
            df_hist=df_hist,
        )

    if df_fc is None or df_fc.empty:
        return pd.DataFrame(), model_used

    df_fc = df_fc.copy()
    df_fc["DateTime"] = pd.to_datetime(df_fc["DateTime"], errors="coerce")
    next_day = forecast_date + pd.Timedelta(days=1)

    df_fc = df_fc[
        (df_fc["DateTime"] >= forecast_date) & (df_fc["DateTime"] < next_day)
    ].sort_values("DateTime")

    return df_fc, model_used

def forecast_week_after_last_point_lstm(
    route_id: str,
    city: str,
    zone: str,
    ctx,
    n_days: int = 7,
):
    """
    Forecast n_days (mặc định 7) sau NGÀY CUỐI CÙNG trong dữ liệu thật
    bằng LSTM, kiểu NO SHIFT (giống forecast_week_after_last_point).
    Trả về:
        - df_fc_raw: DataFrame forecast trên timeline thật
        - anchor_day_raw: ngày cuối trong dữ liệu (normalize 00:00)
    """
    # 1) Load LSTM artifacts theo family_name (I94, Seattle_FremontBridge, ...)
    lstm_art = load_lstm_artifacts_for_family(ctx.family_name)
    if lstm_art is None:
        print(f"[LSTM-week] Không tìm thấy artifacts cho family={ctx.family_name}")
        return pd.DataFrame(), None

    model_lstm = lstm_art["model"]
    meta_lstm = lstm_art["meta"]
    scaler_lstm = lstm_art["scaler"]
    routes_lstm = lstm_art["routes"]
    rid2idx_lstm = lstm_art["rid2idx"]

    # 2) Load toàn bộ series của route
    df_full = load_slice(
        city=city,
        zone=zone,
        routes=[route_id],
        start_dt=None,
        end_dt=None,
    )
    if df_full is None or df_full.empty:
        print(f"[LSTM-week] Không có dữ liệu full cho route={route_id}")
        return pd.DataFrame(), None

    df_full["DateTime"] = pd.to_datetime(df_full["DateTime"], errors="coerce")
    df_full = df_full.dropna(subset=["DateTime", "Vehicles"])
    df_full = df_full.sort_values("DateTime")

    last_dt = df_full["DateTime"].max()
    anchor_day_raw = last_dt.normalize()  # ví dụ 2018-10-31 00:00

    # 3) History tổng hợp (ban đầu = dữ liệu thật)
    hist = df_full.copy()

    all_fc = []
    LOOKBACK = ctx.lookback

    for k in range(1, n_days + 1):
        # base_date = đầu ngày thứ k sau anchor_day_raw
        base_date = anchor_day_raw + pd.Timedelta(days=k)

        # history LOOKBACK giờ trước base_date
        hist_start = base_date - pd.Timedelta(hours=LOOKBACK)
        df_hist = hist[
            (hist["DateTime"] >= hist_start) & (hist["DateTime"] < base_date)
        ].copy()

        if len(df_hist) < LOOKBACK:
            print(
                f"[LSTM-week] Route {route_id}: thiếu history ({len(df_hist)}h) cho ngày {base_date}, dừng."
            )
            break

        # Forecast 1 ngày bằng LSTM
        df_fc_day, model_used = forecast_lstm(
            route_id=route_id,
            base_date=base_date,
            model=model_lstm,
            meta=meta_lstm,
            scaler=scaler_lstm,
            routes_model=routes_lstm,
            rid2idx=rid2idx_lstm,
            df_hist=df_hist,
        )

        if df_fc_day is None or df_fc_day.empty:
            print(f"[LSTM-week] Forecast rỗng cho ngày {base_date}, dừng.")
            break

        all_fc.append(df_fc_day)

        # append prediction vào hist để ngày sau dùng luôn cả data forecast
        tmp = df_fc_day.rename(columns={"PredictedVehicles": "Vehicles"})
        hist = pd.concat(
            [hist, tmp[["DateTime", "Vehicles", "RouteId"]]],
            ignore_index=True,
        )

    if not all_fc:
        return pd.DataFrame(), anchor_day_raw

    df_fc_raw = pd.concat(all_fc, ignore_index=True)
    return df_fc_raw, anchor_day_raw



def vn_weekday_label(dt: pd.Timestamp) -> str:
    """
    Trả về label tiếng Việt cho 1 ngày, ví dụ: 'Thứ 2 21/11'
    """
    dt = pd.Timestamp(dt)
    wd = dt.weekday()  # 0=Mon ... 6=Sun
    if wd == 6:
        thu = "Chủ nhật"
    else:
        thu = f"Thứ {wd + 2}"
    return f"{thu} {dt.strftime('%d/%m')}"


def load_top2_summary(family_name: str, route_id: str):
    """
    Đọc file <route_id>_top2_last_quarter.json nếu có.
    Trả về dict hoặc None.
    """
    model_dir = Path("model") / family_name
    summary_path = model_dir / f"{route_id}_top2_last_quarter.json"
    if not summary_path.exists():
        return None
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as ex:
        print(f"[load_top2_summary] Error reading {summary_path}: {ex}")
        return None

# ==== HCMC CONGESTION – GRU dự báo Mức độ kẹt xe 2h tới ====

HCMC_CSV_PATH = Path("data/raw/hcmc/train.csv")
HCMC_LOOKBACK = 16          # phải khớp với LOOKBACK khi train GRU HCMC
HCMC_STEP_MINUTES = 30      # mỗi period = 30'
HCMC_FC_STEPS = 4           # 4 bước = 2 giờ tới


def render_hcmc_eval_summary_for_route(route_id: str):
    """
    Đọc hcmc_eval_summary.csv và hiển thị MSE / RMSE / MAE / SMAPE / Accuracy
    cho tuyến HCMC đang chọn.
    """
    eval_path = os.path.join(BASE_DIR, "data", "hcmc_eval", "hcmc_eval_summary.csv")
    if not os.path.exists(eval_path):
        st.info("Chưa tìm thấy file đánh giá HCMC (hcmc_eval_summary.csv).")
        return

    df = pd.read_csv(eval_path)

    if "slug" not in df.columns:
        st.warning("File summary không có cột 'slug'.")
        return

    row = df[df["slug"] == route_id]
    if row.empty:
        st.info("Chưa có metric đánh giá cho tuyến này.")
        return

    r = row.iloc[0]

    st.markdown("### 📊 Đánh giá độ tin cậy mô hình (HCMC)")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("MSE", f"{r['MSE']:.4f}")
    with col2:
        st.metric("RMSE", f"{r['RMSE']:.4f}")
    with col3:
        st.metric("MAE", f"{r['MAE']:.4f}")

    # col4, col5 = st.columns(2)
    with col4:
        st.metric("SMAPE", f"{r['SMAPE']:.2f} %")
    with col5:
        st.metric("Accuracy", f"{r['Accuracy']:.1f} %")


@lru_cache(maxsize=None)
def _load_hcmc_raw_df():
    """Đọc raw HCMC + tính cột DateTime từ date + period_x_y."""
    if not HCMC_CSV_PATH.exists():
        print(f"[HCMC] Không tìm thấy file {HCMC_CSV_PATH}")
        return None

    df = pd.read_csv(HCMC_CSV_PATH)

    # Cần tối thiểu các cột này
    if not {"date", "period", "street_name", "LOS"} <= set(df.columns):
        print("[HCMC] Thiếu cột bắt buộc trong train.csv")
        return None

    df["date"] = pd.to_datetime(df["date"])
    period_num = df["period"].str.extract(r"period_(\d+)_(\d+)", expand=True).astype(int)
    df["hour"] = period_num[0]
    df["minute"] = period_num[1]
    df["DateTime"] = (
        df["date"]
        + pd.to_timedelta(df["hour"], unit="h")
        + pd.to_timedelta(df["minute"], unit="m")
    )
    return df


def _load_hcmc_series_for_route(route_id: str, routes_geo_all: pd.DataFrame):
    """
    Từ route_id (slug trong routes_geo) → tìm street_name gốc trong train.csv,
    rồi build series nhị phân: 1 = tắc, 0 = không tắc. Index = DateTime.
    """
    df_geo = routes_geo_all[
        (routes_geo_all["city"] == "HoChiMinh")
        & (routes_geo_all["route_id"] == route_id)
    ]
    if df_geo.empty:
        print(f"[HCMC] Không tìm thấy routes_geo cho route_id={route_id}")
        return None

    full_name = df_geo.iloc[0]["name"]             # VD: "Lý Thường Kiệt (HCMC)"
    street_name = str(full_name).replace(" (HCMC)", "")  # "Lý Thường Kiệt"

    df = _load_hcmc_raw_df()
    if df is None:
        return None

    df_st = df[df["street_name"] == street_name].copy()
    if df_st.empty:
        print(f"[HCMC] Không có dữ liệu cho street_name='{street_name}'")
        return None

    def is_congested(group: pd.Series) -> int:
        ratio_congested = (group.isin({"D", "E", "F"})).mean()
        return int(ratio_congested >= 0.5)

    s = (
        df_st.groupby("DateTime")["LOS"]
        .apply(is_congested)
        .sort_index()
        .astype(float)
    )
    print(f"[HCMC] '{street_name}': {len(s)} mốc thời gian (sau group)")
    return s, full_name, street_name

def estimate_travel_time_from_prob(
    p_cong: float,
    length_km: float,
    v_free_kmh: float = HCMC_FREE_FLOW_SPEED_KMH,
) -> tuple[float, float, str]:
    """
    Từ xác suất tắc đường p_cong (0–1), ước lượng:
    - thời gian di chuyển để đi hết tuyến (phút)
    - độ trễ so với điều kiện thoáng (phút)
    - nhãn mức độ giảm tốc (low / medium / high)
    """
    p = float(max(0.0, min(1.0, p_cong)))

    # Thời gian đi nếu đường thoáng
    T_free = 60.0 * length_km / max(v_free_kmh, 1e-6)

    # Map p -> hệ số giảm tốc (speed factor)
    # p thấp => gần free-flow; p cao => chạy chậm
    if p <= 0.3:
        factor = 0.9   # gần như thoáng
        level = "low"
    elif p <= 0.7:
        factor = 0.6   # hơi đông
        level = "medium"
    else:
        factor = 0.3   # rất đông
        level = "high"

    v_eff = max(v_free_kmh * factor, 5.0)  # tránh chia cho tốc độ quá nhỏ
    T_travel = 60.0 * length_km / v_eff
    delay = T_travel - T_free
    return T_travel, delay, level


def make_travel_time_table_for_slots(df_slots: "pd.DataFrame", route_id: str) -> "pd.DataFrame":
    """
    Nhận vào DataFrame các slot dự báo 2 giờ tới và route_id,
    trả về DataFrame mới với cột thời gian di chuyển & độ trễ.

    ⚠ Giả sử df_slots có:
        - cột 'SlotLabel' (hoặc 'TimeLabel'): label khung giờ (vd '16:30', '17:00')
        - cột 'P_cong' (0–1): xác suất tắc đường trong khung đó

    Nếu code hiện tại của bạn dùng tên khác, chỉ cần đổi lại cho đúng bên dưới.
    """
    import pandas as pd

    length_km = get_hcmc_route_length_km(route_id)
    v_free = HCMC_FREE_FLOW_SPEED_KMH
    T_free = 60.0 * length_km / max(v_free, 1e-6)

    rows = []
    for _, r in df_slots.iterrows():
        # 👉 ĐỔI tên cột ở đây nếu cần:
        p_cong = float(r["P_cong"])  # ví dụ nếu cột là 'P_tac' thì sửa thành r["P_tac"]
        slot_label = str(r["SlotLabel"])  # hoặc 'TimeLabel', tùy DataFrame hiện tại

        T_travel, delay, level = estimate_travel_time_from_prob(p_cong, length_km, v_free)

        rows.append(
            {
                "Khung giờ": slot_label,
                "P tắc (%)": round(p_cong * 100.0, 1),
                "Thời gian di chuyển (phút)": round(T_travel, 1),
                "Độ trễ so với đường thoáng (phút)": round(delay, 1),
                "Mức độ kẹt (low/medium/high)": level,
            }
        )

    df_out = pd.DataFrame(rows)
    # Sắp xếp theo thời gian nếu cần (giả sử SlotLabel ở dạng 'HH:MM')
    try:
        df_out = df_out.sort_values("Khung giờ")
    except Exception:
        pass

    # Thêm T_free vào thuộc tính để hiển thị metric nhanh (dùng getattr bên ngoài)
    df_out._T_free = T_free
    df_out._length_km = length_km
    return df_out

@st.cache_resource
def _load_hcmc_gru_model_for_route(route_id: str):
    """
    Load model GRU congestion cho 1 tuyến HCMC.
    Giả định file: model/hcmc/gru_congestion_<route_id>.keras
    """
    from tensorflow.keras.models import load_model
    model_path = Path("model") / "hcmc" / f"gru_congestion_{route_id}.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"[HCMC] Không tìm thấy model: {model_path}")
    print(f"[HCMC] Load model {model_path}")
    model = load_model(model_path)
    return model


@st.cache_resource
def _load_hcmc_lstm_model_for_route(route_id: str):
    """
    Load model LSTM congestion cho 1 tuyến HCMC.
    Giả định file: model/hcmc/lstm_congestion_<route_id>.keras
    """
    from tensorflow.keras.models import load_model

    model_path = Path("model") / "hcmc" / f"lstm_congestion_{route_id}.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"[HCMC] Không tìm thấy model LSTM: {model_path}")

    print(f"[HCMC] Load LSTM model {model_path}")
    model = load_model(model_path)
    return model


def forecast_hcmc_next_2h(route_id: str, routes_geo_all: pd.DataFrame):
    """
    Dùng GRU + LSTM congestion để dự báo Mức độ kẹt xe cho 4 bước tiếp theo (2h tới).
    Trả về (df_fc, full_name) hoặc None.
    """
    out = _load_hcmc_series_for_route(route_id, routes_geo_all)
    if out is None:
        return None
    s, full_name, street_name = out

    if len(s) <= HCMC_LOOKBACK:
        print(
            f"[HCMC] Quá ít time step ({len(s)}) cho route_id={route_id}, "
            f"LOOKBACK={HCMC_LOOKBACK}"
        )
        return None

    times = list(s.index)
    y_vals = list(s.values.astype(float))

    def rollout_with_model(model):
        preds = []
        t_local = list(times)
        y_local = list(y_vals)

        for _ in range(HCMC_FC_STEPS):
            window_times = pd.DatetimeIndex(t_local[-HCMC_LOOKBACK:])
            window_y = np.array(y_local[-HCMC_LOOKBACK:], dtype=float)

            total_minutes = window_times.hour * 60 + window_times.minute
            sin_t = np.sin(2 * np.pi * total_minutes / (24 * 60))
            cos_t = np.cos(2 * np.pi * total_minutes / (24 * 60))

            weekday = window_times.weekday
            sin_w = np.sin(2 * np.pi * weekday / 7.0)
            cos_w = np.cos(2 * np.pi * weekday / 7.0)

            F_window = np.stack([window_y, sin_t, cos_t, sin_w, cos_w], axis=1)
            X = F_window[np.newaxis, :, :]

            p = float(model.predict(X, verbose=0).ravel()[0])

            # cập nhật history bên trong "thế giới data"
            last_time = t_local[-1]
            new_time = last_time + pd.Timedelta(minutes=HCMC_STEP_MINUTES)
            t_local.append(new_time)
            y_local.append(1.0 if p >= 0.5 else 0.0)

            preds.append(p)

        return preds

    preds_dict: dict[str, list[float]] = {}

    for model_name, loader in (
        ("GRU", _load_hcmc_gru_model_for_route),
        ("LSTM", _load_hcmc_lstm_model_for_route),
    ):
        try:
            model = loader(route_id)
            preds_dict[model_name] = rollout_with_model(model)
        except FileNotFoundError as ex:
            print(ex)

    if not preds_dict:
        return None

    seq_len = max(len(v) for v in preds_dict.values())

    prob_columns = {}
    for name in ("GRU", "LSTM"):
        vals = preds_dict.get(name)
        if vals is None:
            prob_columns[name] = [np.nan] * seq_len
        elif len(vals) == seq_len:
            prob_columns[name] = vals
        else:
            # bảo đảm cùng độ dài bằng cách padding NaN phía sau
            pad_len = seq_len - len(vals)
            prob_columns[name] = vals + [np.nan] * pad_len

    preds_stack = np.array(list(prob_columns.values()), dtype=float)
    preds_avg = np.nanmean(preds_stack, axis=0)

    # --- Phần này là MỚI: build trục thời gian theo "bây giờ" ---
    now = pd.Timestamp.now(tz="Asia/Ho_Chi_Minh")

    # làm tròn về slot gần nhất: 00 hoặc 30 phút
    minute_bin = 0 if now.minute < 30 else 30
    current_slot = now.replace(minute=minute_bin, second=0, microsecond=0)

    display_times = [
        current_slot + pd.Timedelta(minutes=HCMC_STEP_MINUTES * (i + 1))
        for i in range(len(preds_avg))
    ]

    df_fc = pd.DataFrame({"DateTime": display_times, "ProbCongested": preds_avg})
    df_fc["Prob_GRU"] = prob_columns["GRU"]
    df_fc["Prob_LSTM"] = prob_columns["LSTM"]

    for name, vals in prob_columns.items():
        df_fc[f"Prob_{name}"] = vals

    return df_fc, full_name


def render_hcmc_congestion_next_2h(route_id: str, routes_geo_all: pd.DataFrame):
    """
    UI cho HCMC: biểu đồ + bảng ngang Mức độ kẹt xe 2h tới cho tuyến đang chọn,
    + ước lượng thời gian di chuyển theo từng khung 30 phút.
    """
    out = forecast_hcmc_next_2h(route_id, routes_geo_all)
    if out is None:
        st.info("Không đủ dữ liệu để dự báo tắc đường cho tuyến HCMC này.")
        return

    df_fc, full_name = out

    st.subheader(f" Dự báo nguy cơ tắc đường trong 2 giờ tới – {full_name}")

    df_fc = df_fc.copy()
    df_fc["DateTime"] = pd.to_datetime(df_fc["DateTime"], errors="coerce")
    df_fc = df_fc.dropna(subset=["DateTime"])
    df_fc["TimeLabel"] = df_fc["DateTime"].dt.strftime("%H:%M")

    def level_from_p(p: float) -> str:
        if p >= 0.7:
            return "high"
        elif p >= 0.4:
            return "medium"
        return "low"

    df_fc["Level"] = df_fc["ProbCongested"].apply(level_from_p)

    # ======== TÓM TẮT NHANH 2 GIỜ TỚI ========
    probs = df_fc["ProbCongested"].clip(0.0, 1.0).values
    expected_congested_minutes = HCMC_STEP_MINUTES * float(np.sum(probs))
    avg_prob = float(np.mean(probs))

    avoid_slots = df_fc[df_fc["ProbCongested"] >= 0.7]["TimeLabel"].tolist()
    good_slots = df_fc[df_fc["ProbCongested"] <= 0.3]["TimeLabel"].tolist()

    st.markdown("### Tóm tắt nhanh 2 giờ tới")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Thời gian kỳ vọng có nguy cơ tắc",
            f"{expected_congested_minutes:,.0f} phút",
            help="Tổng Mức độ kẹt xe của 4 khung × 30 phút",
        )
    with col2:
        st.metric(
            "Số khung 30' nguy cơ cao",
            f"{len(avoid_slots)} / {len(df_fc)}",
            help="Mức độ kẹt xe ≥ 0.7 được coi là nguy cơ cao",
        )
    with col3:
        st.metric(
            "Mức độ kẹt xe trung bình (GRU/LSTM)",
            f"{avg_prob*100:,.1f} %",
        )

    summary_lines = []
    if avoid_slots:
        summary_lines.append(
            "• **Khung nên tránh** (Mức độ kẹt xe ≥ 0.7): " + ", ".join(avoid_slots)
        )
    else:
        summary_lines.append(
            "• Không có khung giờ nào Mức độ kẹt xe ≥ 0.7 trong 2 giờ tới."
        )

    if good_slots:
        summary_lines.append(
            "• **Khung nên đi** (Mức độ kẹt xe ≤ 0.3): " + ", ".join(good_slots)
        )
    else:
        summary_lines.append(
            "• Không có khung giờ nào thực sự rất thoáng (Mức độ kẹt xe ≤ 0.3) trong 2 giờ tới."
        )

    st.markdown("<br>".join(summary_lines), unsafe_allow_html=True)

    # ======== BIỂU ĐỒ P(TẮC) 2H TỚI ========
    p_min = float(df_fc["ProbCongested"].min())
    p_max = float(df_fc["ProbCongested"].max())
    span = max(1e-3, p_max - p_min)
    pad = max(0.02, span * 0.3)

    y_low = max(0.0, p_min - pad)
    y_high = min(1.0, p_max + pad)

    base = alt.Chart(df_fc).encode(
        x=alt.X("DateTime:T", title="Thời gian (30' tiếp theo)"),
    )

    tooltip = [
        alt.Tooltip("DateTime:T", title="Thời gian"),
        alt.Tooltip("ProbCongested:Q", title="Trung bình (GRU/LSTM)", format=".2f"),
        alt.Tooltip("Prob_GRU:Q", title="GRU", format=".2f"),
        alt.Tooltip("Prob_LSTM:Q", title="LSTM", format=".2f"),
        alt.Tooltip("Level:N", title="Mức độ"),
    ]

    color_scale = alt.Scale(
        domain=["low", "medium", "high"],
        range=["seagreen", "orange", "red"],
    )

    area = base.mark_area(opacity=0.25).encode(
        y=alt.Y(
            "ProbCongested:Q",
            title="Mức độ kẹt xe",
            scale=alt.Scale(domain=[y_low, y_high]),
        ),
        color=alt.value("#eeeeee"),
    )

    line = base.mark_line().encode(
        y=alt.Y(
            "ProbCongested:Q",
            title="Mức độ kẹt xe",
            scale=alt.Scale(domain=[y_low, y_high]),
        ),
        tooltip=tooltip,
    )

    points = base.mark_point(size=80).encode(
        y="ProbCongested:Q",
        color=alt.Color(
            "Level:N",
            title="Mức độ tắc",
            scale=color_scale,
            legend=alt.Legend(
                title="Mức độ tắc",
                orient="top",
            ),
        ),
        tooltip=tooltip,
    )

    labels = base.mark_text(
        dy=-8,
        fontSize=11,
        fontWeight="bold",
        color="#333333",
    ).encode(
        y="ProbCongested:Q",
        text=alt.Text("ProbCongested:Q", format=".0%"),
    )

    chart = (area + line + points + labels).properties(
        height=260,
        title="Dự báo xác suất tắc trong 2 giờ tới",
    ).interactive()

    st.altair_chart(chart, use_container_width=True)

    # =========================
    # Ước lượng thời gian di chuyển trong 2 giờ tới
    # =========================

    # Chuẩn hóa df_slots cho hàm make_travel_time_table_for_slots
    df_slots = df_fc[["TimeLabel", "ProbCongested"]].copy()
    df_slots.rename(
        columns={
            "TimeLabel": "SlotLabel",
            "ProbCongested": "P_cong",
        },
        inplace=True,
    )

    try:
        df_tt = make_travel_time_table_for_slots(df_slots, route_id)
    except Exception as ex:
        st.warning(
            "Không tính được thời gian di chuyển "
            "(kiểm tra lại make_travel_time_table_for_slots / tên cột). "
            f"Chi tiết: {ex}"
        )
        # vẫn tiếp tục hiển thị bảng ngang mức độ kẹt xe
        df_tt = None

    if df_tt is not None:
        T_free = getattr(df_tt, "_T_free", None)
        length_km = getattr(df_tt, "_length_km", None)

        st.markdown("### ⏱ Ước lượng thời gian di chuyển trong 2 giờ tới")

        avg_travel = float(df_tt["Thời gian di chuyển (phút)"].mean())
        worst_travel = float(df_tt["Thời gian di chuyển (phút)"].max())
        worst_slot = df_tt.loc[
            df_tt["Thời gian di chuyển (phút)"].idxmax(), "Khung giờ"
        ]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Thời gian trong điều kiện thoáng",
                f"{T_free:,.1f} phút" if T_free is not None else "-",
                help=(
                    f"Ước tính với chiều dài tuyến ~{length_km:.1f} km, "
                    f"tốc độ thoáng ~{HCMC_FREE_FLOW_SPEED_KMH:.0f} km/h."
                    if (T_free is not None and length_km is not None)
                    else None
                ),
            )
        with col2:
            st.metric(
                "Thời gian di chuyển trung bình (4 khung)",
                f"{avg_travel:,.1f} phút",
            )
        with col3:
            st.metric(
                "Tệ nhất trong 2 giờ tới",
                f"{worst_travel:,.1f} phút",
                help=f"Khung giờ dự kiến tốn thời gian nhất: {worst_slot}.",
            )

        st.markdown("#### Bảng chi tiết theo từng khung 30 phút")
        st.dataframe(df_tt, use_container_width=True)

    # ==== Bảng ngang Mức độ kẹt xe theo từng khung 30' ====
    prob_pct = (df_fc.set_index("TimeLabel")["ProbCongested"] * 100).round(1)
    tbl = prob_pct.to_frame().T
    tbl.index = ["Mức độ kẹt xe (%)"]

    styled_tbl = (
        tbl.style
        .format("{:,.1f}", na_rep="-")
        .background_gradient(axis=1, cmap="RdYlGn_r")
        .highlight_max(axis=1, color="#8B0000")
    )

    st.dataframe(styled_tbl, use_container_width=True, height=80)

    st.markdown(
        """
        <div style="font-size:0.9rem; margin-top:4px;">
          <b>Chú thích màu:</b>
          <span style="display:inline-block;width:14px;height:14px;background-color:#006400;border-radius:3px;margin:0 4px 0 8px;border:1px solid #ccc;"></span>
          Xanh = nguy cơ tắc thấp
          <span style="display:inline-block;width:14px;height:14px;background-color:#FFD700;border-radius:3px;margin:0 4px 0 12px;border:1px solid #ccc;"></span>
          Vàng = trung bình
          <span style="display:inline-block;width:14px;height:14px;background-color:#8B0000;border-radius:3px;margin:0 4px 0 12px;border:1px solid #ccc;"></span>
          Đỏ = nguy cơ tắc cao
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hcmc_departure_advisor(route_id: str, routes_geo_all: pd.DataFrame):
    """
    Trợ lý chọn giờ đi đường cho HCMC:
    - Dựa trên lịch sử train.csv
    - Gợi ý khung giờ nên đi / nên tránh trong ngày hôm nay
      cho tuyến đã chọn.
    """
    out = _load_hcmc_series_for_route(route_id, routes_geo_all)
    if out is None:
        st.info("Không đủ dữ liệu lịch sử để tư vấn giờ đi cho tuyến này.")
        return

    s, full_name, street_name = out

    # Chuẩn bị DataFrame lịch sử: mỗi mốc thời gian = 0/1 (kẹt / không)
    df_hist = s.to_frame(name="is_congested")
    df_hist["DateTime"] = df_hist.index
    df_hist["hour"] = df_hist["DateTime"].dt.hour
    df_hist["minute"] = df_hist["DateTime"].dt.minute
    df_hist["weekday"] = df_hist["DateTime"].dt.weekday

    st.subheader(" Trợ lý chọn giờ đi đường")

    st.markdown(
        f"Dựa trên dữ liệu lịch sử của tuyến **{full_name}**, "
        "gợi ý khung giờ nên đi / nên tránh cho **ngày hôm nay**."
    )

    now = pd.Timestamp.now(tz="Asia/Ho_Chi_Minh")
    today_wd = now.weekday()

    # Chọn khung giờ quan tâm
    window_label = st.selectbox(
        "Chọn khung giờ bạn quan tâm",
        ["Sáng (06:00–09:00)", "Chiều (16:00–19:00)"],
        key="hcmc_advisor_window",
    )

    if window_label.startswith("Sáng"):
        start_hour, end_hour = 6, 9
    else:
        start_hour, end_hour = 16, 19

    # Tạo list slot 30' trong khoảng [start_hour, end_hour)
    slots = []
    h = start_hour
    m = 0
    while h < end_hour:
        slots.append((h, m))
        if m == 0:
            m = 30
        else:
            m = 0
            h += 1

    rows = []
    for (h, m) in slots:
        subset = df_hist[(df_hist["hour"] == h) & (df_hist["minute"] == m)]
        if subset.empty:
            mean_cong = np.nan
        else:
            # Ưu tiên dùng đúng thứ trong tuần hôm nay, nếu đủ mẫu
            subset_today = subset[subset["weekday"] == today_wd]
            if len(subset_today) >= 5:
                mean_cong = subset_today["is_congested"].mean()
            else:
                mean_cong = subset["is_congested"].mean()
        rows.append({"hour": h, "minute": m, "MeanCongestion": mean_cong})

    df_window = pd.DataFrame(rows).dropna(subset=["MeanCongestion"])
    if df_window.empty:
        st.info(
            "Không đủ dữ liệu lịch sử để tư vấn khung giờ cho tuyến này trong khoảng đã chọn."
        )
        return

    df_window["TimeLabel"] = df_window.apply(
        lambda r: f"{int(r['hour']):02d}:{int(r['minute']):02d}", axis=1
    )
    df_window["CongestionPct"] = (df_window["MeanCongestion"] * 100.0).round(1)

    # ====== Tìm khung nên đi / nên tránh theo ngưỡng phần trăm ======
    avg_pct = float(df_window["CongestionPct"].mean())

    GOOD_THR = 30.0  # <= 30%: nên đi
    BAD_THR = 70.0   # >= 70%: nên tránh

    good = df_window[df_window["CongestionPct"] <= GOOD_THR]
    bad = df_window[df_window["CongestionPct"] >= BAD_THR]

    # Khung nên đi: ưu tiên tất cả khung "good"; nếu không có thì lấy 1–2 khung nhỏ nhất
    if not good.empty:
        best_list = (
            good.sort_values("CongestionPct")[["TimeLabel"]]
            .drop_duplicates()
            .iloc[:, 0]
            .tolist()
        )
    else:
        best_list = (
            df_window.nsmallest(2, "CongestionPct")[["TimeLabel"]]
            .iloc[:, 0]
            .tolist()
        )

    # Khung nên tránh: ưu tiên tất cả khung "bad"; nếu không có và có kẹt >0% thì lấy 1–2 khung lớn nhất
    if not bad.empty:
        worst_list = (
            bad.sort_values("CongestionPct", ascending=False)[["TimeLabel"]]
            .drop_duplicates()
            .iloc[:, 0]
            .tolist()
        )
    else:
        if df_window["CongestionPct"].max() > 0:
            worst_list = (
                df_window.nlargest(2, "CongestionPct")[["TimeLabel"]]
                .iloc[:, 0]
                .tolist()
            )
        else:
            worst_list = []

    best_str = ", ".join(best_list)
    worst_str = ", ".join(worst_list)


    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Khung nên đi (ít kẹt nhất)",
            best_str or "-",
        )
    with col2:
        st.metric(
            "Khung nên tránh (kẹt nhất)",
            worst_str or "-",
        )
    with col3:
        st.metric(
            "Mức độ kẹt xe trung bình",
            f"{avg_pct:,.1f} %",
        )

    # st.markdown(
    #     f"- **Khung nên đi**: {best_str if best_str else 'chưa rõ do thiếu dữ liệu'}  \n"
    #     f"- **Khung nên tránh**: {worst_str if worst_str else 'chưa rõ do thiếu dữ liệu'}"
    # )

    # Biểu đồ cột mức độ kẹt theo từng slot
    chart = (
        alt.Chart(df_window)
        .mark_bar()
        .encode(
            x=alt.X("TimeLabel:N", title="Khung giờ (30 phút)"),
            y=alt.Y(
                "CongestionPct:Q",
                title="Mức độ kẹt xe trung bình (%)",
            ),
            color=alt.Color(
                "CongestionPct:Q",
                scale=alt.Scale(scheme="RdYlGn_r"),  # thấp = xanh, cao = đỏ
                legend=alt.Legend(title="Kẹt xe (%)"),
            ),
            tooltip=[
                alt.Tooltip("TimeLabel:N", title="Khung giờ"),
                alt.Tooltip(
                    "CongestionPct:Q",
                    title="Mức độ kẹt xe (%)",
                    format=".1f",
                ),
            ],
        )
        .properties(height=260, title="Mức độ kẹt xe trung bình theo khung 30 phút")
    )

    st.altair_chart(chart, use_container_width=True)

def render_hcmc_weekly_pattern(route_id: str, routes_geo_all: pd.DataFrame):
    """
    Hiển thị 'heatmap' mẫu hình kẹt xe theo giờ & thứ trong tuần
    cho một tuyến HCMC, dạng bảng màu (pandas.style).
    """
    out = _load_hcmc_series_for_route(route_id, routes_geo_all)
    if out is None:
        st.info("Không đủ dữ liệu lịch sử để hiển thị mẫu hình tuần cho tuyến này.")
        return

    s, full_name, street_name = out

    df = s.to_frame(name="is_congested")
    if df.empty:
        st.info("Không đủ dữ liệu lịch sử để hiển thị mẫu hình tuần cho tuyến này.")
        return

    df["DateTime"] = df.index
    df["hour"] = df["DateTime"].dt.hour
    df["weekday"] = df["DateTime"].dt.weekday  # 0=Mon ... 6=Sun

    weekday_map = {
        0: "Thứ 2",
        1: "Thứ 3",
        2: "Thứ 4",
        3: "Thứ 5",
        4: "Thứ 6",
        5: "Thứ 7",
        6: "Chủ nhật",
    }
    df["weekday_label"] = df["weekday"].map(weekday_map)

    # Nhóm theo (weekday_label, hour) để lấy tỉ lệ kẹt trung bình
    grp = (
        df.groupby(["weekday_label", "hour"], as_index=False)["is_congested"]
        .mean()
    )
    if grp.empty:
        st.info("Không đủ dữ liệu lịch sử để hiển thị mẫu hình tuần cho tuyến này.")
        return

    grp["CongestionPct"] = (grp["is_congested"] * 100.0).round(1)
    grp["HourStr"] = grp["hour"].astype(int).astype(str).str.zfill(2) + ":00"

    st.subheader("Mẫu hình kẹt xe trong tuần theo giờ")
    st.markdown(
        "Màu càng đỏ = tuyến càng thường xuyên kẹt tại khung giờ đó "
        "(tính theo lịch sử trong tập dữ liệu HCMC)."
    )

    # Pivot thành bảng 7 x 24 (thứ x giờ)
    pivot = grp.pivot_table(
        index="weekday_label",
        columns="HourStr",
        values="CongestionPct",
        aggfunc="mean",
    )

    # Sắp xếp thứ theo đúng thứ tự
    order_idx = ["Thứ 2", "Thứ 3", "Thứ 4", "Thứ 5", "Thứ 6", "Thứ 7", "Chủ nhật"]
    pivot = pivot.reindex(order_idx)

    # Sắp xếp giờ theo thứ tự thời gian
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)

    # Đảm bảo giá trị là float (NaN chuẩn)
    pivot_float = pivot.astype("float")

    # Hàm style riêng cho ô không có dữ liệu
    def style_na(v):
        if pd.isna(v):
            # nền trắng, chữ xám nhạt (có thể đổi 'No data' tùy thích)
            return "background-color: #ffffff; color: #999999;"
        return ""

    styled = (
        pivot_float.style
        # tô heatmap cho các ô có số
        .background_gradient(cmap="RdYlGn_r", axis=None)
        # format số, ô NaN thì để trống hoặc ghi 'None' tùy bạn
        .format("{:.1f}", na_rep="None")   # hoặc na_rep="" nếu muốn ô trống
        # override lại style cho ô NaN (đặt sau background_gradient để đè màu)
        .applymap(style_na)
    )

    st.dataframe(styled, use_container_width=True)


# ======================================================
# MAIN APP
# ======================================================
def main():
    if "last_clicked_route_id" not in st.session_state:
        st.session_state["last_clicked_route_id"] = None

    st.set_page_config(page_title="Traffic Forecast App", layout="wide")
    st.title(" Traffic Forecast App ")

    # Apply pending selection từ map (trước khi tạo widget)
    if "pending_city" in st.session_state:
        st.session_state["city"] = st.session_state.pop("pending_city")
    if "pending_zone" in st.session_state:
        st.session_state["zone"] = st.session_state.pop("pending_zone")
    if "pending_route" in st.session_state:
        # route từ map → đồng bộ trực tiếp vào widget selectbox "Route"
        st.session_state["route"] = st.session_state.pop("pending_route")

    # ====================================
    # 1) SIDEBAR: CITY / ZONE / ROUTE
    # ====================================

    # ----- CITY -----
    cities = list_cities()
    if "HoChiMinh" not in cities: #TODO :  enhance this later
        cities.append("HoChiMinh")
    if not cities:
        st.error("Không tìm thấy city nào trong data/processed_ds.")
        return

    CITY_PLACEHOLDER = "(Chọn city)"
    city_options = [CITY_PLACEHOLDER] + cities

    if "city" not in st.session_state:
        st.session_state["city"] = CITY_PLACEHOLDER

    city_selected = st.sidebar.selectbox(
        "City",
        city_options,
        key="city",
    )

    has_city = city_selected != CITY_PLACEHOLDER
    current_city = city_selected if has_city else None

    # ----- ZONE -----
    if not has_city:
        # Chưa chọn city → disable zone, dùng key khác (không phải "zone")
        st.sidebar.selectbox(
            "Zone",
            ["(Chọn city trước)"],
            key="zone_placeholder",
            disabled=True,
        )
        zone = None
        current_zone = None
    else:
        zones = list_zones(current_city)

        # Trường hợp city không có zone (ví dụ HoChiMinh)
        if not zones:
            st.sidebar.selectbox(
                "Zone",
                ["(Không có zone – dùng toàn city)"],
                key="zone_info",
                disabled=True,
            )
            zone = None
            current_zone = None
        else:
            # Nếu có nhiều zone:
            #  - đưa "(All)" lên đầu
            #  - nếu chưa có "(All)" mà >1 zone → thêm "(All)" vào đầu
            if "(All)" in zones:
                zones = ["(All)"] + [z for z in zones if z != "(All)"]
            elif len(zones) > 1:
                zones = ["(All)"] + zones

            # Default zone:
            #   - Nếu có "(All)" → chọn "(All)"
            #   - Nếu chỉ có 1 zone → chọn đúng zone đó
            if "zone" not in st.session_state or st.session_state["zone"] not in zones:
                default_zone = "(All)" if "(All)" in zones else zones[0]
                st.session_state["zone"] = default_zone

            zone = st.sidebar.selectbox(
                "Zone",
                zones,
                key="zone",        # CHỈ dùng key="zone" ở đây
                disabled=False,
            )
            current_zone = zone
    # alias cho phần còn lại của code
    city = current_city
    zone = current_zone

    # ====================================
    # 2) LOAD MODEL CONTEXT (FALLBACK zone='(All)')
    # ====================================
    # Mặc định: chưa có ctx / model nếu chưa chọn city
    ctx = None
    MODEL_GRU = None
    MODEL_RNN = None
    META = None
    SCALER = None
    ROUTES_MODEL = None
    RID2IDX = None
    LOOKBACK = None
    HORIZON = None

    if has_city:
        ctx = None
        zone_for_model = None if zone == "(All)" else zone

        # HoChiMinh: KHÔNG dùng ModelManager seq2seq (I94/Fremont),
        # mà dùng pipeline riêng GRU congestion → bỏ qua
        if city != "HoChiMinh":
            try:
                ctx = get_model_context(city, zone_for_model)
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
                        st.error(str(e))
                        return
                    else:
                        st.info(
                            f"Không có model tổng cho city={city}, zone='(All)'. "
                            f"Đang dùng model của zone='{zone_for_model}'."
                        )
                else:
                    st.error(str(e))
                    return
        else:
            # HoChiMinh không có ctx seq2seq
            zone_for_model = None
            ctx = None

        # Tách context khi đã load được ctx (chỉ áp dụng cho Minneapolis / Seattle)
        if ctx is not None:
            MODEL_GRU = ctx.gru_model
            MODEL_RNN = getattr(ctx, "rnn_model", None)
            META = ctx.meta
            SCALER = ctx.scaler
            ROUTES_MODEL = ctx.routes_model
            RID2IDX = ctx.rid2idx
            LOOKBACK = ctx.lookback
            HORIZON = ctx.horizon
        else:
            MODEL_GRU = None
            MODEL_RNN = None
            META = None
            SCALER = None
            ROUTES_MODEL = None
            RID2IDX = None
            LOOKBACK = None
            HORIZON = None


    # ====================================
    # 3) ROUTE (sidebar)
    # ====================================
    ROUTE_PLACEHOLDER = "(Chọn route)"

    # luôn khai báo raw_routes, kể cả khi chưa chọn city
    raw_routes = []

    if not has_city:
        # Chưa chọn city → disable route
        route_selected = st.sidebar.selectbox(
            "Route",
            [ROUTE_PLACEHOLDER],
            key="route",
            disabled=True,
        )
        route_id = None
    else:
        if city == "HoChiMinh":
            # HCMC: lấy route từ routes_geo, hiển thị name, value = route_id
            routes_geo_all_sidebar = load_routes_geo().fillna("")
            df_geo_city_sb = routes_geo_all_sidebar[
                routes_geo_all_sidebar["city"] == "HoChiMinh"
            ].copy()

            if df_geo_city_sb.empty:
                st.error("Không tìm thấy tuyến HCMC nào trong routes_geo.")
                route_selected = ROUTE_PLACEHOLDER
                route_id = None
            else:
                route_ids = df_geo_city_sb["route_id"].astype(str).tolist()
                id2name = {
                    r["route_id"]: r["name"]
                    for _, r in df_geo_city_sb.iterrows()
                }

                options = [ROUTE_PLACEHOLDER] + route_ids

                if "route" not in st.session_state:
                    st.session_state["route"] = ROUTE_PLACEHOLDER
                elif (
                    st.session_state["route"] != ROUTE_PLACEHOLDER
                    and st.session_state["route"] not in route_ids
                ):
                    st.session_state["route"] = ROUTE_PLACEHOLDER

                route_selected = st.sidebar.selectbox(
                    "Route",
                    options,
                    key="route",
                    format_func=lambda rid: (
                        id2name.get(rid, rid)
                        if rid != ROUTE_PLACEHOLDER
                        else ROUTE_PLACEHOLDER
                    ),
                )
                route_id = None if route_selected == ROUTE_PLACEHOLDER else route_selected
        else:
            # City khác (Minneapolis, Seattle, ...) dùng route từ parquet như cũ
            raw_routes = list_routes(city, None if zone == "(All)" else zone)
            if not raw_routes:
                st.error("⚠️ Không tìm thấy RouteId nào trong parquet cho city/zone này.")
                return

            route_options = [ROUTE_PLACEHOLDER] + raw_routes

            if "route" not in st.session_state:
                st.session_state["route"] = ROUTE_PLACEHOLDER
            elif (
                st.session_state["route"] != ROUTE_PLACEHOLDER
                and st.session_state["route"] not in raw_routes
            ):
                st.session_state["route"] = ROUTE_PLACEHOLDER

            route_selected = st.sidebar.selectbox(
                "Route",
                route_options,
                key="route",
                disabled=False,
            )
            route_id = None if route_selected == ROUTE_PLACEHOLDER else route_selected

    # ====================================
    # 4) TOP-2 MODELS (cho ensemble forecast)
    # ====================================
    top_models = []

    # Chỉ load summary khi đã có ctx + đã chọn route
    if ctx is not None and route_id:
        summary_top2 = load_top2_summary(ctx.family_name, route_id)
        if summary_top2 and "top_models" in summary_top2:
            top_models = summary_top2["top_models"]
        else:
            # fallback: nếu không có summary, ưu tiên GRU rồi RNN
            if ctx.gru_model is not None:
                top_models.append("GRU")
            if getattr(ctx, "rnn_model", None) is not None:
                top_models.append("RNN")

        if not top_models:
            top_models = ["GRU"]

        # Luôn thêm LSTM nếu có artifacts để ensemble với GRU
        lstm_available = load_lstm_artifacts_for_family(ctx.family_name) is not None
        if lstm_available and "LSTM" not in top_models:
            top_models.append("LSTM")
    else:
        # Chưa chọn city/route → chưa show gì, chỉ map + message "chọn route"
        pass

    # ----- OPTIONS -----
    tab = st.sidebar.radio("Options", ["FORECAST", "METRICS AND EVALUATION"])

    # ====================================
    # 5) MAP COMPONENT
    # ====================================
    st.subheader("Bản đồ các tuyến đường")

    routes_geo_all = load_routes_geo().fillna("")

    df_geo_city = routes_geo_all[routes_geo_all["city"] == city].copy()
    routes_data = df_geo_city.to_dict("records")
    df_all_geo = routes_geo_all.dropna(subset=["lat", "lon"]).copy()
    all_routes_list = df_all_geo.to_dict("records")

    clicked_route_id = map_routes(
        routes_data=routes_data,
        selected_route_id=route_id,
        all_routes=all_routes_list,
        key="traffic_map",
    )

    if clicked_route_id is not None:
        # Chỉ xử lý nếu thực sự khác lần trước
        if clicked_route_id != st.session_state.get("last_clicked_route_id"):
            st.session_state["last_clicked_route_id"] = clicked_route_id

            row = routes_geo_all[
                routes_geo_all["route_id"].str.strip().str.lower()
                == str(clicked_route_id).strip().lower()
                ]

            if not row.empty:
                st.session_state["pending_city"] = row.iloc[0]["city"]
                st.session_state["pending_zone"] = row.iloc[0]["zone"]
                st.session_state["pending_route"] = clicked_route_id
            else:
                st.session_state["pending_route"] = clicked_route_id

            st.rerun()
    if route_id:
        display_name = route_id
        if city == "HoChiMinh":
            row_dn = routes_geo_all[
                (routes_geo_all["city"] == "HoChiMinh")
                & (routes_geo_all["route_id"] == route_id)
            ]
            if not row_dn.empty:
                display_name = row_dn.iloc[0]["name"]
        st.write(f"**Đang chọn tuyến:** {display_name}")
    else:
        st.write("**Chưa chọn tuyến nào**")


    # nếu chưa có route thì chỉ show map, không load data/model
    if not route_id:
        st.info("👆 Hãy chọn một tuyến ở sidebar hoặc click vào marker trên bản đồ để xem forecast chi tiết.")
        return

    # HCMC: dùng GRU congestion riêng, không dùng pipeline Vehicles/h như I-94/Fremont
    if city == "HoChiMinh":
        # 1) Dự báo 2 giờ tới cho tuyến đang chọn
        render_hcmc_congestion_next_2h(route_id, routes_geo_all)

        st.markdown("---")

        # 2) Trợ lý chọn giờ đi đường (dựa trên lịch sử)
        render_hcmc_departure_advisor(route_id, routes_geo_all)
        # 3) Heatmap mẫu hình cả tuần
        render_hcmc_weekly_pattern(route_id, routes_geo_all)
        st.markdown("---")
        render_hcmc_eval_summary_for_route(route_id)
        return

    # ====================================
    # 6) LOAD FULL DATA FOR ROUTE
    # ====================================
    df_full = load_slice(
        city=city,
        zone=None if zone == "(All)" else zone,
        routes=[route_id],
        start_dt=None,
        end_dt=None,
    )

    if df_full.empty:
        # st.error("⚠️ Không đọc được dữ liệu history cho route này.")
        return

    df_full = df_full.copy()
    df_full["DateTime"] = pd.to_datetime(df_full["DateTime"], errors="coerce")
    df_full = df_full.dropna(subset=["DateTime"])

    min_dt = df_full["DateTime"].min()
    max_dt = df_full["DateTime"].max()

    # ====================================
    # 7) FORECAST – tuần kế tiếp sau dữ liệu gốc (ensemble GRU/RNN)
    # ====================================
    if tab == "FORECAST":
        st.header(" Dự đoán lưu lượng giao thông cho hôm nay và 7 ngày tới")

        dfs_for_ensemble = []

        for m_name in top_models:
            if m_name not in ("GRU", "RNN", "LSTM"):
                # bỏ qua model lạ (ví dụ ARIMA) nếu lỡ ghi vào JSON
                continue

            if m_name in ("GRU", "RNN"):
                # logic cũ: dùng forecast_week_after_last_point với GRU/RNN
                df_m, anchor_m = forecast_week_after_last_point(
                    route_id=route_id,
                    city=city,
                    zone=None if zone == "(All)" else zone,
                    ctx=ctx,
                    n_days=8,
                    model_type=m_name,
                )
            elif m_name == "LSTM":
                # NEW: forecast tuần bằng LSTM riêng
                df_m, anchor_m = forecast_week_after_last_point_lstm(
                    route_id=route_id,
                    city=city,
                    zone=None if zone == "(All)" else zone,
                    ctx=ctx,
                    n_days=8,
                )
            else:
                df_m, anchor_m = None, None

            if df_m is not None and not df_m.empty:
                dfs_for_ensemble.append((m_name, df_m, anchor_m))


        if not dfs_for_ensemble:
            st.warning("Không forecast được bằng GRU/RNN top-2, fallback GRU.")
            df_fc_raw, anchor_day_raw = forecast_week_after_last_point(
                route_id=route_id,
                city=city,
                zone=None if zone == "(All)" else zone,
                ctx=ctx,
                n_days=8,
                model_type="GRU",
            )
            if df_fc_raw is not None and not df_fc_raw.empty:
                df_fc_raw = df_fc_raw.copy()
                df_fc_raw["DateTime"] = pd.to_datetime(
                    df_fc_raw["DateTime"], errors="coerce"
                )
                df_fc_raw = df_fc_raw.dropna(subset=["DateTime"])
                df_fc_raw = df_fc_raw.rename(
                    columns={"PredictedVehicles": "Pred_GRU"}
                )
                df_fc_raw["Pred_ENSEMBLE"] = df_fc_raw["Pred_GRU"]
                df_fc_raw["PredictedVehicles"] = df_fc_raw["Pred_ENSEMBLE"]
        else:
            anchor_day_raw = dfs_for_ensemble[0][2]

            df_merge = None
            for m_name, df_m, _ in dfs_for_ensemble:
                col = f"Pred_{m_name}"
                tmp = (
                    df_m[["DateTime", "PredictedVehicles"]]
                    .rename(columns={"PredictedVehicles": col})
                )
                df_merge = tmp if df_merge is None else df_merge.merge(
                    tmp, on="DateTime", how="inner"
                )

            if df_merge is not None and not df_merge.empty:
                model_pred_cols = [
                    f"Pred_{m}" for m in top_models if f"Pred_{m}" in df_merge.columns
                ]
                if model_pred_cols:
                    df_merge["Pred_ENSEMBLE"] = df_merge[model_pred_cols].mean(
                        axis=1
                    )
                else:
                    df_merge["Pred_ENSEMBLE"] = np.nan

                # Trung bình riêng của GRU + LSTM để hiển thị tooltip khi hover chart
                if {"Pred_GRU", "Pred_LSTM"} <= set(df_merge.columns):
                    df_merge["Pred_GRU_LSTM_AVG"] = df_merge[
                        ["Pred_GRU", "Pred_LSTM"]
                    ].mean(axis=1)

                df_merge["PredictedVehicles"] = df_merge["Pred_ENSEMBLE"]
                df_fc_raw = df_merge.copy()
            else:
                df_fc_raw = None

        if df_fc_raw is None or df_fc_raw.empty:
            st.warning("Không forecast được (thiếu dữ liệu history).")
        else:
            target_today = pd.Timestamp.today().normalize()
            df_fc = shift_forecast_to_today(
                df_fc_raw,
                anchor_day_raw,
                target_today=target_today,
                drop_past_hours=True,
            )

            if df_fc is None or df_fc.empty:
                st.warning("Không có forecast hợp lệ sau khi dịch về hôm nay.")
                return

            df_fc["DateTime"] = pd.to_datetime(df_fc["DateTime"], errors="coerce")
            df_fc = df_fc.dropna(subset=["DateTime"])

            days = (
                df_fc["DateTime"]
                .dt.normalize()
                .drop_duplicates()
                .sort_values()
                .tolist()
            )

            if days:
                day_tabs = st.tabs([vn_weekday_label(d) for d in days])

                for d, t in zip(days, day_tabs):
                    with t:
                        day_start = d
                        day_end = d + pd.Timedelta(days=1)

                        df_day = df_fc[
                            (df_fc["DateTime"] >= day_start)
                            & (df_fc["DateTime"] < day_end)
                        ].copy()

                        if df_day.empty:
                            st.info("Không có forecast cho ngày này.")
                            continue

                        df_day["DateTime"] = pd.to_datetime(df_day["DateTime"], errors="coerce")
                        df_day = df_day.dropna(subset=["DateTime"])

                        # Chuẩn hóa về chính xác từng giờ (trong trường hợp có phút lẻ)
                        df_day["DateTime"] = df_day["DateTime"].dt.floor("H")

                        # Chỉ gộp các cột số (Vehicles, PredictedVehicles, Pred_GRU, Pred_LSTM, ...)
                        num_cols = df_day.select_dtypes(include="number").columns.tolist()
                        # nếu có cột không muốn gộp thì bỏ ra khỏi num_cols ở đây

                        df_day = (
                            df_day.groupby("DateTime", as_index=False)[num_cols]
                            .mean()
                            .sort_values("DateTime")
                        )

                        # Cột dùng để phân tích: ưu tiên ensemble
                        metric_col = "PredictedVehicles_Ensemble"
                        if metric_col not in df_day.columns:
                            metric_col = "PredictedVehicles"

                        df_day["DateTime"] = pd.to_datetime(
                            df_day["DateTime"], errors="coerce"
                        )
                        df_day = df_day.dropna(subset=["DateTime"])

                        s = (
                            df_day.set_index("DateTime")[metric_col]
                            .astype(float)
                            .sort_index()
                        )

                        if s.empty:
                            st.info("Không có dữ liệu forecast hợp lệ cho ngày này.")
                            continue

                        # === Phân tích giờ cao điểm / vắng nhất / trung bình ===
                        peak_time = s.idxmax()
                        peak_val = float(s.max())

                        low_time = s.idxmin()
                        low_val = float(s.min())

                        avg_val = float(s.mean())
                        st.markdown("### 📈 Phân tích nhanh trong ngày")

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(
                                "Giờ cao điểm nhất",
                                f"{peak_time:%H:%M}",
                                help=f"Khoảng {peak_val:,.0f} vehicles/h",
                            )
                        with col2:
                            st.metric(
                                "Giờ vắng nhất",
                                f"{low_time:%H:%M}",
                                help=f"Khoảng {low_val:,.0f} vehicles/h",
                            )
                        with col3:
                            st.metric(
                                "Lưu lượng trung bình",
                                f"{avg_val:,.0f} xe/giờ",
                            )
                        # Bảng ngang
                        # st.markdown("### 🧮 Lưu lượng xe cộ theo giờ")
                        #
                        # # s: Series index = DateTime, value = Vehicles/h (ensemble)
                        # s_label = s.copy()
                        # s_label.index = s_label.index.strftime("%H:%M")
                        # s_label_int = s_label.round(0).astype("Int64")  # convert to int, nullable
                        #
                        # # 1 dòng, các cột là giờ
                        # tbl = s_label_int.to_frame().T
                        # tbl.index = ["Vehicles/h"]
                        #
                        # styled_tbl = (
                        #     tbl.style
                        #     .format("{:,.0f}", na_rep="-")  # hiển thị int, có phân cách
                        #     .background_gradient(axis=1, cmap="YlOrRd")  # thấp = vàng nhạt, cao = đỏ
                        #     .highlight_max(axis=1, color="#7f0000   ")  # giờ cao điểm nhất tô đỏ hẳn
                        # )
                        #
                        # st.dataframe(styled_tbl, use_container_width=True, height=70)
                        # st.dataframe(styled_tbl, use_container_width=True, height=140)

                        st.markdown("### 🧮 Lưu lượng xe cộ theo giờ")

                        # s: Series index = DateTime, value = Vehicles/h (ensemble)
                        s_label = s.copy()
                        s_label.index = s_label.index.strftime("%H:%M")
                        s_label_int = s_label.round(0).astype("Int64")  # convert to int, nullable

                        # Gộp theo giờ nếu vì lý do gì đó có trùng label (vd nhiều ngày dính chung)
                        # -> đảm bảo mỗi giờ (cột) chỉ xuất hiện 1 lần
                        s_hour = s_label_int.groupby(s_label_int.index).mean()

                        # 1 dòng, các cột là giờ
                        tbl = s_hour.to_frame().T
                        tbl.index = ["Vehicles/h"]

                        # Phòng trường hợp vẫn còn trùng cột (rất hiếm) -> bỏ bớt bản trùng
                        tbl = tbl.loc[:, ~tbl.columns.duplicated()]

                        styled_tbl = (
                            tbl.style
                            .format("{:,.0f}", na_rep="-")  # hiển thị int, có phân cách
                            .background_gradient(axis=1, cmap="YlOrRd")  # thấp = vàng nhạt, cao = đỏ
                            .highlight_max(axis=1, color="#ff4b4b")  # giờ cao điểm nhất tô đỏ hẳn
                        )

                        st.dataframe(styled_tbl, use_container_width=True, height=70)

                        # Chú giải màu
                        st.markdown(
                            """
                            <div style="font-size:0.9rem; margin-top:4px;">
                              <b>Chú thích màu:</b>
                              <span style="display:inline-block;width:14px;height:14px;background-color:#008000;border-radius:3px;margin:0 4px 0 8px;border:1px solid #ccc;"></span>
                              Xanh lá  = lưu lượng thấp / thưa thớt
                              <span style="display:inline-block;width:14px;height:14px;background-color:#FFD700;border-radius:3px;margin:0 4px 0 12px;border:1px solid #ccc;"></span>
                              Vàng = trung bình
                              <span style="display:inline-block;width:14px;height:14px;background-color:#CC0000;border-radius:3px;margin:0 4px 0 12px;border:1px solid #ccc;"></span>
                              Đỏ = giờ rất đông (cao điểm)
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        st.markdown(f"**Mô hình sử dụng:** {', '.join(top_models)}")

                        # Tooltip hiển thị từng model nếu có
                        tooltip_fields = [
                            alt.Tooltip("DateTime:T", title="Thời gian"),
                            alt.Tooltip(
                                "Pred_ENSEMBLE:Q",
                                title="Gía trị trung bình",
                                format=".0f",
                            ),
                        ]
                        if "Pred_GRU" in df_day.columns:
                            tooltip_fields.append(
                                alt.Tooltip("Pred_GRU:Q", title="GRU", format=".0f")
                            )
                        if "Pred_RNN" in df_day.columns:
                            tooltip_fields.append(
                                alt.Tooltip("Pred_RNN:Q", title="RNN", format=".0f")
                            )
                        if "Pred_LSTM" in df_day.columns:
                            tooltip_fields.append(
                                alt.Tooltip("Pred_LSTM:Q", title="LSTM", format=".0f")
                            )

                        df_day = df_day.copy()

                        q_low = df_day["PredictedVehicles"].quantile(0.2)
                        q_high = df_day["PredictedVehicles"].quantile(0.8)

                        def level_label(v):
                            if v >= q_high:
                                return "Rất đông"
                            elif v <= q_low:
                                return "Thưa thớt"
                            else:
                                return "Trung bình"

                        df_day["TrafficLevel"] = df_day["PredictedVehicles"].apply(level_label)

                        base = alt.Chart(df_day).encode(
                            x=alt.X("DateTime:T", title="Thời gian")
                        )

                        line = base.mark_line(color="lightgray").encode(
                            y=alt.Y("PredictedVehicles:Q", title="Vehicles"),
                        )
                        color_scale = alt.Scale(
                            domain=["Thưa thớt", "Trung bình", "Rất đông"],
                            range=["#008000", "#0000ff", "#CC0000"],
                        )
                        points = base.mark_point(size=70).encode(
                            y="PredictedVehicles:Q",
                            color=alt.Color(
                                "TrafficLevel:N",
                                scale=color_scale,
                                legend=alt.Legend(title="Mức lưu lượng"),
                            ),
                            tooltip=tooltip_fields,
                        )

                        labels = base.mark_text(
                            dy=-8,
                            fontSize=11,
                            fontWeight="bold",
                            color="#333333",
                        ).encode(
                            y="PredictedVehicles:Q",
                            text=alt.Text("PredictedVehicles:Q", format=".0f"),
                        )

                        chart = (line + points + labels).interactive().properties(
                            height=320,
                            title=f"Dự báo cho {vn_weekday_label(day_start)}",
                        )
                        st.altair_chart(chart, use_container_width=True)
            else:
                st.info("Không có ngày nào trong forecast.")

    # ====================================
    # 8) DAILY TRAFFIC – 3 THÁNG GẦN NHẤT
    #     Actual vs GRU / RNN / LSTM / ARIMA / SARIMA + Metrics tổng 3 tháng
    # ====================================
    elif tab == "METRICS AND EVALUATION":
        st.header("📚 Thống kê và đánh giá")

        # Đọc cache do script precompute_daily_3months.py sinh ra:
        #   model/<family_name>/<route_id>_daily_3months.parquet
        cache_dir = Path("model") / ctx.family_name
        cache_path = cache_dir / f"{route_id}_daily_3months.parquet"

        if not cache_path.exists():
            st.info(
                f"⚠️ Chưa tìm thấy file cache daily: {cache_path}. "
                "Hãy chạy scripts/precompute_daily_3months.py trước, hoặc bật lại chế độ tính trực tiếp trong app."
            )
            return

        try:
            df_eval = pd.read_parquet(cache_path)
        except Exception as ex:
            st.error(f"Lỗi đọc file cache daily: {ex}")
            return

        if df_eval is None or df_eval.empty:
            st.info("File cache daily trống, không có dữ liệu để hiển thị.")
            return

        # Chuẩn hóa cột Date
        if "Date" not in df_eval.columns or "DailyActual" not in df_eval.columns:
            st.warning(
                "File cache daily không có đủ cột 'Date' / 'DailyActual'. Kiểm tra lại file precompute."
            )
            return

        df_eval = df_eval.copy()
        df_eval["Date"] = pd.to_datetime(df_eval["Date"]).dt.normalize()
        # ----------------------------------------------------------
        # Fallback: nếu cache chưa có Daily_ARIMA / Daily_SARIMA,
        # nhưng app import được ARIMA/SARIMA thì tính bổ sung tại chỗ.
        # ----------------------------------------------------------
        dates = df_eval["Date"].dropna().drop_duplicates().sort_values().tolist()

        # ---- Fallback ARIMA ----
        if HAS_ARIMA and forecast_arima_for_day is not None and "Daily_ARIMA" not in df_eval.columns:
            records = []
            for d in dates:
                day_start = pd.Timestamp(d).normalize()
                day_end = day_start + pd.Timedelta(days=1)

                try:
                    # theo fix trước đây: forecast_arima_for_day(df_full, day_start)
                    out = forecast_arima_for_day(df_full, day_start)
                    if isinstance(out, tuple):
                        df_fc_arima = out[0]
                    else:
                        df_fc_arima = out
                except Exception as ex:
                    print(f"[Daily-ARIMA] error {day_start.date()}: {ex}")
                    continue

                if df_fc_arima is None or df_fc_arima.empty:
                    continue

                df_a = df_fc_arima.copy()
                df_a["DateTime"] = pd.to_datetime(df_a["DateTime"], errors="coerce")
                df_a = df_a.dropna(subset=["DateTime"])
                df_a = df_a[
                    (df_a["DateTime"] >= day_start)
                    & (df_a["DateTime"] < day_end)
                    ]
                if df_a.empty:
                    continue

                # tuỳ arima_utils: ưu tiên Pred_ARIMA, fallback PredictedVehicles
                pred_col = "Pred_ARIMA" if "Pred_ARIMA" in df_a.columns else "PredictedVehicles"
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
                )
                df_eval = df_eval.merge(df_arima, on="Date", how="left")

        # ---- Fallback SARIMA ----
        if HAS_SARIMA and forecast_sarima_for_day is not None and "Daily_SARIMA" not in df_eval.columns:
            records = []
            for d in dates:
                day_start = pd.Timestamp(d).normalize()
                day_end = day_start + pd.Timedelta(days=1)

                try:
                    out = forecast_sarima_for_day(df_full, day_start)
                    if isinstance(out, tuple):
                        df_fc_sarima = out[0]
                    else:
                        df_fc_sarima = out
                except Exception as ex:
                    print(f"[Daily-SARIMA] error {day_start.date()}: {ex}")
                    continue

                if df_fc_sarima is None or df_fc_sarima.empty:
                    continue

                df_s = df_fc_sarima.copy()
                df_s["DateTime"] = pd.to_datetime(df_s["DateTime"], errors="coerce")
                df_s = df_s.dropna(subset=["DateTime"])
                df_s = df_s[
                    (df_s["DateTime"] >= day_start)
                    & (df_s["DateTime"] < day_end)
                    ]
                if df_s.empty:
                    continue

                pred_col = "Pred_SARIMA" if "Pred_SARIMA" in df_s.columns else "PredictedVehicles"
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
                )
                df_eval = df_eval.merge(df_sarima, on="Date", how="left")

        # ---- Tab ----
        tab_cmp_hourly, tab_cmp_daily, tab_cmp_weekly, tab_cmp_monthly = st.tabs([
            "Hourly", "Daily", "Weekly", "Monthly"
        ])

        # -----------------
        # 7.0 Tab Hourly
        # -----------------
        with tab_cmp_hourly:
            st.subheader("HOURLY – trung bình theo giờ trong 1 tháng gần nhất")

            if df_full.empty:
                st.info("Không có dữ liệu để tính trung bình theo giờ.")
            else:
                max_date = df_full["DateTime"].max()
                start_dt = max_date - pd.DateOffset(months=1)
                df_last_month = df_full[df_full["DateTime"] >= start_dt].copy()

                if df_last_month.empty:
                    st.info("Không có dữ liệu trong 1 tháng gần nhất để hiển thị.")
                else:
                    df_last_month["Hour"] = df_last_month["DateTime"].dt.hour
                    df_hourly = (
                        df_last_month
                        .groupby("Hour")["Vehicles"]
                        .mean()
                        .reindex(range(24))
                        .reset_index()
                        .rename(columns={"Vehicles": "VehiclesPerHour"})
                    )

                    df_hourly["VehiclesPerHour"] = df_hourly["VehiclesPerHour"].round(2)

                    st.caption(
                        f"Khoảng dữ liệu: {start_dt.date()} → {max_date.date()}"
                    )

                    chart_hourly = (
                        alt.Chart(df_hourly)
                        .mark_bar()
                        .encode(
                            x=alt.X("Hour:O", title="Giờ trong ngày"),
                            y=alt.Y(
                                "VehiclesPerHour:Q",
                                title="Lưu lượng trung bình (vehicles/giờ)",
                            ),
                            tooltip=[
                                alt.Tooltip("Hour:O", title="Giờ"),
                                alt.Tooltip(
                                    "VehiclesPerHour:Q",
                                    title="Lưu lượng trung bình",
                                    format=",.2f",
                                ),
                            ],
                        )
                        .properties(height=320)
                    )

                    st.altair_chart(chart_hourly, use_container_width=True)

                    df_table = df_hourly.rename(
                        columns={
                            "Hour": "Giờ",
                            "VehiclesPerHour": "Lưu lượng trung bình",
                        }
                    ).copy()
                    df_table["Giờ"] = df_table["Giờ"].apply(lambda h: f"{h}h")
                    df_table["Lưu lượng trung bình"] = df_table["Lưu lượng trung bình"].apply(
                        lambda v: f"{v:,.2f}" if pd.notna(v) else ""
                    )

                    st.markdown("#### Bảng tổng hợp theo giờ")
                    st.dataframe(df_table, use_container_width=True)

        # -----------------
        # 7.1 Tab Daily
        # -----------------
        with tab_cmp_daily:
            st.subheader("DAILY (Actual + Models) – 3 tháng gần nhất")
            # ==== Chart multi-line (Actual + Models) ====
            frames = [
                df_eval[["Date", "DailyActual"]]
                .rename(columns={"DailyActual": "DailyValue"})
                .assign(Source="Actual")
            ]

            for m_name in ["GRU", "RNN", "LSTM", "ARIMA", "SARIMA"]:
                col_name = f"Daily_{m_name}"
                if col_name in df_eval.columns and df_eval[col_name].notna().any():
                    frames.append(
                        df_eval[["Date", col_name]]
                        .rename(columns={col_name: "DailyValue"})
                        .assign(Source=m_name)
                    )

            if frames:
                df_chart = pd.concat(frames, ignore_index=True)
                df_chart = df_chart.sort_values("Date")

                chart_daily = (
                    alt.Chart(df_chart)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("Date:T", title="Date"),
                        y=alt.Y("DailyValue:Q", title="Vehicles / day"),
                        color=alt.Color("Source:N", title="Series"),
                        tooltip=[
                            alt.Tooltip("Date:T", title="Date"),
                            alt.Tooltip("Source:N", title="Series"),
                            alt.Tooltip("DailyValue:Q", title="Vehicles/day", format=","),
                        ],
                    )
                    .properties(height=300)
                )
                st.altair_chart(chart_daily, use_container_width=True)

                with st.expander(
                        "🔍 Xem thống kê lưu lượng giao thông theo từng ngày  (Actual + Models) trong 3 tháng gần nhất"
                ):
                    df_show = df_eval.copy()
                    weekday_map = {
                        0: "Thứ 2",
                        1: "Thứ 3",
                        2: "Thứ 4",
                        3: "Thứ 5",
                        4: "Thứ 6",
                        5: "Thứ 7",
                        6: "Chủ nhật",
                    }
                    if "Date" in df_show.columns:
                        df_show["Thứ"] = pd.to_datetime(df_show["Date"]).dt.dayofweek.map(
                            weekday_map
                        )
                        cols = df_show.columns.tolist()
                        thu_idx = cols.index("Thứ")
                        date_idx = cols.index("Date")
                        if thu_idx != date_idx + 1:
                            cols.insert(date_idx + 1, cols.pop(thu_idx))
                        df_show = df_show[cols]
                    for c in df_show.columns:
                        if c.startswith("Daily"):
                            df_show[c] = df_show[c].round().astype("Int64").apply(lambda x: f"{x:,.0f}")
                    df_show = df_show.sort_values("Date")
                    if "Thứ" in df_show.columns:
                        weekend_labels = {"Thứ 7", "Chủ nhật"}
                        styler = df_show.style.applymap(
                            lambda v: "font-weight: bold" if v in weekend_labels else "",
                            subset=pd.IndexSlice[:, ["Thứ"]],
                        )
                        st.dataframe(styler, use_container_width=True)
                    else:
                        st.dataframe(df_show, use_container_width=True)
            else:
                st.info("Không có series nào (GRU/RNN/LSTM/ARIMA/SARIMA) để hiển thị.")

            # ==== Metrics tổng 3 tháng cho từng model (nếu có cột) ====
            metrics_rows = []
            for m_name in ["GRU", "RNN", "LSTM", "ARIMA", "SARIMA"]:
                col_name = f"Daily_{m_name}"
                if col_name not in df_eval.columns:
                    continue
                valid = df_eval[["DailyActual", col_name]].dropna()
                if valid.empty:
                    continue

                actual = valid["DailyActual"].values.astype(float)
                pred = valid[col_name].values.astype(float)

                mse = mean_squared_error(actual, pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(actual, pred)

                if np.any(actual != 0):
                    mape = (
                            np.mean(
                                np.abs((actual - pred)[actual != 0] / actual[actual != 0])
                            )
                            * 100.0
                    )
                else:
                    mape = np.nan

                denom = np.abs(actual) + np.abs(pred)
                smape = (
                        np.mean(
                            2.0 * np.abs(pred - actual) / np.where(denom == 0, 1.0, denom)
                        )
                        * 100.0
                )

                r2 = r2_score(actual, pred)

                metrics_rows.append(
                    {
                        "Model": m_name,
                        "MSE": mse,
                        "RMSE": rmse,
                        "MAE": mae,
                        "MAPE (%)": mape,
                        "SMAPE (%)": smape,
                        "R²": r2,
                    }
                )

            if metrics_rows:
                st.subheader(" Đánh giá sai số theo từng model trong 3 tháng gần nhất")
                df_metrics = pd.DataFrame(metrics_rows)
                for c in ["MSE", "RMSE", "MAE"]:
                    df_metrics[c] = df_metrics[c].round(2)
                for c in ["MAPE (%)", "SMAPE (%)", "R²"]:
                    df_metrics[c] = df_metrics[c].round(3)

                # ---- Format số theo dạng 000,000,000.00 ----
                format_cols = ["MSE", "RMSE", "MAE", "MAPE (%)", "SMAPE (%)", "R²"]
                df_formatted = df_metrics.copy()
                for c in ["MSE", "RMSE", "MAE"]:
                    df_formatted[c] = df_formatted[c].apply(lambda x: f"{x:,.2f}")
                for c in ["MAPE (%)", "SMAPE (%)", "R²"]:
                    df_formatted[c] = df_formatted[c].apply(lambda x: f"{x:,.3f}")

                st.dataframe(df_formatted, use_container_width=True)

            # ==== Biểu đồ cột cho từng đánh giá ====
            st.subheader("📊 Biểu đồ cột cho từng đánh giá sai số")
            metrics_list = ["MSE", "RMSE", "MAE", "MAPE (%)", "SMAPE (%)", "R²"]
            cols = st.columns(2) # Tạo layout 2 cột

            for i, metric in enumerate(metrics_list):
                chart = (
                    alt.Chart(df_metrics)
                    .mark_bar()
                    .encode(
                        x=alt.X("Model:N", title="Model", axis=alt.Axis(labelAngle=0)),
                        y=alt.Y(f"{metric}:Q", title=metric),
                        tooltip=["Model", metric]
                    )
                    .properties(
                        height=300,
                        title=alt.TitleParams(
                            f"{metric}",
                            fontSize=24,
                            fontWeight="bold",
                            color="#333",
                            anchor="middle"  # căn giữa
                        )
                    )
                )

                # Vẽ đúng cột (0 hoặc 1)
                cols[i % 2].altair_chart(chart, use_container_width=True)

                # Sau mỗi 2 biểu đồ → tạo hàng mới
                if i % 2 == 1 and i < len(metrics_list) - 1:
                    cols = st.columns(2)

        # -----------------
        # 7.2 Tab Weekly
        # -----------------
        with tab_cmp_weekly:

            df_weekly = df_eval.copy()
            df_weekly["Date"] = pd.to_datetime(df_weekly["Date"])

            # Convert thành tuần
            df_weekly["WeekStart"] = df_weekly["Date"].dt.to_period("W").apply(lambda r: r.start_time)
            df_weekly["WeekEnd"] = df_weekly["Date"].dt.to_period("W").apply(lambda r: r.end_time)

            # Gom weekly (sum cho traffic)
            # Lấy toàn bộ cột Daily_*
            daily_cols = [c for c in df_weekly.columns if c.startswith("Daily")]

            # Tạo dict động cho agg
            agg_dict = {c: "sum" for c in daily_cols}

            # Group
            df_weekly = (
                df_weekly.groupby(["WeekStart", "WeekEnd"])
                .agg(agg_dict)
                .reset_index()
            )

            # Đổi tên cột Daily* → Weekly*
            df_weekly = df_weekly.rename(
                columns={c: c.replace("Daily", "Weekly") for c in daily_cols}
            )

            # Format range: YYYY-MM-DD → YYYY-MM-DD
            df_weekly["WeekRange"] = df_weekly["WeekStart"].dt.strftime("%Y-%m-%d") + " → " + \
                                     df_weekly["WeekEnd"].dt.strftime("%Y-%m-%d")

            # ==== Chart multi-line Weekly (Actual + Models) ====
            st.subheader("WEEKLY (Actual + Models) – 3 tháng gần nhất")

            frames = [
                df_weekly[["WeekStart", "WeeklyActual"]]
                .rename(columns={"WeeklyActual": "WeeklyValue"})
                .assign(Source="Actual")
            ]

            for m_name in ["GRU", "RNN", "LSTM", "ARIMA", "SARIMA"]:
                col_name = f"Weekly_{m_name}"
                if col_name in df_weekly.columns and df_weekly[col_name].notna().any():
                    frames.append(
                        df_weekly[["WeekStart", col_name]]
                        .rename(columns={col_name: "WeeklyValue"})
                        .assign(Source=m_name)
                    )

            if frames:
                df_chart_w = pd.concat(frames, ignore_index=True)
                df_chart_w = df_chart_w.sort_values("WeekStart")

                chart_weekly = (
                    alt.Chart(df_chart_w)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("WeekStart:T", title="Week (Start Date)"),
                        y=alt.Y("WeeklyValue:Q", title="Vehicles / week"),
                        color=alt.Color("Source:N", title="Series"),
                        tooltip=[
                            alt.Tooltip("WeekStart:T", title="Week Start"),
                            alt.Tooltip("Source:N", title="Series"),
                            alt.Tooltip("WeeklyValue:Q", title="Vehicles/week", format=","),
                        ],
                    )
                    .properties(height=300)
                )
                st.altair_chart(chart_weekly, use_container_width=True)

                with st.expander("Xem bảng Weekly (Actual + Models) – tổng hợp theo tuần"):
                    df_weekly_show = df_weekly.copy()
                    for c in df_weekly_show.columns:
                        if c.startswith("Weekly"):
                            df_weekly_show[c] = df_weekly_show[c].round().astype("Int64").apply(lambda x: f"{x:,.0f}")
                    st.dataframe(
                        df_weekly_show[["WeekRange"] +
                                  [c for c in df_weekly_show.columns if
                                   c not in ["Date", "Year", "Week", "WeekRange", "WeekStart", "WeekEnd"]]],
                        use_container_width=True
                    )
            else:
                st.info("Không có series nào (GRU/RNN/LSTM/ARIMA/SARIMA) để hiển thị.")

            # ==== Metrics tổng Weekly cho từng model (nếu có cột) ====
            metrics_rows = []

            for m_name in ["GRU", "RNN", "LSTM", "ARIMA", "SARIMA"]:
                col_name = f"Weekly_{m_name}"
                if col_name not in df_weekly.columns:
                    continue

                valid = df_weekly[["WeeklyActual", col_name]].dropna()
                if valid.empty:
                    continue

                actual = valid["WeeklyActual"].values.astype(float)
                pred = valid[col_name].values.astype(float)

                # Sai số
                mse = mean_squared_error(actual, pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(actual, pred)

                mape = (
                        np.mean(np.abs((actual - pred) / np.where(actual == 0, np.nan, actual))) * 100
                )

                denom = np.abs(actual) + np.abs(pred)
                smape = (
                        np.mean(2.0 * np.abs(pred - actual) / np.where(denom == 0, 1.0, denom)) * 100
                )

                r2 = r2_score(actual, pred)

                metrics_rows.append(
                    {
                        "Model": m_name,
                        "MSE": mse,
                        "RMSE": rmse,
                        "MAE": mae,
                        "MAPE (%)": mape,
                        "SMAPE (%)": smape,
                        "R²": r2,
                    }
                )

            if metrics_rows:
                st.subheader("Đánh giá sai số theo từng model – dữ liệu Weekly (3 tháng gần nhất)")
                df_metrics_weekly = pd.DataFrame(metrics_rows)

                for c in ["MSE", "RMSE", "MAE"]:
                    df_metrics_weekly[c] = df_metrics_weekly[c].round(2)
                for c in ["MAPE (%)", "SMAPE (%)", "R²"]:
                    df_metrics_weekly[c] = df_metrics_weekly[c].round(3)

                # ---- Format số theo dạng 000,000,000.00 ----
                format_cols = ["MSE", "RMSE", "MAE", "MAPE (%)", "SMAPE (%)", "R²"]
                df_formatted_weekly = df_metrics_weekly.copy()
                for c in ["MSE", "RMSE", "MAE"]:
                    df_formatted_weekly[c] = df_formatted_weekly[c].apply(lambda x: f"{x:,.2f}")
                for c in ["MAPE (%)", "SMAPE (%)", "R²"]:
                    df_formatted_weekly[c] = df_formatted_weekly[c].apply(lambda x: f"{x:,.3f}")

                st.dataframe(df_formatted_weekly, use_container_width=True)

            else:
                st.info("Không có dữ liệu Weekly để tính metrics.")

            # ==== Biểu đồ cột cho từng đánh giá ====
            st.subheader("📊 Biểu đồ cột cho từng đánh giá sai số")
            metrics_list = ["MSE", "RMSE", "MAE", "MAPE (%)", "SMAPE (%)", "R²"]
            cols = st.columns(2)  # 2 cột mỗi hàng

            for i, metric in enumerate(metrics_list):
                chart = (
                    alt.Chart(df_metrics_weekly)
                    .mark_bar()
                    .encode(
                        x=alt.X("Model:N", title="Model", axis=alt.Axis(labelAngle=0)),
                        y=alt.Y(f"{metric}:Q", title=metric),
                        tooltip=["Model", metric]
                    )
                    .properties(
                        height=300,
                        title=alt.TitleParams(
                            f"{metric}",
                            fontSize=24,
                            fontWeight="bold",
                            color="#333",
                            anchor="middle"  # căn giữa
                        )
                    )
                )

                # vẽ vào đúng cột
                cols[i % 2].altair_chart(chart, use_container_width=True)

                # tạo hàng kế tiếp sau mỗi 2 chart
                if i % 2 == 1 and i < len(metrics_list) - 1:
                    cols = st.columns(2)

        # -----------------
        # 7.2 Tab Monthly
        # -----------------
        with tab_cmp_monthly:
            df_monthly = df_eval.copy()
            df_monthly["Date"] = pd.to_datetime(df_monthly["Date"])

            # Convert thành tháng
            df_monthly["MonthStart"] = df_monthly["Date"].dt.to_period("M").apply(lambda r: r.start_time)
            df_monthly["MonthEnd"] = df_monthly["Date"].dt.to_period("M").apply(lambda r: r.end_time)

            # Gom monthly (sum cho traffic)
            daily_cols = [c for c in df_monthly.columns if c.startswith("Daily")]

            agg_dict = {c: "sum" for c in daily_cols}

            df_monthly = (
                df_monthly.groupby(["MonthStart", "MonthEnd"])
                .agg(agg_dict)
                .reset_index()
            )

            # Đổi tên Daily* → Monthly*
            df_monthly = df_monthly.rename(
                columns={c: c.replace("Daily", "Monthly") for c in daily_cols}
            )

            # Hiển thị dạng "YYYY-MM-DD → YYYY-MM-DD"
            df_monthly["MonthRange"] = (
                    df_monthly["MonthStart"].dt.strftime("%Y-%m-%d") +
                    " → " +
                    df_monthly["MonthEnd"].dt.strftime("%Y-%m-%d")
            )

            # ==== Chart multi-line Monthly (Actual + Models) ====
            st.subheader("MONTHLY (Actual + Models) – 3 tháng gần nhất")

            frames_m = [
                df_monthly[["MonthStart", "MonthlyActual"]]
                .rename(columns={"MonthlyActual": "MonthlyValue"})
                .assign(Source="Actual")
            ]

            for m_name in ["GRU", "RNN", "LSTM", "ARIMA", "SARIMA"]:
                col_name = f"Monthly_{m_name}"
                if col_name in df_monthly.columns and df_monthly[col_name].notna().any():
                    frames_m.append(
                        df_monthly[["MonthStart", col_name]]
                        .rename(columns={col_name: "MonthlyValue"})
                        .assign(Source=m_name)
                    )

            df_chart_m = pd.concat(frames_m, ignore_index=True).sort_values("MonthStart")

            chart_monthly = (
                alt.Chart(df_chart_m)
                .mark_line(point=True)
                .encode(
                    x=alt.X("MonthStart:T", title="Month (Start Date)"),
                    y=alt.Y("MonthlyValue:Q", title="Vehicles / month"),
                    color=alt.Color("Source:N"),
                    tooltip=[
                        alt.Tooltip("MonthStart:T", title="Month Start"),
                        alt.Tooltip("Source:N", title="Series"),
                        alt.Tooltip("MonthlyValue:Q", format=","),
                    ],
                )
                .properties(height=300)
            )

            st.altair_chart(chart_monthly, use_container_width=True)

            with st.expander("Xem bảng Monthly (Actual + Models)"):
                df_monthly_show = df_monthly.copy()
                for c in df_monthly_show.columns:
                    if c.startswith("Monthly"):
                        df_monthly_show[c] = df_monthly_show[c].round().astype("Int64").apply(lambda x: f"{x:,.0f}")
                st.dataframe(
                    df_monthly_show[
                        ["MonthRange"] +
                        [c for c in df_monthly_show.columns if c not in
                         ["Date", "MonthStart", "MonthEnd", "MonthRange"]]
                        ],
                    use_container_width=True
                )

            # ==== Metrics tổng Weekly cho từng model (nếu có cột) ====
            metrics_rows = []

            for m_name in ["GRU", "RNN", "LSTM", "ARIMA", "SARIMA"]:
                col_name = f"Monthly_{m_name}"
                if col_name not in df_monthly.columns:
                    continue

                valid = df_monthly[["MonthlyActual", col_name]].dropna()
                if valid.empty:
                    continue

                actual = valid["MonthlyActual"].values.astype(float)
                pred = valid[col_name].values.astype(float)

                mse = mean_squared_error(actual, pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(actual, pred)
                mape = np.mean(np.abs((actual - pred) / np.where(actual == 0, np.nan, actual))) * 100

                denom = np.abs(actual) + np.abs(pred)
                smape = np.mean(2 * np.abs(pred - actual) / np.where(denom == 0, 1, denom)) * 100

                r2 = r2_score(actual, pred)

                metrics_rows.append({
                    "Model": m_name,
                    "MSE": mse,
                    "RMSE": rmse,
                    "MAE": mae,
                    "MAPE (%)": mape,
                    "SMAPE (%)": smape,
                    "R²": r2
                })

            if metrics_rows:
                st.subheader("Đánh giá sai số theo từng model – dữ liệu Monthly")
                df_metrics_monthly = pd.DataFrame(metrics_rows)
                for c in ["MSE", "RMSE", "MAE"]:
                    df_metrics_monthly[c] = df_metrics_monthly[c].round(2)
                for c in ["MAPE (%)", "SMAPE (%)", "R²"]:
                    df_metrics_monthly[c] = df_metrics_monthly[c].round(3)

                # ---- Format số theo dạng 000,000,000.00 ----
                format_cols = ["MSE", "RMSE", "MAE", "MAPE (%)", "SMAPE (%)", "R²"]
                df_formatted_monthly = df_metrics_monthly.copy()
                for c in ["MSE", "RMSE", "MAE"]:
                    df_formatted_monthly[c] = df_formatted_monthly[c].apply(lambda x: f"{x:,.2f}")
                for c in ["MAPE (%)", "SMAPE (%)", "R²"]:
                    df_formatted_monthly[c] = df_formatted_monthly[c].apply(lambda x: f"{x:,.3f}")

                st.dataframe(df_formatted_monthly, use_container_width=True)

            # ==== Biểu đồ cột cho từng đánh giá ====
            st.subheader("📊 Biểu đồ cột cho từng đánh giá sai số")
            metrics_list = ["MSE", "RMSE", "MAE", "MAPE (%)", "SMAPE (%)", "R²"]
            cols = st.columns(2)

            for i, metric in enumerate(metrics_list):
                chart = (
                    alt.Chart(df_metrics_monthly)
                    .mark_bar()
                    .encode(
                        x=alt.X("Model:N", title="Model", axis=alt.Axis(labelAngle=0)),
                        y=alt.Y(f"{metric}:Q", title=metric),
                        tooltip=["Model", metric]
                    )
                    .properties(
                        height=300,
                        title=alt.TitleParams(
                            f"{metric}",
                            fontSize=24,
                            fontWeight="bold",
                            color="#333",
                            anchor="middle"  # căn giữa
                        )
                    )
                )

                cols[i % 2].altair_chart(chart, use_container_width=True)

                if i % 2 == 1 and i < len(metrics_list) - 1:
                    cols = st.columns(2)


if __name__ == "__main__":
    main()
