#!/usr/bin/env python
import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
from pathlib import Path
import joblib
import json
import tensorflow as tf

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
)
from modules.model_manager import load_model_context

@st.cache_resource
def get_model_context(city: str, zone: str | None):
    """
    Cache ModelContext cho mỗi (city, zone) để tránh load model nhiều lần.
    """
    return load_model_context(city, zone)



# ======================================================
# LSTM artifacts loader (dùng chung cho forecast_one_day)
# ======================================================
# @lru_cache(maxsize=None)
# def load_lstm_artifacts(city: str, zone: str | None):
#     """
#     Load LSTM model theo city/zone giống flow GRU/RNN.
#
#     Tức là tìm trong:
#         model/<City>_<Zone>/
#         model/<City>/
#         model/
#
#     Và trả về:
#         { "model", "meta", "scaler", "routes", "rid2idx" }
#     hoặc None nếu không tìm thấy.
#     """
#     from pathlib import Path
#     import json
#     import joblib
#     from tensorflow.keras.models import load_model
#
#     base = Path("model")
#     city_str = (city or "").replace(" ", "_")
#     zone_str = (zone or "").replace(" ", "_") if zone else None
#
#     # Build danh sách folder theo thứ tự ưu tiên
#     candidates = []
#
#     if zone_str and zone_str != "(All)":
#         candidates.append(base / f"{city_str}_{zone_str}")
#
#     candidates.append(base / city_str)
#     candidates.append(base)
#
#     for d in candidates:
#         meta_path = d / "lstm_meta.json"
#         model_path = d / "traffic_lstm.keras"
#         scaler_path = d / "vehicles_scaler.pkl"
#
#         if meta_path.exists() and model_path.exists() and scaler_path.exists():
#             print(f"[LSTM] Using model dir: {d}")
#
#             meta = json.load(open(meta_path, "r"))
#             model = load_model(model_path)
#             scaler = joblib.load(scaler_path)
#
#             routes = list(meta.get("routes", []))
#             rid2idx = {rid: i for i, rid in enumerate(routes)}
#
#             return {
#                 "model": model,
#                 "meta": meta,
#                 "scaler": scaler,
#                 "routes": routes,
#                 "rid2idx": rid2idx,
#                 "dir": str(d),
#             }
#
#     print(f"[LSTM] No valid LSTM model found for city={city}, zone={zone}")
#     return None


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


# ======================================================
# MAIN APP
# ======================================================
def main():
    if "last_clicked_route_id" not in st.session_state:
        st.session_state["last_clicked_route_id"] = None

    st.set_page_config(page_title="Traffic Forecast App", layout="wide")
    st.title("🚦 Traffic Forecast App ")

    # --------------------------------------------------
    # Apply pending selection từ map (trước khi tạo widget)
    if "pending_city" in st.session_state:
        st.session_state["city"] = st.session_state.pop("pending_city")
    if "pending_zone" in st.session_state:
        st.session_state["zone"] = st.session_state.pop("pending_zone")
    if "pending_route" in st.session_state:
        st.session_state["route_id"] = st.session_state.pop("pending_route")

    # ====================================
    # 1) SIDEBAR: CITY / ZONE / ROUTE
    # ====================================
    cities = list_cities()
    if not cities:
        st.error("Không tìm thấy city nào trong data/processed_ds.")
        return

    if "city" not in st.session_state:
        st.session_state["city"] = cities[0]

    city = st.sidebar.selectbox(
        "City",
        cities,
        index=cities.index(st.session_state["city"])
        if st.session_state["city"] in cities
        else 0,
        key="city",
    )

    zones = list_zones(city)
    if "(All)" not in zones:
        zones = ["(All)"] + zones

    if "zone" not in st.session_state:
        st.session_state["zone"] = zones[0]

    zone = st.sidebar.selectbox(
        "Zone",
        zones,
        index=zones.index(st.session_state["zone"])
        if st.session_state["zone"] in zones
        else 0,
        key="zone",
    )

    # ====================================
    # 2) LOAD MODEL CONTEXT (FALLBACK zone='(All)')
    # ====================================
    zone_for_model = None if zone == "(All)" else zone

    try:
        ctx = get_model_context(city, None if zone == "(All)" else zone)
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

    # Tách context
    MODEL_GRU = ctx.gru_model
    MODEL_RNN = getattr(ctx, "rnn_model", None)
    META = ctx.meta
    SCALER = ctx.scaler
    ROUTES_MODEL = ctx.routes_model
    RID2IDX = ctx.rid2idx
    LOOKBACK = ctx.lookback
    HORIZON = ctx.horizon

    # ====================================
    # 3) ROUTE (sidebar)
    # ====================================
    raw_routes = list_routes(city, None if zone == "(All)" else zone)
    if not raw_routes:
        st.error("⚠️ Không tìm thấy RouteId nào trong parquet cho city/zone này.")
        return

    if "route_id" not in st.session_state:
        st.session_state["route_id"] = raw_routes[0]

    if st.session_state["route_id"] not in raw_routes:
        st.session_state["route_id"] = raw_routes[0]

    route_id = st.sidebar.selectbox(
        "Route",
        raw_routes,
        index=raw_routes.index(st.session_state["route_id"])
        if st.session_state["route_id"] in raw_routes
        else 0,
        key="route",
    )
    st.session_state["route_id"] = route_id
    route_id = st.session_state["route_id"]

    # ====================================
    # 4) TOP-2 MODELS (cho ensemble forecast)
    # ====================================
    summary_top2 = load_top2_summary(ctx.family_name, route_id)
    if summary_top2 and "top_models" in summary_top2:
        top_models = summary_top2["top_models"]
    else:
        # fallback: nếu không có summary, ưu tiên GRU rồi RNN
        top_models = []
        if ctx.gru_model is not None:
            top_models.append("GRU")
        if getattr(ctx, "rnn_model", None) is not None:
            top_models.append("RNN")

    if not top_models:
        top_models = ["GRU"]

    st.markdown(
        f"**Top models (last year, ensemble dùng cho Forecast):** "
        f"`{', '.join(top_models)}`"
    )

    # ====================================
    # 5) MAP COMPONENT
    # ====================================
    st.subheader("🗺 Routes Map")

    routes_geo_all = load_routes_geo().fillna("")

    df_geo_city = routes_geo_all[routes_geo_all["city"] == city].copy()
    routes_data = df_geo_city.to_dict("records")

    df_all_geo = routes_geo_all.dropna(subset=["lat", "lon"]).copy()
    all_routes_list = df_all_geo.to_dict("records")

    if df_geo_city.empty:
        st.info("Không có thông tin geo cho city hiện tại.")
        routes_data = []

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

            row = routes_geo_all[routes_geo_all["route_id"] == clicked_route_id]
            if not row.empty:
                st.session_state["pending_city"] = row.iloc[0]["city"]
                st.session_state["pending_zone"] = row.iloc[0]["zone"]
                st.session_state["pending_route"] = clicked_route_id
            else:
                st.session_state["pending_route"] = clicked_route_id

            st.rerun()

    st.write(f"**Đang chọn tuyến:** {route_id}")

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
        st.error("⚠️ Không đọc được dữ liệu history cho route này.")
        return

    df_full = df_full.copy()
    df_full["DateTime"] = pd.to_datetime(df_full["DateTime"], errors="coerce")
    df_full = df_full.dropna(subset=["DateTime"])

    min_dt = df_full["DateTime"].min()
    max_dt = df_full["DateTime"].max()

    st.sidebar.markdown(
        f"**Data range (parquet):** {min_dt.date()} → {max_dt.date()}  \n"
        f"**Lookback:** {LOOKBACK}h  \n"
        f"**Horizon:** {HORIZON}h  \n"
        f"**Model routes:** {len(ROUTES_MODEL)}"
    )

    # ====================================
    # 7) FORECAST – tuần kế tiếp sau dữ liệu gốc (ensemble GRU/RNN)
    # ====================================
    st.header("🔮 Forecast – tuần kế tiếp sau dữ liệu gốc (NO SHIFT)")

    st.caption(
        "Forecast dùng ensemble các model top-2 (nếu có), ví dụ GRU + RNN. "
        "Đường vẽ là giá trị trung bình, tooltip hiển thị từng model."
    )

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
                n_days=7,
                model_type=m_name,
            )
        elif m_name == "LSTM":
            # NEW: forecast tuần bằng LSTM riêng
            df_m, anchor_m = forecast_week_after_last_point_lstm(
                route_id=route_id,
                city=city,
                zone=None if zone == "(All)" else zone,
                ctx=ctx,
                n_days=7,
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
            n_days=7,
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

            df_merge["PredictedVehicles"] = df_merge["Pred_ENSEMBLE"]
            df_fc_raw = df_merge.copy()
        else:
            df_fc_raw = None

    if df_fc_raw is None or df_fc_raw.empty:
        st.warning("Không forecast được (thiếu dữ liệu history).")
    else:
        df_fc = df_fc_raw.copy()
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

                    # peak_threshold = float(s.quantile(0.9))
                    # peak_hours_mask = s >= peak_threshold
                    # peak_hours_list = [
                    #     idx.strftime("%H:%M") for idx, v in s[peak_hours_mask].items()
                    # ]

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
                            f"{avg_val:,.0f} vehicles/h",
                        )
                    #
                    # if peak_hours_list:
                    #     st.markdown(
                    #         "**Các khung giờ cao điểm (≥ 90th percentile):** "
                    #         + ", ".join(peak_hours_list)
                    #     )

                    # Bảng ngang
                    st.markdown("### 🧮 Lưu lượng theo giờ")

                    s_label = s.copy()
                    s_label.index = s_label.index.strftime("%H:%M")
                    tbl = pd.DataFrame([s_label.values], columns=s_label.index)
                    tbl.index = ["Vehicles/h"]
                    st.dataframe(tbl, use_container_width=True)

                    st.markdown(
                        f"**Ensemble models:** {', '.join(top_models)}"
                    )

                    # Tooltip hiển thị từng model nếu có
                    tooltip_fields = [
                        alt.Tooltip("DateTime:T", title="Thời gian"),
                        alt.Tooltip(
                            "Pred_ENSEMBLE:Q",
                            title="Dự báo ensemble",
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

                    tooltip_fields.append(
                        alt.Tooltip(
                            "PredictedVehicles:Q",
                            title="Ensemble (avg)",
                            format=".0f",
                        )
                    )

                    base = alt.Chart(df_day).encode(
                        x=alt.X("DateTime:T", title="Thời gian")
                    )

                    line = base.mark_line().encode(
                        y=alt.Y("PredictedVehicles:Q", title="Vehicles"),
                        tooltip=tooltip_fields,
                    )

                    points = base.mark_point(size=70).encode(
                        y="PredictedVehicles:Q",
                        color=alt.condition(
                            alt.datum.IsPeak,
                            alt.value("red"),
                            alt.value("steelblue"),
                        ),
                        tooltip=tooltip_fields,
                    )

                    chart = (line + points).interactive().properties(
                        height=320,
                        title=f"Dự báo cho {vn_weekday_label(day_start)}",
                    )
                    st.altair_chart(chart, use_container_width=True)

                    st.write(
                        "Min / Max / Mean (Ensemble):",
                        float(df_day["PredictedVehicles"].min()),
                        "/",
                        float(df_day["PredictedVehicles"].max()),
                        "/",
                        float(df_day["PredictedVehicles"].mean()),
                    )
        else:
            st.info("Không có ngày nào trong forecast.")

    # ====================================
    # 8) DAILY TRAFFIC – 3 THÁNG GẦN NHẤT
    #     Actual vs GRU / RNN / LSTM + Metrics tổng 3 tháng
    # ====================================
    st.header("📊 Daily traffic – 3 tháng gần nhất (Actual vs Models)")

    df_full_route = df_full.copy()
    if not df_full_route.empty:
        max_dt_norm = df_full_route["DateTime"].max().normalize()
        start_3m = max_dt_norm - pd.Timedelta(days=90)

        df_last = df_full_route[df_full_route["DateTime"] >= start_3m].copy()
        df_last["Date"] = df_last["DateTime"].dt.normalize()

        # ==== Actual daily ====
        df_daily_actual = (
            df_last.groupby("Date", as_index=False)["Vehicles"]
            .sum()
            .rename(columns={"Vehicles": "DailyActual"})
            .sort_values("Date")
        )

        if df_daily_actual.empty:
            st.info("Không có dữ liệu trong 3 tháng gần nhất.")
        else:
            dates = df_daily_actual["Date"].tolist()
            df_eval = df_daily_actual.set_index("Date")

            # ==== Helper loop dự đoán daily cho 1 model ====
            def _compute_daily_for_model(model_name: str):
                if model_name == "GRU" and MODEL_GRU is None:
                    return None
                if model_name == "RNN" and MODEL_RNN is None:
                    return None
                # LSTM không cần check model ở ctx, vì forecast_one_day sẽ tự fallback
                records = []
                for d in dates:
                    df_fc_day, _used = forecast_one_day(
                        route_id=route_id,
                        forecast_date=d,
                        city=city,
                        zone=zone,
                        ctx=ctx,
                        seq_model_type=model_name,
                    )
                    if df_fc_day is None or df_fc_day.empty:
                        continue
                    v = df_fc_day["PredictedVehicles"].sum()
                    records.append({"Date": d, "DailyPred": float(v)})
                if not records:
                    return None
                df_model = (
                    pd.DataFrame(records)
                    .groupby("Date", as_index=False)["DailyPred"]
                    .mean()
                    .rename(columns={"DailyPred": f"Daily_{model_name}"})
                )
                return df_model.set_index("Date")

            # GRU
            daily_pred_cols = []
            for m_name in ["GRU", "RNN", "LSTM"]:
                df_m = _compute_daily_for_model(m_name)
                if df_m is not None:
                    df_eval = df_eval.join(df_m, how="left")
                    daily_pred_cols.append(m_name)

            df_eval = df_eval.reset_index()

            # ==== Metrics tổng 3 tháng cho từng model ====
            metrics_rows = []
            for m_name in ["GRU", "RNN", "LSTM"]:
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
                st.subheader("📌 Metrics tổng 3 tháng (Daily)")
                df_metrics = pd.DataFrame(metrics_rows)
                # Làm đẹp số
                for c in ["MSE", "RMSE", "MAE"]:
                    df_metrics[c] = df_metrics[c].round(2)
                for c in ["MAPE (%)", "SMAPE (%)", "R²"]:
                    df_metrics[c] = df_metrics[c].round(3)
                st.dataframe(df_metrics, use_container_width=True)

            # ==== Chart multi-line (Actual + Models) ====
            frames = [
                df_eval[["Date", "DailyActual"]]
                .rename(columns={"DailyActual": "DailyValue"})
                .assign(Source="Actual")
            ]

            for m_name in ["GRU", "RNN", "LSTM"]:
                col_name = f"Daily_{m_name}"
                if col_name in df_eval.columns and df_eval[col_name].notna().any():
                    frames.append(
                        df_eval[["Date", col_name]]
                        .rename(columns={col_name: "DailyValue"})
                        .assign(Source=m_name)
                    )

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

            with st.expander("🔍 Xem bảng daily (Actual + Models) – 3 tháng gần nhất"):
                df_show = df_eval.copy()
                # cast về int cho dễ nhìn
                for c in df_show.columns:
                    if c.startswith("Daily"):
                        df_show[c] = df_show[c].round().astype("Int64")
                st.dataframe(df_show.sort_values("Date"), use_container_width=True)


if __name__ == "__main__":
    main()
