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
    # if "pending_city" in st.session_state:
    #     st.session_state["city"] = st.session_state.pop("pending_city")
    # if "pending_zone" in st.session_state:
    #     st.session_state["zone"] = st.session_state.pop("pending_zone")
    # if "pending_route" in st.session_state:
    #     st.session_state["route_id"] = st.session_state.pop("pending_route")

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
        # Chưa chọn city → disable zone
        zone = st.sidebar.selectbox(
            "Zone",
            ["(Chọn city trước)"],
            key="zone",
            disabled=True,
        )
        current_zone = None
    else:
        zones = list_zones(current_city)

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
            key="zone",
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
        zone_for_model = None if zone == "(All)" else zone

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

        # Tách context khi đã load được ctx
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
    else:
        # Chưa chọn city/route → chưa show gì, chỉ map + message "chọn route"
        pass

    # ----- OPTIONS -----
    tab = st.sidebar.radio("Options", ["FORECAST", "METRICS AND EVALUATION"])

    # ====================================
    # 5) MAP COMPONENT
    # ====================================
    st.subheader("🗺 Routes Map")

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
        st.write(f"**Đang chọn tuyến:** {route_id}")
    else:
        st.write("**Chưa chọn tuyến nào**")

    # nếu chưa có route thì chỉ show map, không load data/model
    if not route_id:
        st.info(" Hãy chọn một tuyến ở sidebar hoặc click vào marker trên bản đồ để xem forecast chi tiết.")
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
        st.error("⚠️ Không đọc được dữ liệu history cho route này.")
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
        st.header(" Dự đoán lưu lượng giao thông cho 7 ngày tới")

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
                        st.markdown("### 🧮 Lưu lượng xe cộ theo giờ")

                        # s: Series index = DateTime, value = Vehicles/h (ensemble)
                        s_label = s.copy()
                        s_label.index = s_label.index.strftime("%H:%M")
                        s_label_int = s_label.round(0).astype("Int64")  # convert to int, nullable

                        # 1 dòng, các cột là giờ
                        tbl = s_label_int.to_frame().T
                        tbl.index = ["Vehicles/h"]

                        styled_tbl = (
                            tbl.style
                            .format("{:,.0f}", na_rep="-")  # hiển thị int, có phân cách
                            .background_gradient(axis=1, cmap="YlOrRd")  # thấp = vàng nhạt, cao = đỏ
                            .highlight_max(axis=1, color="#7f0000   ")  # giờ cao điểm nhất tô đỏ hẳn
                        )

                        st.dataframe(styled_tbl, use_container_width=True, height=70)
                        # st.dataframe(styled_tbl, use_container_width=True, height=140)

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

                        st.markdown(f"**Ensemble models:** {', '.join(top_models)}")

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

                        chart = (line + points).interactive().properties(
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
        tab_cmp_daily, tab_cmp_weekly, tab_cmp_monthly = st.tabs(["Daily", "Weekly", "Monthly"])

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
                        "🔍 Xem bảng daily (Actual + Models) – 3 tháng gần nhất"
                ):
                    df_show = df_eval.copy()
                    for c in df_show.columns:
                        if c.startswith("Daily"):
                            df_show[c] = df_show[c].round().astype("Int64").apply(lambda x: f"{x:,.0f}")
                    st.dataframe(df_show.sort_values("Date"), use_container_width=True)
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
