#!/usr/bin/env python
import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
from pathlib import Path
import json

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from modules.data_loader import load_slice, list_cities, list_zones, list_routes
from modules.geo_routes import load_routes_geo
from map_component import map_routes  # custom map component

from modules.model_utils import (
    forecast_gru,
    forecast_rnn,
    forecast_week_after_last_point,
)
from modules.model_manager import load_model_context
from modules.arima_utils import forecast_arima_for_day  # ARIMA helper


# ======================================================
# HELPER: Forecast 24h cho 1 ngày cụ thể (GRU / RNN)
# (dùng cho Compare + ARIMA)
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
    Forecast 24h cho 1 ngày cụ thể (00:00 -> 23:00) bằng GRU hoặc RNN,
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
    else:
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
    Đọc file <route_id>_top2_last_year.json nếu có.
    Trả về dict hoặc None.
    """
    model_dir = Path("model") / family_name
    summary_path = model_dir / f"{route_id}_top2_last_year.json"
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
    st.set_page_config(page_title="Traffic Forecast App", layout="wide")

    st.title("🚦 Traffic Forecast App – I94 + Fremont + Multi-Model")

    # --------------------------------------------------
    # Apply pending selection từ map (trước khi tạo widget)
    # --------------------------------------------------
    if "pending_city" in st.session_state:
        st.session_state["city"] = st.session_state["pending_city"]
        del st.session_state["pending_city"]

    if "pending_zone" in st.session_state:
        st.session_state["zone"] = st.session_state["pending_zone"]
        del st.session_state["pending_zone"]

    if "pending_route" in st.session_state:
        st.session_state["route_id"] = st.session_state["pending_route"]
        del st.session_state["pending_route"]

    # ====================================
    # 1) SIDEBAR: CITY / ZONE / ROUTE / MODEL
    # ====================================
    cities = list_cities()
    if not cities:
        st.error("Không tìm thấy city nào trong data/processed_ds.")
        return

    # --- Init session_state mặc định ---
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
    # 3) LOAD MODEL CONTEXT (FALLBACK NẾU city=Seattle, zone=(All))
    # ====================================
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
    # 4) ROUTE + SEQ MODEL CHOICE (FORECAST)
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
    # Xác định top-2 model cho route hiện tại (nếu có)
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

    # Radio chọn sequence model dùng cho phần Compare (tab GRU/RNN)
    seq_model_type = st.sidebar.radio(
        "Sequence model (Forecast/Compare)",
        ["GRU", "RNN"],
        index=0,
        key="seq_model_type",
    )
    if seq_model_type == "RNN" and MODEL_RNN is None:
        st.sidebar.warning("Không tìm thấy mô hình RNN cho city/zone này, sẽ dùng GRU.")
        seq_model_type = "GRU"

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

    if clicked_route_id is not None and clicked_route_id != route_id:
        row = routes_geo_all[routes_geo_all["route_id"] == clicked_route_id]
        if not row.empty:
            clicked_city = row.iloc[0]["city"]
            clicked_zone = row.iloc[0]["zone"]

            st.session_state["pending_city"] = clicked_city
            st.session_state["pending_zone"] = clicked_zone
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
    # 7) FORECAST SECTION (WEEK AFTER LAST POINT)
    # ====================================
    st.header("🔮 Forecast – tuần kế tiếp sau dữ liệu gốc (NO SHIFT)")

    st.caption(
        "Forecast dùng ensemble các model top-2 (nếu có), ví dụ GRU + RNN. "
        "Đường vẽ là giá trị trung bình, tooltip hiển thị từng model."
    )

    # ================= Forecast tuần bằng ensemble top-2 =================
    dfs_for_ensemble = []

    for m_name in top_models:
        if m_name not in ("GRU", "RNN"):
            continue  # hiện tại chỉ ensemble GRU/RNN

        df_m, anchor_m = forecast_week_after_last_point(
            route_id=route_id,
            city=city,
            zone=None if zone == "(All)" else zone,
            ctx=ctx,
            n_days=7,
            model_type=m_name,  # "GRU" hoặc "RNN"
        )
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
        # Chuẩn hoá về format ensemble (1 model)
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
        # Giả sử anchor_day giống nhau, lấy anchor đầu tiên
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
            # Tính ensemble (trung bình các model đã dùng)
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

                    # Đánh dấu giờ cao điểm: >= 80th percentile của ensemble
                    if df_day["PredictedVehicles"].notna().sum() > 0:
                        q80 = df_day["PredictedVehicles"].quantile(0.8)
                        df_day["IsPeak"] = df_day["PredictedVehicles"] >= q80
                    else:
                        df_day["IsPeak"] = False

                    st.markdown(
                        f"**Ngày gốc (theo data):** {day_start.strftime('%Y-%m-%d')}  "
                        f"| Anchor (ngày cuối data thật): {anchor_day_raw.date()}  "
                        f"| Ensemble models: {', '.join(top_models)}"
                    )

                    # Build tooltip động tuỳ theo model nào đang có
                    tooltip_fields = [
                        alt.Tooltip(
                            "DateTime:T",
                            title="Thời gian",
                            format="%Y-%m-%d %H:%M",
                        )
                    ]

                    if "Pred_GRU" in df_day.columns:
                        tooltip_fields.append(
                            alt.Tooltip("Pred_GRU:Q", title="GRU", format=".0f")
                        )
                    if "Pred_RNN" in df_day.columns:
                        tooltip_fields.append(
                            alt.Tooltip("Pred_RNN:Q", title="RNN", format=".0f")
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
    # 8) DAILY TRAFFIC – 1 NĂM GẦN NHẤT (RAW)
    # ====================================
    st.subheader("📊 Daily traffic volume – 1 năm gần nhất (Actual)")

    df_full_route = df_full.copy()
    if not df_full_route.empty:
        max_dt_norm = df_full_route["DateTime"].max().normalize()
        start_last_year = max_dt_norm - pd.Timedelta(days=365)

        df_last_year = df_full_route[
            df_full_route["DateTime"] >= start_last_year
        ].copy()
        df_last_year["Date"] = df_last_year["DateTime"].dt.normalize()

        df_daily = (
            df_last_year.groupby("Date", as_index=False)["Vehicles"]
            .sum()
            .rename(columns={"Vehicles": "DailyVehicles"})
        )

        chart_daily = (
            alt.Chart(df_daily)
            .mark_line(point=True)
            .encode(
                x="Date:T",
                y="DailyVehicles:Q",
                tooltip=["Date:T", "DailyVehicles:Q"],
            )
            .properties(height=250)
        )
        st.altair_chart(chart_daily, use_container_width=True)

    # ====================================
    # 9) SO SÁNH MODEL (GRU/RNN/ARIMA) – 1 NGÀY
    # ====================================
    st.header("📊 Compare – GRU / RNN / ARIMA vs Actual (1 ngày)")

    min_actual_date = df_full["DateTime"].min().normalize().date()
    max_actual_date = df_full["DateTime"].max().normalize().date()

    report_date = pd.to_datetime(
        st.date_input(
            "Report date (dùng chung cho GRU, RNN & ARIMA)",
            value=max_actual_date,
            min_value=min_actual_date,
            max_value=max_actual_date,
            key="cmp_report_date_seq",
        )
    )

    day_start = report_date.normalize()
    day_end = day_start + pd.Timedelta(days=1)

    # ---- ACTUAL ----
    df_actual = load_slice(
        city=city,
        zone=None if zone == "(All)" else zone,
        routes=[route_id],
        start_dt=day_start,
        end_dt=day_end,
    )

    if df_actual.empty:
        st.warning(
            f"⚠️ Không tìm thấy actual trong parquet cho ngày {report_date.date()}."
        )
        return

    df_actual = df_actual.copy()
    df_actual["DateTime"] = pd.to_datetime(df_actual["DateTime"], errors="coerce")
    df_actual = df_actual.dropna(subset=["DateTime"])

    df_actual_hourly = (
        df_actual.set_index("DateTime")["Vehicles"].resample("1H").mean().dropna()
    )
    df_actual_hourly = df_actual_hourly.reset_index().rename(
        columns={"Vehicles": "Actual"}
    )

    # ---- PREDICT GRU & RNN ----
    preds_by_model = {}
    merged_by_model = {}

    if MODEL_GRU is not None:
        df_gru_day, _ = forecast_one_day(
            route_id=route_id,
            forecast_date=day_start,
            city=city,
            zone=zone,
            ctx=ctx,
            seq_model_type="GRU",
        )
        if not df_gru_day.empty:
            df_gru_day = df_gru_day.copy()
            df_gru_day["DateTime"] = pd.to_datetime(
                df_gru_day["DateTime"], errors="coerce"
            )
            df_gru_day = df_gru_day.dropna(subset=["DateTime"])
            df_gru_day = df_gru_day.rename(
                columns={"PredictedVehicles": "PredictedVehicles_GRU"}
            )
            m = df_actual_hourly.merge(df_gru_day, on="DateTime", how="left")
            m = m.dropna(subset=["PredictedVehicles_GRU"])
            merged_by_model["GRU"] = m
            preds_by_model["GRU"] = m[["DateTime", "PredictedVehicles_GRU"]].rename(
                columns={"PredictedVehicles_GRU": "Pred_GRU"}
            )

    if MODEL_RNN is not None:
        df_rnn_day, _ = forecast_one_day(
            route_id=route_id,
            forecast_date=day_start,
            city=city,
            zone=zone,
            ctx=ctx,
            seq_model_type="RNN",
        )
        if not df_rnn_day.empty:
            df_rnn_day = df_rnn_day.copy()
            df_rnn_day["DateTime"] = pd.to_datetime(
                df_rnn_day["DateTime"], errors="coerce"
            )
            df_rnn_day = df_rnn_day.dropna(subset=["DateTime"])
            df_rnn_day = df_rnn_day.rename(
                columns={"PredictedVehicles": "PredictedVehicles_RNN"}
            )
            m = df_actual_hourly.merge(df_rnn_day, on="DateTime", how="left")
            m = m.dropna(subset=["PredictedVehicles_RNN"])
            merged_by_model["RNN"] = m
            preds_by_model["RNN"] = m[["DateTime", "PredictedVehicles_RNN"]].rename(
                columns={"PredictedVehicles_RNN": "Pred_RNN"}
            )

    # ---- ARIMA ----
    # forecast_arima_for_day(df_full, day_start, day_end, value_col="Vehicles", order=(1,0,1))
    df_arima, arima_model_used = forecast_arima_for_day(
        df_full=df_full,
        day_start=day_start,
        day_end=day_end,
        value_col="Vehicles",
    )

    if df_arima is not None and not df_arima.empty:
        df_arima = df_arima.copy()

        # Đảm bảo có DateTime
        if "DateTime" not in df_arima.columns and df_arima.index.name == "DateTime":
            df_arima = df_arima.reset_index()

        if "DateTime" in df_arima.columns:
            df_arima["DateTime"] = pd.to_datetime(
                df_arima["DateTime"], errors="coerce"
            )
            df_arima = df_arima.dropna(subset=["DateTime"])
        else:
            st.warning("ARIMA: không tìm thấy cột 'DateTime', bỏ qua ARIMA.")
            df_arima = None

        # Đảm bảo có cột dự báo đúng tên 'Pred_ARIMA'
        if df_arima is not None:
            if "Pred_ARIMA" not in df_arima.columns:
                st.warning("ARIMA: không tìm thấy cột 'Pred_ARIMA', bỏ qua ARIMA.")
                df_arima = None

        if df_arima is not None:
            m = df_actual_hourly.merge(df_arima, on="DateTime", how="left")
            # Chỉ dropna nếu cột tồn tại
            if "Pred_ARIMA" in m.columns:
                m = m.dropna(subset=["Pred_ARIMA"])
                if not m.empty:
                    merged_by_model["ARIMA"] = m
                    preds_by_model["ARIMA"] = m[["DateTime", "Pred_ARIMA"]]


    if not merged_by_model:
        st.warning("Không có model nào forecast được cho ngày này.")
        return

    # Tabs cho từng model
    tabs_models = []
    tab_indices = {}
    model_names = list(merged_by_model.keys())
    for i, name in enumerate(model_names):
        tabs_models.append(name)
        tab_indices[name] = i

    model_tabs = st.tabs(tabs_models)

    # ========== TAB GRU ==========
    if "GRU" in merged_by_model:
        with model_tabs[tab_indices["GRU"]]:
            df_m = merged_by_model["GRU"]
            st.subheader("📌 GRU vs Actual")

            chart_gru = (
                alt.Chart(df_m)
                .transform_fold(
                    ["Actual", "PredictedVehicles_GRU"],
                    as_=["Series", "Value"],
                )
                .mark_line(point=True)
                .encode(
                    x=alt.X("DateTime:T", title="Thời gian"),
                    y=alt.Y("Value:Q", title="Vehicles"),
                    color=alt.Color(
                        "Series:N",
                        scale=alt.Scale(
                            domain=["Actual", "PredictedVehicles_GRU"],
                            range=["#000000", "#1f77b4"],
                        ),
                    ),
                    tooltip=[
                        alt.Tooltip(
                            "DateTime:T", title="Time", format="%Y-%m-%d %H:%M"
                        ),
                        alt.Tooltip("Actual:Q", title="Actual", format=".0f"),
                        alt.Tooltip(
                            "PredictedVehicles_GRU:Q", title="GRU", format=".0f"
                        ),
                    ],
                )
                .properties(height=300)
            )
            st.altair_chart(chart_gru, use_container_width=True)

            actual = df_m["Actual"].values
            pred = df_m["PredictedVehicles_GRU"].values

            mse = mean_squared_error(actual, pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(actual, pred)

            if np.any(actual != 0):
                mape = (
                    np.mean(
                        np.abs(
                            (actual - pred)[actual != 0] / actual[actual != 0]
                        )
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

            st.subheader("📌 Metrics – GRU vs Actual")
            st.write(f"**MSE:**   {mse:.2f}")
            st.write(f"**RMSE:**  {rmse:.2f}")
            st.write(f"**MAE:**   {mae:.2f}")
            st.write(f"**MAPE:**  {mape:.2f}%")
            st.write(f"**SMAPE:** {smape:.2f}%")
            st.write(f"**R²:**    {r2:.3f}")

    # ========== TAB RNN ==========
    if "RNN" in merged_by_model:
        with model_tabs[tab_indices["RNN"]]:
            df_m = merged_by_model["RNN"]
            st.subheader("📌 RNN vs Actual")

            chart_rnn = (
                alt.Chart(df_m)
                .transform_fold(
                    ["Actual", "PredictedVehicles_RNN"],
                    as_=["Series", "Value"],
                )
                .mark_line(point=True)
                .encode(
                    x=alt.X("DateTime:T", title="Thời gian"),
                    y=alt.Y("Value:Q", title="Vehicles"),
                    color=alt.Color(
                        "Series:N",
                        scale=alt.Scale(
                            domain=["Actual", "PredictedVehicles_RNN"],
                            range=["#000000", "#ff7f0e"],
                        ),
                    ),
                    tooltip=[
                        alt.Tooltip(
                            "DateTime:T", title="Time", format="%Y-%m-%d %H:%M"
                        ),
                        alt.Tooltip("Actual:Q", title="Actual", format=".0f"),
                        alt.Tooltip(
                            "PredictedVehicles_RNN:Q", title="RNN", format=".0f"
                        ),
                    ],
                )
                .properties(height=300)
            )
            st.altair_chart(chart_rnn, use_container_width=True)

            actual = df_m["Actual"].values
            pred = df_m["PredictedVehicles_RNN"].values

            mse = mean_squared_error(actual, pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(actual, pred)

            if np.any(actual != 0):
                mape = (
                    np.mean(
                        np.abs(
                            (actual - pred)[actual != 0] / actual[actual != 0]
                        )
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

            st.subheader("📌 Metrics – RNN vs Actual")
            st.write(f"**MSE:**   {mse:.2f}")
            st.write(f"**RMSE:**  {rmse:.2f}")
            st.write(f"**MAE:**   {mae:.2f}")
            st.write(f"**MAPE:**  {mape:.2f}%")
            st.write(f"**SMAPE:** {smape:.2f}%")
            st.write(f"**R²:**    {r2:.3f}")

    # ========== TAB ARIMA (nếu có) ==========
    if "ARIMA" in merged_by_model:
        with model_tabs[tab_indices["ARIMA"]]:
            df_m = merged_by_model["ARIMA"]
            st.subheader("📌 ARIMA vs Actual")

            chart_arima = (
                alt.Chart(df_m)
                .transform_fold(
                    ["Actual", "Pred_ARIMA"],
                    as_=["Series", "Value"],
                )
                .mark_line(point=True)
                .encode(
                    x=alt.X("DateTime:T", title="Thời gian"),
                    y=alt.Y("Value:Q", title="Vehicles"),
                    color=alt.Color(
                        "Series:N",
                        scale=alt.Scale(
                            domain=["Actual", "Pred_ARIMA"],
                            range=["#000000", "#2ca02c"],
                        ),
                    ),
                    tooltip=[
                        alt.Tooltip("DateTime:T", title="Time", format="%Y-%m-%d %H:%M"),
                        alt.Tooltip("Actual:Q", title="Actual", format=".0f"),
                        alt.Tooltip("Pred_ARIMA:Q", title="ARIMA", format=".0f"),
                    ],
                )
                .properties(height=300)
            )
            st.altair_chart(chart_arima, use_container_width=True)

            actual = df_m["Actual"].values
            pred = df_m["Pred_ARIMA"].values

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

            st.subheader("📌 Metrics – ARIMA vs Actual")
            st.write(f"**MSE:**   {mse:.2f}")
            st.write(f"**RMSE:**  {rmse:.2f}")
            st.write(f"**MAE:**   {mae:.2f}")
            st.write(f"**MAPE:**  {mape:.2f}%")
            st.write(f"**SMAPE:** {smape:.2f}%")
            st.write(f"**R²:**    {r2:.3f}")

    # ========== BẢNG TỔNG HỢP COMPARISON (Actual + các model) ==========
    st.subheader("📋 Tổng hợp Actual + GRU/RNN/ARIMA (nếu có)")

    df_compare = df_actual_hourly.rename(columns={"Actual": "Actual"}).copy()

    if "GRU" in preds_by_model:
        df_compare = df_compare.merge(
            preds_by_model["GRU"], on="DateTime", how="left"
        )
        df_compare["AbsErr_GRU"] = (df_compare["Pred_GRU"] - df_compare["Actual"]).abs()

    if "RNN" in preds_by_model:
        df_compare = df_compare.merge(
            preds_by_model["RNN"], on="DateTime", how="left"
        )
        df_compare["AbsErr_RNN"] = (df_compare["Pred_RNN"] - df_compare["Actual"]).abs()

    if "ARIMA" in preds_by_model:
        df_compare = df_compare.merge(
            preds_by_model["ARIMA"], on="DateTime", how="left"
        )
        df_compare["AbsErr_ARIMA"] = (
            df_compare["Pred_ARIMA"] - df_compare["Actual"]
        ).abs()

    st.dataframe(
        df_compare.sort_values("DateTime").reset_index(drop=True),
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
