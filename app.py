#!/usr/bin/env python
import streamlit as st
import pandas as pd
import altair as alt
import numpy as np

from modules.data_loader import load_slice, list_cities, list_zones, list_routes
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from modules.geo_routes import load_routes_geo
from map_component import map_routes  # custom map component

from modules.model_utils import (
    forecast_gru,
    forecast_week_after_last_point,
)
from modules.model_manager import load_model_context


# ======================================================
# HELPER: Forecast 24h cho 1 ngày cụ thể (GRU + fallback nội bộ)
# (giữ lại nếu cần dùng sau, hiện tại không gọi trực tiếp)
# ======================================================
def forecast_one_day(
    route_id,
    forecast_date: pd.Timestamp,
    city,
    zone,
    model,
    meta,
    scaler,
    routes_model,
    rid2idx,
):
    LOOKBACK = int(meta.get("LOOKBACK", 168))

    # Chuẩn hoá ngày dự đoán (00:00)
    forecast_date = pd.Timestamp(forecast_date).normalize()
    base_date = forecast_date

    # History window dùng cho GRU
    start_dt = base_date - pd.Timedelta(hours=LOOKBACK)
    end_dt = base_date

    df_hist = load_slice(
        city=city,
        zone=None if zone == "(All)" else zone,
        routes=[route_id],
        start_dt=start_dt,
        end_dt=end_dt,
    )

    df_fc, model_used = forecast_gru(
        route_id=route_id,
        base_date=base_date,
        model=model,
        meta=meta,
        scaler=scaler,
        routes_model=routes_model,
        rid2idx=rid2idx,
        df_hist=df_hist,
    )

    if df_fc is None or df_fc.empty:
        return pd.DataFrame(), model_used

    df_fc = df_fc.copy()
    df_fc["DateTime"] = pd.to_datetime(df_fc["DateTime"], errors="coerce")
    df_fc = df_fc.dropna(subset=["DateTime"])

    next_day = forecast_date + pd.Timedelta(days=1)
    df_fc = df_fc[
        (df_fc["DateTime"] >= forecast_date)
        & (df_fc["DateTime"] < next_day)
    ]

    df_fc["ForecastDate"] = forecast_date.date()
    df_fc["Model"] = model_used
    return df_fc, model_used


def vn_weekday_label(dt: pd.Timestamp) -> str:
    dt = pd.Timestamp(dt)
    wd = dt.weekday()  # 0=Mon ... 6=Sun
    if wd == 6:
        thu = "Chủ nhật"
    else:
        thu = f"Thứ {wd + 2}"
    return f"{thu} {dt.strftime('%d/%m')}"


# ======================================================
# MAIN APP
# ======================================================
def main():
    st.set_page_config(page_title="Traffic Forecast (Parquet only)", layout="wide")
    st.sidebar.title("🚦 Traffic App (Parquet)")

    # ====================================
    # 1) APPLY PENDING STATE FROM MAP (TRƯỚC KHI TẠO WIDGET)
    # ====================================
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
    # 2) SIDEBAR: CITY / ZONE / ROUTE (STATE SẠCH, KHÔNG INDEX)
    # ====================================

    # ----- CITY -----
    cities = list_cities()
    if not cities:
        st.error("⚠️ Không tìm thấy thư mục data/processed_ds.")
        return

    if "city" not in st.session_state:
        st.session_state["city"] = cities[0]

    city = st.sidebar.selectbox(
        "City",
        options=cities,
        key="city",  # không truyền index, dùng session_state["city"]
    )

    # ----- ZONE -----
    zones = list_zones(city)
    if not zones:
        st.error("⚠️ Không tìm thấy zone nào cho city này.")
        return

    # nếu zone cũ không còn trong list mới → reset
    if "zone" not in st.session_state or st.session_state["zone"] not in zones:
        st.session_state["zone"] = zones[0]

    zone = st.sidebar.selectbox(
        "Zone",
        options=zones,
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
            # Không có model tổng cho city → thử từng zone con
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

    MODEL = ctx.gru_model
    META = ctx.meta
    SCALER = ctx.scaler
    ROUTES_MODEL = ctx.routes_model
    RID2IDX = ctx.rid2idx
    LOOKBACK = ctx.lookback
    HORIZON = ctx.horizon
    ROUTES = ROUTES_MODEL

    # ====================================
    # 4) ROUTE SELECTBOX
    # ====================================
    raw_routes = list_routes(city, None if zone == "(All)" else zone)
    if not raw_routes:
        st.error("⚠️ Không tìm thấy RouteId nào trong parquet cho city/zone này.")
        return

    if "route_id" not in st.session_state or st.session_state["route_id"] not in raw_routes:
        st.session_state["route_id"] = raw_routes[0]

    st.sidebar.selectbox(
        "Route",
        options=raw_routes,
        key="route_id",
    )

    route_id = st.session_state["route_id"]

    # ====================================
    # 5) MAP COMPONENT
    # ====================================
    st.subheader("🗺 Routes Map")

    routes_geo_all = load_routes_geo().fillna("")

    df_geo_city = routes_geo_all[routes_geo_all["city"] == city].copy()
    if df_geo_city.empty:
        st.info("Không có thông tin geo cho city hiện tại.")
        routes_data = []
    else:
        routes_data = df_geo_city.to_dict("records")

    df_all = routes_geo_all.dropna(subset=["lat", "lon"])
    all_routes_list = df_all.to_dict("records")

    clicked_route_id = map_routes(
        routes_data=routes_data,
        selected_route_id=route_id,
        all_routes=all_routes_list,
        key="traffic_map",
    )

    if clicked_route_id is not None and clicked_route_id != route_id:
        # Tìm city/zone tương ứng route vừa click
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
    # 6) LOAD FULL DATA (CHO ROUTE HIỆN TẠI)
    # ====================================
    df_full = load_slice(
        city=city,
        zone=None if zone == "(All)" else zone,
        routes=[route_id],
        start_dt=None,
        end_dt=None,
    )
    if df_full.empty:
        st.error("⚠️ Không có data nào trong parquet cho city/zone/route này.")
        return

    df_full = df_full.copy()
    df_full["DateTime"] = pd.to_datetime(df_full["DateTime"], errors="coerce")
    df_full = df_full.dropna(subset=["DateTime"])

    min_dt = df_full["DateTime"].min()
    max_dt = df_full["DateTime"].max()

    tab = st.sidebar.radio("Tab", ["Forecast", "Compare (GRU vs Actual)"])

    st.sidebar.markdown(
        f"**Data range (parquet):** {min_dt.date()} → {max_dt.date()}  \n"
        f"**Lookback:** {LOOKBACK}h  \n"
        f"**Horizon:** {HORIZON}h  \n"
        f"**Model routes:** {len(ROUTES_MODEL)}"
    )

    # ======================================================
    # TAB 1: FORECAST – DÙNG TRỰC TIẾP WEEK SAU CỦA DATA (KHÔNG SHIFT SANG 2025)
    # ======================================================
    if tab == "Forecast":
        st.subheader("🔮 Forecast – tuần kế tiếp sau dữ liệu gốc (NO SHIFT)")

        df_fc_raw, anchor_day_raw = forecast_week_after_last_point(
            route_id=route_id,
            city=city,
            zone=None if zone == "(All)" else zone,
            ctx=ctx,
            n_days=7,
        )

        if df_fc_raw is None or df_fc_raw.empty:
            st.warning("Không forecast được (thiếu dữ liệu history).")
        else:
            df_fc = df_fc_raw.copy()
            df_fc["DateTime"] = pd.to_datetime(df_fc["DateTime"], errors="coerce")
            df_fc = df_fc.dropna(subset=["DateTime"])

            # Lấy list các ngày (normalize) trong forecast
            days = (
                df_fc["DateTime"]
                .dt.normalize()
                .drop_duplicates()
                .sort_values()
                .tolist()
            )

            if not days:
                st.info("Không có ngày nào trong forecast.")
                return

            day_tabs = st.tabs(
                [vn_weekday_label(d) for d in days]
            )

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

                    st.markdown(
                        f"**Ngày gốc (theo data):** {day_start.strftime('%Y-%m-%d')} "
                        f"| Anchor (ngày cuối data thật): {anchor_day_raw.date()}"
                    )

                    chart = (
                        alt.Chart(df_day)
                        .mark_line(point=True)
                        .encode(
                            x=alt.X("DateTime:T", title="Thời gian"),
                            y=alt.Y("PredictedVehicles:Q", title="Vehicles"),
                            tooltip=[
                                alt.Tooltip(
                                    "DateTime:T",
                                    title="Thời gian",
                                    format="%Y-%m-%d %H:%M",
                                ),
                                alt.Tooltip(
                                    "PredictedVehicles:Q",
                                    title="Vehicles",
                                    format=".0f",
                                ),
                            ],
                        )
                        .interactive()
                        .properties(
                            height=320,
                            title=f"Dự báo cho {vn_weekday_label(day_start)} (theo trục thời gian gốc)",
                        )
                    )

                    st.altair_chart(chart, use_container_width=True)

                    st.write(
                        "Min / Max / Mean:",
                        float(df_day["PredictedVehicles"].min()),
                        "/",
                        float(df_day["PredictedVehicles"].max()),
                        "/",
                        float(df_day["PredictedVehicles"].mean()),
                    )

    # ======================================================
    # TAB 2: COMPARE – GRU vs Actual
    # ======================================================
    else:
        st.header("📊 Compare GRU Predicted vs Actual")

        df_all = df_full

        min_dt = df_all["DateTime"].min().normalize()
        max_dt = df_all["DateTime"].max().normalize()

        if pd.isna(min_dt) or pd.isna(max_dt):
            st.warning("⚠️ Không xác định được min/max DateTime từ dữ liệu.")
            return

        HORIZON = int(META.get("HORIZON", 24))
        LOOKBACK = int(META.get("LOOKBACK", 168))

        min_actual_date = (min_dt + pd.Timedelta(days=1)).date()
        max_actual_date = max_dt.date()

        report_date = pd.to_datetime(
            st.date_input(
                "Report date",
                value=max_actual_date,
                min_value=min_actual_date,
                max_value=max_actual_date,
                key="cmp_report_date_gru",
            )
        )

        day_start = report_date.normalize()
        day_end = day_start + pd.Timedelta(days=1)

        st.subheader("📉 GRU vs Actual (per-hour)")

        df_actual_g = load_slice(
            city=city,
            zone=None if zone == "(All)" else zone,
            routes=[route_id],
            start_dt=day_start,
            end_dt=day_end,
        )

        if df_actual_g.empty:
            st.warning(
                f"⚠️ Không tìm thấy actual trong parquet cho ngày {report_date.date()}."
            )
            return

        df_actual_g = df_actual_g.copy()
        df_actual_g["DateTime"] = pd.to_datetime(
            df_actual_g["DateTime"], errors="coerce"
        )
        df_actual_g = df_actual_g.dropna(subset=["DateTime"])

        df_actual_g = (
            df_actual_g.set_index("DateTime")["Vehicles"]
            .resample("1H")
            .mean()
            .dropna()
            .reset_index()
        )

        st.caption(
            f"[GRU] Actual date: {report_date.date()} | actual hourly rows = {len(df_actual_g)}"
        )

        hist_start = day_start - pd.Timedelta(hours=LOOKBACK)
        hist_end = day_start

        df_hist = load_slice(
            city=city,
            zone=None if zone == "(All)" else zone,
            routes=[route_id],
            start_dt=hist_start,
            end_dt=hist_end,
        )

        if df_hist.empty:
            st.warning(
                f"⚠️ Không có đủ lịch sử ({LOOKBACK}h) trước ngày {report_date.date()} → GRU có thể fallback."
            )

        df_fc_gru, model_used_gru = forecast_gru(
            route_id=route_id,
            base_date=day_start,
            model=MODEL,
            meta=META,
            scaler=SCALER,
            routes_model=ROUTES,
            rid2idx=RID2IDX,
            df_hist=df_hist,
        )

        if df_fc_gru is None or df_fc_gru.empty:
            st.error("❌ GRU forecast trả về rỗng (có thể đã fallback & vẫn lỗi).")
            return

        df_fc_gru = df_fc_gru.copy()
        df_fc_gru["DateTime"] = pd.to_datetime(
            df_fc_gru["DateTime"], errors="coerce"
        )
        df_fc_gru = df_fc_gru.dropna(subset=["DateTime"])

        df_fc_gru = df_fc_gru[
            (df_fc_gru["DateTime"] >= day_start)
            & (df_fc_gru["DateTime"] < day_end)
        ]

        if df_fc_gru.empty:
            st.warning(
                "⚠️ GRU forecast không có timestamp nào rơi đúng trong ngày report được chọn."
            )
            return

        merged_gru = pd.merge(
            df_actual_g,
            df_fc_gru[["DateTime", "PredictedVehicles"]],
            on="DateTime",
            how="inner",
        )

        if merged_gru.empty:
            st.warning(
                "⚠️ Không có timestamp trùng giữa actual & predicted trong ngày này."
            )
            return

        merged_gru = merged_gru.rename(
            columns={
                "Vehicles": "Actual",
                "PredictedVehicles": "Predicted",
            }
        )

        long_gru = merged_gru.melt(
            id_vars="DateTime",
            value_vars=["Actual", "Predicted"],
            var_name="Type",
            value_name="Value",
        )

        chart_gru = (
            alt.Chart(long_gru)
            .mark_line(point=True)
            .encode(
                x="DateTime:T",
                y="Value:Q",
                color="Type:N",
                tooltip=["DateTime:T", "Type:N", "Value:Q"],
            )
            .properties(height=400)
        )
        st.altair_chart(chart_gru, use_container_width=True)

        actual = merged_gru["Actual"].values.astype(float)
        pred = merged_gru["Predicted"].values.astype(float)

        mse_g = mean_squared_error(actual, pred)
        mae_g = mean_absolute_error(actual, pred)
        r2_g = r2_score(actual, pred)
        rmse_g = np.sqrt(mse_g)

        mask_nonzero = actual != 0
        if mask_nonzero.any():
            mape_g = (
                np.mean(
                    np.abs(
                        (actual[mask_nonzero] - pred[mask_nonzero])
                        / actual[mask_nonzero]
                    )
                )
                * 100.0
            )
        else:
            mape_g = np.nan

        denom = np.abs(actual) + np.abs(pred)
        smape_g = (
            np.mean(
                2.0 * np.abs(pred - actual) / np.where(denom == 0, 1.0, denom)
            )
            * 100.0
        )

        st.subheader("📌 Evaluation Metrics – GRU vs Actual (per-hour)")
        st.write(f"**MSE:**   {mse_g:.2f}")
        st.write(f"**RMSE:**  {rmse_g:.2f}")
        st.write(f"**MAE:**   {mae_g:.2f}")
        st.write(f"**MAPE:**  {mape_g:.2f}%")
        st.write(f"**SMAPE:** {smape_g:.2f}%")
        st.write(f"**R²:**    {r2_g:.3f}")

        st.caption(
            f"Model used: **{model_used_gru}**  \n"
            f"Report date: {report_date.date()}  \n"
            f"Samples: {len(merged_gru)} (per-hour)"
        )

        merged_gru["AbsError"] = (merged_gru["Predicted"] - merged_gru["Actual"]).abs()

        st.subheader("📋 Bảng chi tiết (GRU)")
        st.dataframe(
            merged_gru[["DateTime", "Actual", "Predicted", "AbsError"]]
            .sort_values("DateTime")
            .reset_index(drop=True)
        )


if __name__ == "__main__":
    main()
