#!/usr/bin/env python
import streamlit as st
import pandas as pd
import altair as alt
import tensorflow as tf
import pickle
import json
from pathlib import Path

from modules.data_loader import load_slice, list_cities, list_zones, list_routes
from modules.model_utils import forecast_gru, forecast_mlp


# ======================================================
# HELPER: Forecast 24h cho 1 ngày cụ thể
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
    """
    Forecast 24h cho đúng ngày forecast_date (00:00 → 24:00).

    Quy ước:
    - base_date = forecast_date (bắt đầu forecast từ 00:00 ngày đó)
    - GRU dùng history window = [base_date - LOOKBACK, base_date)
    - Nếu không đủ history / lỗi → fallback MLP
    - Nếu MLP lỗi → fallback Baseline
    """
    LOOKBACK = int(meta.get("LOOKBACK", 168))
    HORIZON = int(meta.get("HORIZON", 24))

    # Chuẩn hoá ngày dự đoán (00:00)
    forecast_date = pd.Timestamp(forecast_date).normalize()
    base_date = forecast_date  # base_date = chính ngày cần dự đoán

    # History window dùng cho GRU
    start_dt = base_date - pd.Timedelta(hours=LOOKBACK)
    end_dt = base_date

    # Lấy history từ parquet
    df_hist = load_slice(
        city=city,
        zone=None if zone == "(All)" else zone,
        routes=[route_id],
        start_dt=start_dt,
        end_dt=end_dt,
    )

    # Nếu không có history ⇒ bỏ GRU, dùng MLP/Baseline
    if df_hist.empty:
        df_fc, model_used = forecast_mlp(
            route_id=route_id,
            base_date=base_date,
            scaler=scaler,
            routes=routes_model,
            horizon=HORIZON,
        )
    else:
        # Có history ⇒ ưu tiên GRU, nếu GRU fail thì forecast_gru sẽ fallback MLP/Baseline
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

    # Nếu vì lý do gì đó vẫn không có forecast
    if df_fc is None or df_fc.empty:
        return pd.DataFrame(), model_used

    df_fc = df_fc.copy()
    df_fc["DateTime"] = pd.to_datetime(df_fc["DateTime"], errors="coerce")
    df_fc = df_fc.dropna(subset=["DateTime"])

    # Lọc đúng 24h của forecast_date (00:00 → 24:00 cùng ngày)
    next_day = forecast_date + pd.Timedelta(days=1)
    df_fc = df_fc[
        (df_fc["DateTime"] >= forecast_date)
        & (df_fc["DateTime"] < next_day)
    ]

    # Đánh dấu ngày / model (phục vụ UI)
    df_fc["ForecastDate"] = forecast_date.date()
    df_fc["Model"] = model_used

    return df_fc, model_used


def vn_weekday_label(dt: pd.Timestamp) -> str:
    """Trả về label kiểu 'Thứ 6 15/11' hoặc 'Chủ nhật 17/11'."""
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

    # ---------- Load model / meta / scaler ----------
    MODEL_PATH = Path("model/traffic_seq.keras")
    META_PATH = Path("model/seq_meta.json")
    SCALER_PATH = Path("model/vehicles_scaler.pkl")

    if not MODEL_PATH.exists():
        st.error("⚠️ Thiếu GRU model: model/traffic_seq.keras. Hãy train trước.")
        return
    MODEL = tf.keras.models.load_model(MODEL_PATH)

    if not SCALER_PATH.exists():
        st.error("⚠️ Thiếu scaler: model/vehicles_scaler.pkl. Hãy train trước.")
        return
    with open(SCALER_PATH, "rb") as f:
        SCALER = pickle.load(f)

    if not META_PATH.exists():
        st.error("⚠️ Thiếu meta: model/seq_meta.json. Hãy chạy prep_seq_light + train_seq trước.")
        return
    with open(META_PATH, "r") as f:
        META = json.load(f)

    ROUTES_MODEL = META.get("routes", [])
    if not ROUTES_MODEL:
        st.error("⚠️ seq_meta.json không có key 'routes'.")
        return

    ROUTES = ROUTES_MODEL
    RID2IDX = {r: i for i, r in enumerate(ROUTES)}
    LOOKBACK = int(META.get("LOOKBACK", 168))
    HORIZON = int(META.get("HORIZON", 24))

    # ---------- Sidebar: chọn data (city/zone/route) + tab ----------
    st.sidebar.title("🚦 Traffic App (Parquet)")

    cities = list_cities()
    if not cities:
        st.error("⚠️ Không tìm thấy thư mục data/processed_ds.")
        return
    city = st.sidebar.selectbox("City", cities)

    zones = list_zones(city)
    zone = st.sidebar.selectbox("Zone", zones)

    raw_routes = list_routes(city, None if zone == "(All)" else zone)
    # chỉ giữ route có trong model GRU
    routes = [r for r in raw_routes if r in ROUTES_MODEL]
    if not routes:
        st.error("⚠️ Không có RouteId nào trong parquet khớp với model.")
        return

    route_id = st.sidebar.selectbox("Route", ROUTES)

    # Đọc full data một lần để biết min/max date
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

    min_dt = df_full["DateTime"].min()
    max_dt = df_full["DateTime"].max()

    tab = st.sidebar.radio("Tab", ["Forecast", "Compare"])

    st.sidebar.markdown(
        f"**Data range (parquet):** {min_dt.date()} → {max_dt.date()}  \n"
        f"**Lookback:** {LOOKBACK}h  \n"
        f"**Horizon:** {HORIZON}h  \n"
        f"**Model routes:** {len(ROUTES_MODEL)}"
    )

    # ======================================================
    # TAB 1: FORECAST – Hôm nay (24h) + 7 ngày tới
    # ======================================================
    if tab == "Forecast":
        st.header("📈 Forecast: hôm nay (24h) + 7 ngày kế tiếp")

        now = pd.Timestamp.now().round("S")
        today = now.normalize()

        # === 1) Dự đoán FULL 24h của hôm nay ===
        st.subheader("📅 Hôm nay (24h forecast)")

        df_today_full, model_today = forecast_one_day(
            route_id=route_id,
            forecast_date=today,
            city=city,
            zone=zone,
            model=MODEL,
            meta=META,
            scaler=SCALER,
            routes_model=ROUTES_MODEL,
            rid2idx=RID2IDX,
        )

        if df_today_full.empty:
            st.warning("Không tạo được forecast 24h cho hôm nay (GRU/MLP/Baseline đều fail).")
        else:
            df_today_full = df_today_full.sort_values("DateTime")

            st.caption(f"Model used for today: **{model_today}**")

            chart_today = (
                alt.Chart(df_today_full)
                .mark_line(point=True)
                .encode(
                    x="DateTime:T",
                    y="PredictedVehicles:Q",
                    tooltip=["DateTime:T", "PredictedVehicles:Q"],
                )
                .properties(height=300, title=f"Today {today.date()} (24h)")
            )
            st.altair_chart(chart_today, use_container_width=True)

            st.write("Summary (hôm nay, 24h):")
            st.dataframe(df_today_full["PredictedVehicles"].describe().to_frame().T)

        # === 2) Dự đoán 7 ngày tiếp theo – MỖI NGÀY 1 TAB RIÊNG ===
        st.subheader("📅 7 ngày kế tiếp")

        num_days = 7
        day_results = []  # (label, df_day, model_used)

        for offset in range(1, num_days + 1):
            forecast_date = today + pd.Timedelta(days=offset)
            df_fc_day, model_used = forecast_one_day(
                route_id=route_id,
                forecast_date=forecast_date,
                city=city,
                zone=zone,
                model=MODEL,
                meta=META,
                scaler=SCALER,
                routes_model=ROUTES_MODEL,
                rid2idx=RID2IDX,
            )

            if df_fc_day.empty:
                continue

            label = vn_weekday_label(forecast_date)
            day_results.append((label, df_fc_day.sort_values("DateTime"), model_used))

        if not day_results:
            st.warning("❌ Không tạo được forecast cho 7 ngày tới (GRU/MLP/Baseline đều fail).")
        else:
            tab_labels = [lbl for (lbl, _, _) in day_results]
            tabs = st.tabs(tab_labels)

            for (tab_obj, (label, df_day, model_used)) in zip(tabs, day_results):
                with tab_obj:
                    st.markdown(f"### {label}  \nModel: **{model_used}**")

                    chart_day = (
                        alt.Chart(df_day)
                        .mark_line(point=True)
                        .encode(
                            x="DateTime:T",
                            y="PredictedVehicles:Q",
                            tooltip=["DateTime:T", "PredictedVehicles:Q"],
                        )
                        .properties(height=320, title=label)
                    )
                    st.altair_chart(chart_day, use_container_width=True)

                    st.write(f"Summary ({label}):")
                    st.dataframe(df_day["PredictedVehicles"].describe().to_frame().T)

    # ======================================================
    # TAB 2: COMPARE – MLP vs Actual & GRU vs Actual (chung 1 ngày)
    # ======================================================
    else:  # tab == "Compare"
        st.header("📊 Compare Predicted vs Actual")

        # --- Load toàn bộ lịch sử cho route để xác định khoảng ngày ---
        df_all = load_slice(
            city=city,
            zone=None if zone == "(All)" else zone,
            routes=[route_id],
            start_dt=None,
            end_dt=None,
        )

        if df_all.empty:
            st.warning("⚠️ Không có dữ liệu lịch sử trong parquet cho route đã chọn.")
            return

        df_all = df_all.copy()
        df_all["DateTime"] = pd.to_datetime(df_all["DateTime"], errors="coerce")
        df_all = df_all.dropna(subset=["DateTime"])

        min_dt = df_all["DateTime"].min().normalize()
        max_dt = df_all["DateTime"].max().normalize()

        if pd.isna(min_dt) or pd.isna(max_dt):
            st.warning("⚠️ Không xác định được min/max DateTime từ dữ liệu.")
            return

        HORIZON = int(META.get("HORIZON", 24))
        LOOKBACK = int(META.get("LOOKBACK", 168))

        # --- Chọn 1 ngày chung cho cả MLP & GRU ---
        min_actual_date = (min_dt + pd.Timedelta(days=1)).date()
        max_actual_date = max_dt.date()

        report_date = pd.to_datetime(
            st.date_input(
                "Report date (áp dụng cho cả MLP & GRU)",
                value=max_actual_date,
                min_value=min_actual_date,
                max_value=max_actual_date,
                key="cmp_report_date",
            )
        )

        day_start = report_date.normalize()
        day_end = day_start + pd.Timedelta(days=1)

        tab_cmp_mlp, tab_cmp_gru = st.tabs(["MLP vs Actual", "GRU vs Actual"])

        # ==================================================
        # 2.1 MLP vs Actual
        # ==================================================
        with tab_cmp_mlp:
            st.subheader("🧠 MLP vs Actual (per-hour)")

            # --- Actual từ parquet ---
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
                st.stop()

            df_actual = df_actual.copy()
            df_actual["DateTime"] = pd.to_datetime(df_actual["DateTime"], errors="coerce")
            df_actual = df_actual.dropna(subset=["DateTime"])

            # Resample về hourly mean
            df_actual = (
                df_actual.set_index("DateTime")["Vehicles"]
                .resample("1H")
                .mean()
                .dropna()
                .reset_index()
            )

            st.caption(
                f"[MLP] Actual date: {report_date.date()} | actual hourly rows = {len(df_actual)}"
            )

            # --- Forecast bằng MLP cho đúng ngày đó ---
            base_date_cmp = day_start  # dự báo cho chính ngày này

            df_fc_mlp, model_used_mlp = forecast_mlp(
                route_id=route_id,
                base_date=base_date_cmp,
                scaler=SCALER,
                routes=ROUTES,
                horizon=HORIZON,
            )

            if df_fc_mlp is None or df_fc_mlp.empty:
                st.error(
                    "❌ Không tạo được forecast bằng MLP (model/traffic_mlp.h5 hoặc scaler có thể lỗi)."
                )
                st.stop()

            df_fc_mlp = df_fc_mlp.copy()
            df_fc_mlp["DateTime"] = pd.to_datetime(df_fc_mlp["DateTime"], errors="coerce")
            df_fc_mlp = df_fc_mlp.dropna(subset=["DateTime"])

            # Lọc đúng ngày report_date
            df_fc_mlp = df_fc_mlp[
                (df_fc_mlp["DateTime"] >= day_start)
                & (df_fc_mlp["DateTime"] < day_end)
            ]

            if df_fc_mlp.empty:
                st.warning(
                    "⚠️ MLP forecast không có timestamp nào rơi đúng trong ngày report được chọn."
                )
                st.stop()

            # --- Merge actual vs predicted ---
            merged_mlp = pd.merge(
                df_actual,  # DateTime, Vehicles
                df_fc_mlp[["DateTime", "PredictedVehicles"]],
                on="DateTime",
                how="inner",
            )

            if merged_mlp.empty:
                st.warning(
                    "⚠️ Không có timestamp trùng giữa actual & MLP predicted trong ngày này."
                )
                st.stop()

            merged_mlp = merged_mlp.rename(
                columns={
                    "Vehicles": "Actual",
                    "PredictedVehicles": "Predicted",
                }
            )

            long_mlp = merged_mlp.melt(
                id_vars="DateTime",
                value_vars=["Actual", "Predicted"],
                var_name="Type",
                value_name="Value",
            )

            chart_mlp = (
                alt.Chart(long_mlp)
                .mark_line(point=True)
                .encode(
                    x="DateTime:T",
                    y="Value:Q",
                    color="Type:N",
                    tooltip=["DateTime:T", "Type:N", "Value:Q"],
                )
                .properties(height=400)
            )
            st.altair_chart(chart_mlp, use_container_width=True)

            # --- Bảng chi tiết từng giờ: DateTime - Actual - Predicted - |Error| ---
            merged_mlp["AbsError"] = (merged_mlp["Predicted"] - merged_mlp["Actual"]).abs()

            st.subheader("📋 Bảng chi tiết (MLP)")
            st.dataframe(
                merged_mlp[["DateTime", "Actual", "Predicted", "AbsError"]]
                .sort_values("DateTime")
                .reset_index(drop=True)
            )

            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

            mse = mean_squared_error(merged_mlp["Actual"], merged_mlp["Predicted"])
            mae = mean_absolute_error(merged_mlp["Actual"], merged_mlp["Predicted"])
            r2 = r2_score(merged_mlp["Actual"], merged_mlp["Predicted"])

            st.subheader("📌 Evaluation Metrics – MLP vs Actual (per-hour)")
            st.write(f"**MSE:** {mse:.2f}")
            st.write(f"**MAE:** {mae:.2f}")
            st.write(f"**R²:** {r2:.3f}")
            st.caption(
                f"Model used: **{model_used_mlp}** (expected: MLP)  \n"
                f"Report date: {report_date.date()}  \n"
                f"Forecast base_date: {base_date_cmp.date()} (MLP dự đoán cho chính ngày này)."
            )

        # ==================================================
        # 2.2 GRU vs Actual
        # ==================================================
        with tab_cmp_gru:
            st.subheader("📉 GRU vs Actual (per-hour, có thể fallback MLP/Baseline)")

            # --- Actual từ parquet ---
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
                st.stop()

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

            # --- Chuẩn bị history cho GRU: 168h trước day_start ---
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
                    f"⚠️ Không có đủ lịch sử ({LOOKBACK}h) trước ngày {report_date.date()} → nhiều khả năng GRU sẽ fallback."
                )

            # --- Forecast bằng GRU (có thể fallback MLP/Baseline) ---
            df_fc_gru, model_used_gru = forecast_gru(
                route_id=route_id,
                base_date=day_start,  # dự báo cho chính ngày report_date
                model=MODEL,
                meta=META,
                scaler=SCALER,
                routes_model=ROUTES,
                rid2idx=RID2IDX,
                df_hist=df_hist,
            )

            if df_fc_gru is None or df_fc_gru.empty:
                st.error("❌ GRU forecast trả về rỗng (có thể đã fallback & vẫn lỗi).")
                st.stop()

            df_fc_gru = df_fc_gru.copy()
            df_fc_gru["DateTime"] = pd.to_datetime(
                df_fc_gru["DateTime"], errors="coerce"
            )
            df_fc_gru = df_fc_gru.dropna(subset=["DateTime"])

            # Lọc đúng ngày report_date
            df_fc_gru = df_fc_gru[
                (df_fc_gru["DateTime"] >= day_start)
                & (df_fc_gru["DateTime"] < day_end)
            ]

            if df_fc_gru.empty:
                st.warning(
                    "⚠️ GRU/MLP/Baseline forecast không có timestamp nào rơi đúng trong ngày report được chọn."
                )
                st.stop()

            # --- Merge actual vs predicted ---
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
                st.stop()

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
            # --- Bảng chi tiết từng giờ: DateTime - Actual - Predicted - |Error| ---
            merged_gru["AbsError"] = (merged_gru["Predicted"] - merged_gru["Actual"]).abs()

            st.subheader("📋 Bảng chi tiết (GRU)")
            st.dataframe(
                merged_gru[["DateTime", "Actual", "Predicted", "AbsError"]]
                .sort_values("DateTime")
                .reset_index(drop=True)
            )

            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

            mse_g = mean_squared_error(merged_gru["Actual"], merged_gru["Predicted"])
            mae_g = mean_absolute_error(merged_gru["Actual"], merged_gru["Predicted"])
            r2_g = r2_score(merged_gru["Actual"], merged_gru["Predicted"])

            st.subheader("📌 Evaluation Metrics – GRU vs Actual (per-hour)")
            st.write(f"**MSE:** {mse_g:.2f}")
            st.write(f"**MAE:** {mae_g:.2f}")
            st.write(f"**R²:** {r2_g:.3f}")
            st.caption(
                f"Model used: **{model_used_gru}** (GRU hoặc fallback MLP/Baseline)  \n"
                f"Report date: {report_date.date()}  \n"
                f"GRU base_date: {day_start.date()} (dự đoán cho chính ngày này)."
            )


if __name__ == "__main__":
    main()
