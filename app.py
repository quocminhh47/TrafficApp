import os
import glob
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf

# ---------------- Page config ----------------
st.set_page_config(
    page_title="Traffic Congestion Prediction",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://www.extremelycoolapp.com/help",
        "Report a bug": "https://www.extremelycoolapp.com/bug",
        "About": "# This is a header. This is an *extremely* cool app!",
    },
)

DATASET_ROOT = "data/processed_ds"

# ---------------- Folder-based helpers (không phụ thuộc hive) ----------------
@st.cache_data(show_spinner=False)
def list_cities() -> list[str]:
    if not os.path.isdir(DATASET_ROOT):
        return []
    cities = [
        d
        for d in os.listdir(DATASET_ROOT)
        if os.path.isdir(os.path.join(DATASET_ROOT, d)) and not d.startswith(".")
    ]
    return sorted(cities)

@st.cache_data(show_spinner=False)
def list_zones(city: str) -> list[str]:
    base = os.path.join(DATASET_ROOT, city)
    if not os.path.isdir(base):
        return []
    zones = [
        d
        for d in os.listdir(base)
        if os.path.isdir(os.path.join(base, d)) and not d.startswith(".")
    ]
    return sorted(zones)

def _files_for(city: str, zone: str | None) -> list[str]:
    if not city:
        return []
    pattern = (
        os.path.join(DATASET_ROOT, city, "**", "*.parquet")
        if not zone or zone == "(All)"
        else os.path.join(DATASET_ROOT, city, zone, "**", "*.parquet")
    )
    return sorted(glob.glob(pattern, recursive=True))

@st.cache_data(show_spinner=False)
def list_routes(city: str, zone: str | None) -> list[str]:
    files = _files_for(city, zone)
    if not files:
        return []
    # Đọc nhẹ chỉ cột RouteId; file có thể thiếu RouteId -> skip an toàn
    frames = []
    for f in files:
        try:
            s = pd.read_parquet(f, columns=["RouteId"])
            frames.append(s)
        except Exception:
            pass
    if not frames:
        return []
    routes = pd.concat(frames, ignore_index=True)["RouteId"].dropna().astype(str).unique().tolist()
    return sorted(routes)

@st.cache_data(show_spinner=False)
def load_slice(
    city: str,
    zone: str | None,
    routes: list[str] | None,
    start_dt: datetime | None,
    end_dt: datetime | None,
) -> pd.DataFrame:
    """
    Đọc parquet theo đường dẫn thư mục /City/Zone/*.parquet.
    Nếu file thiếu City/ZoneName, tự gán từ tham số.
    """
    files = _files_for(city, zone)
    if not files:
        return pd.DataFrame(columns=["DateTime","City","ZoneName","RouteId","Vehicles"])

    frames = []
    for f in files:
        try:
            df = pd.read_parquet(f)  # đọc full vì schema có thể khác nhau
            # Bắt buộc có DateTime & Vehicles
            if "DateTime" not in df.columns or "Vehicles" not in df.columns:
                continue
            # Chuẩn hoá thời gian (có thể đã có tz hoặc không)
            df["DateTime"] = pd.to_datetime(df["DateTime"], utc=True, errors="coerce")
            df = df.dropna(subset=["DateTime","Vehicles"])

            # Bổ sung City/ZoneName nếu thiếu trong file
            if "City" not in df.columns:
                df["City"] = city
            if "ZoneName" not in df.columns:
                df["ZoneName"] = zone if zone and zone != "(All)" else ""
            if "RouteId" in df.columns:
                df["RouteId"] = df["RouteId"].astype(str)
            else:
                df["RouteId"] = f"{city}-ALL"

            frames.append(df[["DateTime","City","ZoneName","RouteId","Vehicles"]])
        except Exception:
            # file lỗi schema -> bỏ qua
            pass

    if not frames:
        return pd.DataFrame(columns=["DateTime","City","ZoneName","RouteId","Vehicles"])

    df = pd.concat(frames, ignore_index=True)

    # Filter thời gian
    if start_dt is not None:
        df = df[df["DateTime"] >= pd.Timestamp(start_dt, tz="UTC")]
    if end_dt is not None:
        df = df[df["DateTime"] < pd.Timestamp(end_dt, tz="UTC")]

    # Filter route
    if routes:
        df = df[df["RouteId"].isin(list(map(str, routes)))]

    # Bỏ tz cho vẽ nhanh
    df["DateTime"] = df["DateTime"].dt.tz_convert(None)

    # Ép category để group/plot nhanh
    for c in ["City","ZoneName","RouteId"]:
        df[c] = df[c].astype("category")

    # Thêm cột hiển thị
    if not df.empty:
        dt = df["DateTime"]
        df["Year"] = dt.dt.year
        df["Month"] = dt.dt.month
        df["Date"] = dt.dt.day
        df["Hour"] = dt.dt.hour
        df["Day"] = dt.dt.strftime("%A")
        df["DayOfWeek"] = dt.dt.dayofweek
        df["HourOfDay"] = dt.dt.hour

    return df

# ---------------- Load model (Keras .h5 + sklearn preprocessor) ----------------
try:
    keras_model = tf.keras.models.load_model("./model/traffic_mlp.h5")
    preprocessor = joblib.load("./model/encoder.pkl")
except Exception as e:
    st.error(f"Error loading Keras model/preprocessor: {e}")
    st.stop()

# ---------------- Dữ liệu hợp nhất (chỉ dùng cho tab Report) ----------------
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    df = pd.read_parquet(
        "data/processed/all_cities_hourly.parquet",
        columns=["DateTime","City","ZoneName","RouteId","Vehicles"],
    )
    df["DateTime"] = pd.to_datetime(df["DateTime"], utc=True).dt.tz_convert(None)
    for c in ["City","ZoneName","RouteId"]:
        df[c] = df[c].astype("category")
    dt = df["DateTime"]
    df["Year"] = dt.dt.year
    df["Month"] = dt.dt.month
    df["Date"] = dt.dt.day
    df["Hour"] = dt.dt.hour
    df["Day"] = dt.dt.strftime("%A")
    df["DayOfWeek"] = dt.dt.dayofweek
    df["HourOfDay"] = dt.dt.hour
    return df

Traffic_prediction = load_data()

# ---------------- Feature builders & predict ----------------
def build_rich_feature_df(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index if isinstance(df, pd.DataFrame) else None)

    if "DateTime" in df:
        dt = pd.to_datetime(df["DateTime"])
        out["Year"] = dt.dt.year
        out["Month"] = dt.dt.month
        out["Date"] = dt.dt.day
        out["Hour"] = dt.dt.hour
        out["DayOfWeek"] = dt.dt.weekday
        out["HourOfDay"] = dt.dt.hour
        out["Day"] = dt.dt.day_name()
    else:
        for c in ["Year","Month","Date","Hour","DayOfWeek","HourOfDay"]:
            out[c] = pd.to_numeric(df.get(c, 0), errors="coerce")
        if "Day" in df:
            out["Day"] = df["Day"].astype(str)
        else:
            dow = pd.to_numeric(out.get("DayOfWeek", 0), errors="coerce").fillna(0).astype(int) % 7
            day_map_rev = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            out["Day"] = [day_map_rev[i] for i in dow]

    out["City"] = df.get("City", "Berlin")
    out["ZoneName"] = df.get("ZoneName", "")
    out["RouteId"] = df["RouteId"].astype(str) if "RouteId" in df else "Zone0"
    out["ID"] = pd.to_numeric(df.get("ID", 0), errors="coerce")

    for c in out.columns:
        if out[c].dtype != object:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def predict_any(df_like: pd.DataFrame) -> np.ndarray:
    X_rich = build_rich_feature_df(df_like)
    needed = ["City","ZoneName","RouteId","Year","Month","Date","Hour","DayOfWeek","HourOfDay","Day"]
    for c in needed:
        if c not in X_rich.columns:
            X_rich[c] = "" if c in ["City","ZoneName","RouteId","Day"] else 0
    X_infer = X_rich[needed].copy()
    X_arr = preprocessor.transform(X_infer)
    yhat = keras_model.predict(X_arr, verbose=0).reshape(-1)
    return yhat

# ---------------- Main UI ----------------
def main():
    st.title("Traffic Congestion Prediction")

    with st.sidebar:
        st.title("Menu")

        if st.button("🔄 Reload data"):
            list_cities.clear()
            list_zones.clear()
            list_routes.clear()
            load_slice.clear()
            st.experimental_rerun()

        menu = st.radio("Menu", ["Home", "Forecast", "Report", "Dashboard"])

        # Global filters (áp dụng cho Forecast/Dashboard)
        cities = list_cities()
        if not cities:
            st.error("Không tìm thấy dataset trong data/processed_ds.")
            st.stop()
        city = st.selectbox("City", options=cities, index=0)

        zones = ["(All)"] + list_zones(city)
        zone = st.selectbox("Zone/Area", options=zones, index=0)

        routes = list_routes(city, None if zone == "(All)" else zone)
        default_routes = routes[: min(10, len(routes))]
        route_ids = st.multiselect("Route(s)", options=routes, default=default_routes)

        today = pd.Timestamp.utcnow().normalize().to_pydatetime()
        start_date = st.date_input("Start date", value=(today - timedelta(days=14)).date())
        end_date = st.date_input("End date (exclusive)", value=today.date())

    if menu == "Home":
        st.subheader("Home")
        st.write("Welcome to the Traffic Congestion Prediction App.")
        st.image("./img/AI-in-transportation.webp", caption="Traffic Congestion")

    elif menu == "Forecast":
        st.subheader("Hourly Forecast for Next Day")

        # Context (optional)
        df_ctx = load_slice(
            city=city,
            zone=None if zone == "(All)" else zone,
            routes=route_ids if route_ids else None,
            start_dt=datetime.combine(start_date, datetime.min.time()),
            end_dt=datetime.combine(end_date, datetime.min.time()),
        )

        selected_date = st.date_input("Base date (forecast next day from this date)")
        next_day = pd.to_datetime(selected_date) + pd.Timedelta(days=1)
        hours = pd.date_range(next_day.normalize(), periods=24, freq="H")

        frames = []
        for rid in (route_ids or []):
            if not df_ctx.empty and "ZoneName" in df_ctx.columns:
                sub = df_ctx.loc[df_ctx["RouteId"] == rid]
                zn = sub["ZoneName"].iloc[0] if not sub.empty else (zone if zone != "(All)" else "")
            else:
                zn = zone if zone != "(All)" else ""
            frames.append(pd.DataFrame({
                "City": city,
                "ZoneName": zn,
                "RouteId": rid,
                "DateTime": hours
            }))

        if not frames:
            st.info("No route selected.")
            st.stop()

        df_next_all = pd.concat(frames, ignore_index=True)

        # Dự đoán
        yhat = predict_any(df_next_all)
        forecast_df = df_next_all.copy()
        forecast_df["PredictedVehicles"] = pd.Series(yhat, dtype="float64")

        # Biểu đồ
        chart = (
            alt.Chart(forecast_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("DateTime:T", title="Time"),
                y=alt.Y("PredictedVehicles:Q", title="Predicted Vehicles"),
                color=alt.Color("RouteId:N", title="Route"),
                tooltip=[
                    "City",
                    "ZoneName",
                    "RouteId",
                    "DateTime:T",
                    alt.Tooltip("PredictedVehicles:Q", format=".0f"),
                ],
            )
            .interactive()
        )
        st.altair_chart(chart, use_container_width=True)

        summary = (
            forecast_df.groupby(["City","ZoneName","RouteId"], as_index=False)["PredictedVehicles"]
            .agg(["min","max","mean"]).round(1).reset_index()
        )
        st.write("Summary (Vehicles):")
        st.dataframe(summary)

    elif menu == "Report":
        st.subheader("Model Report")
        st.image("./img/4.jpg", caption="Report")

        y_test = pd.to_numeric(Traffic_prediction["Vehicles"], errors="coerce").fillna(0.0).values
        y_pred = predict_any(Traffic_prediction)
        y_pred = pd.to_numeric(pd.Series(y_pred), errors="coerce").fillna(0.0).values

        report_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
        st.write(report_df.head(10))

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        st.write(f"Mean Squared Error: {mse:.4f}")
        st.write(f"R-squared: {r2:.4f}")

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_test, y=y_pred)
        plt.xlabel("Actual Vehicles")
        plt.ylabel("Predicted Vehicles")
        plt.title("Actual vs Predicted Vehicles")
        st.pyplot(plt)

    elif menu == "Dashboard":
        st.subheader("Dashboard")

        df_ctx = load_slice(
            city=city,
            zone=None if zone == "(All)" else zone,
            routes=route_ids if route_ids else None,
            start_dt=datetime.combine(start_date, datetime.min.time()),
            end_dt=datetime.combine(end_date, datetime.min.time()),
        )
        if df_ctx.empty:
            st.info("No data for selected filters.")
            st.stop()

        st.image("./img/3.jpeg", caption="Dashboard")

        st.write("### Traffic Volume Over Time")
        plt.figure(figsize=(12, 6))
        sns.lineplot(x="DateTime", y="Vehicles", hue="ZoneName", data=df_ctx)
        plt.title("Traffic Volume Over Time")
        plt.ylabel("Vehicles / Metric")
        plt.xlabel("DateTime")
        plt.xticks(rotation=45)
        plt.legend(title="Zone")
        st.pyplot(plt)

        st.write("### Traffic Volume by Zone/Area")
        plt.figure(figsize=(12, 6))
        x_col = "ZoneName" if "ZoneName" in df_ctx.columns else ("RouteId" if "RouteId" in df_ctx.columns else None)
        if x_col:
            sns.violinplot(x=x_col, y="Vehicles", data=df_ctx)
            plt.title(f"Traffic Volume by {x_col}")
            plt.xlabel(x_col)
            plt.ylabel("Vehicles / Metric")
            st.pyplot(plt)

        st.write("### Average Traffic Volume by Hour of Day")
        df_ctx["HourOfDay"] = df_ctx["DateTime"].dt.hour
        hourly = df_ctx.groupby("HourOfDay")["Vehicles"].mean().reset_index()
        plt.figure(figsize=(12, 6))
        sns.barplot(x="HourOfDay", y="Vehicles", data=hourly)
        plt.title("Average by Hour of Day")
        plt.xlabel("Hour")
        plt.ylabel("Average")
        plt.xticks(range(0, 24))
        st.pyplot(plt)

        st.write("### Average Traffic Volume by Day of Week")
        df_ctx["DayName"] = df_ctx["DateTime"].dt.day_name()
        weekly = (
            df_ctx.groupby("DayName")["Vehicles"].mean()
            .reindex(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
            .reset_index()
        )
        plt.figure(figsize=(12, 6))
        sns.barplot(x="DayName", y="Vehicles", data=weekly)
        plt.title("Average by Day of Week")
        plt.xlabel("Day")
        plt.ylabel("Average")
        st.pyplot(plt)

if __name__ == "__main__":
    main()
