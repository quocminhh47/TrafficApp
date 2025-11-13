#!/usr/bin/env python
# app_light.py — Streamlit Traffic Forecast App (GRU + Fallback)
import os, glob, json, pickle
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import tensorflow as tf

# ---------------- CONFIG ----------------
st.set_page_config(page_title="🚦 Traffic Forecast (GRU + Fallback)", layout="wide")

DATA_ROOT = Path("data/processed_ds")
MODEL_DIR = Path("model")


# ---------------- HELPERS ----------------
@st.cache_resource
def load_models():
    """Load trained GRU + metadata + scaler"""
    model = tf.keras.models.load_model(MODEL_DIR / "traffic_seq.keras")
    with open(MODEL_DIR / "seq_meta.json") as f:
        meta = json.load(f)
    with open(MODEL_DIR / "vehicles_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    routes = meta["routes"]
    rid2idx = {r: i for i, r in enumerate(routes)}
    return model, meta, scaler, routes, rid2idx


def time_feats(dt: pd.Series):
    """Encode hour/day-of-week cyclical features"""
    dt = pd.to_datetime(dt)
    hour = dt.dt.hour
    dow = dt.dt.dayofweek
    return np.c_[np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
                 np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)].astype(np.float32)


def _files_for(city, zone=None):
    if zone in (None, "(All)"):
        pat = DATA_ROOT / city / "**" / "*.parquet"
    else:
        pat = DATA_ROOT / city / zone / "**" / "*.parquet"
    return sorted(glob.glob(str(pat), recursive=True))


@st.cache_data
def list_cities():
    if not DATA_ROOT.exists():
        return []
    return sorted([d for d in os.listdir(DATA_ROOT) if (DATA_ROOT / d).is_dir()])


@st.cache_data
def list_zones(city):
    base = DATA_ROOT / city
    if not base.is_dir():
        return []
    return ["(All)"] + sorted([d for d in os.listdir(base) if (base / d).is_dir()])


@st.cache_data
def list_routes(city, zone=None):
    files = _files_for(city, zone)
    frames = []
    for f in files:
        try:
            frames.append(pd.read_parquet(f, columns=["RouteId"]))
        except:
            pass
    if not frames:
        return []
    return sorted(pd.concat(frames)["RouteId"].dropna().astype(str).unique().tolist())


def load_slice(city, zone, route_id, start_dt, end_dt):
    """Load subset of data by route and date range"""
    files = _files_for(city, zone)
    frames = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            if {"DateTime", "Vehicles", "RouteId"} <= set(df.columns):
                df["DateTime"] = pd.to_datetime(df["DateTime"], utc=True, errors="coerce")
                df["Vehicles"] = pd.to_numeric(df["Vehicles"], errors="coerce")
                df = df.dropna(subset=["DateTime", "Vehicles"])
                df = df[df["RouteId"].astype(str) == str(route_id)]
                frames.append(df[["DateTime", "Vehicles", "RouteId"]])
        except Exception as e:
            print(f"[WARN] Skip {f}: {e}")
    if not frames:
        return pd.DataFrame(columns=["DateTime", "Vehicles", "RouteId"])

    df = pd.concat(frames, ignore_index=True)
    df["DateTime"] = pd.to_datetime(df["DateTime"], utc=True, errors="coerce")
    df = df.dropna(subset=["DateTime", "Vehicles"])

    if start_dt:
        start_dt = pd.Timestamp(start_dt, tz="UTC")
        df = df[df["DateTime"] >= start_dt]
    if end_dt:
        end_dt = pd.Timestamp(end_dt, tz="UTC")
        df = df[df["DateTime"] < end_dt]

    return df.sort_values("DateTime").reset_index(drop=True)


# ---------------- FORECAST MODELS ----------------
def forecast_baseline(route_id, base_date):
    """Simple seasonal baseline"""
    st.info(f"📉 Fallback: Seasonal baseline (sin-wave pattern).")
    next_hours = pd.date_range(pd.Timestamp(base_date) + pd.Timedelta(days=1), periods=24, freq="h")
    y = 100 + 20 * np.sin(np.linspace(0, 2 * np.pi, 24))
    return pd.DataFrame({"DateTime": next_hours, "Predicted": y, "RouteId": route_id}), "Baseline"


def forecast_mlp(route_id, base_date, scaler, routes):
    """Fallback to MLP if GRU unavailable"""
    try:
        mlp = tf.keras.models.load_model(MODEL_DIR / "traffic_mlp.h5")
        enc = pickle.load(open(MODEL_DIR / "encoder.pkl", "rb"))
        st.info(f"⚙️ Using fallback model: MLP for route {route_id}.")
    except Exception as e:
        st.warning(f"⚠️ Cannot load MLP ({e}), fallback baseline.")
        return forecast_baseline(route_id, base_date)

    next_hours = pd.date_range(pd.Timestamp(base_date) + pd.Timedelta(days=1), periods=24, freq="h")
    feats = time_feats(next_hours)
    onehot = np.zeros((24, len(routes)), dtype=np.float32)
    if route_id in routes:
        onehot[:, routes.index(route_id)] = 1.0
    X = np.concatenate([feats, onehot], axis=1)
    y_scaled = mlp.predict(X)
    y = scaler.inverse_transform(y_scaled.reshape(-1, 1)).ravel()

    df = pd.DataFrame({"DateTime": next_hours, "Predicted": y, "RouteId": route_id})
    return df, "MLP"


def forecast_gru(route_id, base_date, model, meta, scaler, routes, rid2idx, city, zone):
    """Forecast using GRU → fallback MLP → fallback baseline"""
    LB = meta.get("LOOKBACK", 168)
    HZ = meta.get("HORIZON", 24)

    end_dt = pd.Timestamp(base_date, tz="UTC") + pd.Timedelta(days=1)
    start_dt = end_dt - pd.Timedelta(hours=LB)

    hist = load_slice(city, zone, route_id, start_dt, end_dt)
    if hist.empty or len(hist) < LB:
        st.warning(f"⚠️ Route {route_id}: not enough data (<{LB}h) → fallback MLP.")
        return forecast_mlp(route_id, base_date, scaler, routes)

    # Clean numeric data
    hist["Vehicles"] = pd.to_numeric(hist["Vehicles"], errors="coerce")
    hist = hist.dropna(subset=["Vehicles", "DateTime"])

    # Resample hourly (numeric only)
    g = (
        hist.set_index("DateTime")
        .resample("1h")["Vehicles"]
        .mean()
        .dropna()
        .reset_index()
    )
    g["RouteId"] = route_id

    if len(g) < LB:
        st.warning(f"⚠️ Route {route_id}: only {len(g)}h → fallback MLP.")
        return forecast_mlp(route_id, base_date, scaler, routes)

    # Prepare GRU input
    try:
        v = scaler.transform(g[["Vehicles"]].tail(LB)).astype(np.float32)
        t = time_feats(g["DateTime"].tail(LB))
        onehot = np.zeros((LB, len(routes)), dtype=np.float32)
        onehot[:, rid2idx[route_id]] = 1.0
        X = np.concatenate([v, t, onehot], axis=1)[None, ...]

        y_scaled = model.predict(X, verbose=0).reshape(-1, 1)
        y = scaler.inverse_transform(y_scaled).ravel()
        next_hours = pd.date_range(end_dt, periods=HZ, freq="h")
        df_fc = pd.DataFrame({"DateTime": next_hours, "Predicted": y, "RouteId": route_id})
        st.success(f"✅ Forecast route {route_id} using GRU.")
        return df_fc, "GRU"
    except Exception as e:
        st.warning(f"⚠️ GRU failed for {route_id}: {e} → fallback MLP.")
        return forecast_mlp(route_id, base_date, scaler, routes)


# ---------------- STREAMLIT UI ----------------
def main():
    st.title("🚦 Traffic Forecast App (GRU + Fallback)")

    try:
        MODEL, META, SCALER, ROUTES, RID2IDX = load_models()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

    # Sidebar
    with st.sidebar:
        st.header("Settings")
        cities = list_cities()
        if not cities:
            st.error("No datasets found in data/processed_ds/")
            st.stop()
        city = st.selectbox("City", cities)
        zones = list_zones(city)
        zone = st.selectbox("Zone/Area", zones)
        routes = list_routes(city, None if zone == "(All)" else zone)
        if not routes:
            st.error("No routes found in dataset.")
            st.stop()

        route_id = st.selectbox("Route", routes)
        base_date = st.date_input("Base Date (forecast next 24h)", datetime.utcnow().date() - timedelta(days=1))

    # Forecast
    df_fc, model_used = forecast_gru(route_id, base_date, MODEL, META, SCALER, ROUTES, RID2IDX, city, zone)

    # Display
    if df_fc is not None and not df_fc.empty:
        st.subheader(f"Forecast for {route_id}")
        st.write(f"**Model used:** {model_used}")
        chart = (
            alt.Chart(df_fc)
            .mark_line(point=True)
            .encode(
                x="DateTime:T",
                y="Predicted:Q",
                tooltip=["DateTime:T", "Predicted:Q"]
            )
            .interactive()
        )
        st.altair_chart(chart, use_container_width=True)
        st.dataframe(df_fc.tail(10))
    else:
        st.warning("No forecast data available.")


if __name__ == "__main__":
    main()
