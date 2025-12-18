from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

import pandas as pd
import streamlit as st

from modules.geo_routes import load_routes_geo

HCMC_CITY = "HoChiMinh"
STEP_MINUTES = 30
CACHE_TTL_SECONDS = 600
LOW_THRESHOLD = 0.3
HIGH_THRESHOLD = 0.7


def load_routes_with_district() -> pd.DataFrame:
    """
    Đọc routes_geo và đảm bảo luôn có cột district.
    """
    df = load_routes_geo().copy()
    if "district" not in df.columns:
        df["district"] = None
    return df


def _flatten_districts(values: Iterable[Any]) -> list[str]:
    out: list[str] = []
    for v in values:
        if isinstance(v, (list, tuple, set)):
            for item in v:
                s = str(item).strip()
                if s:
                    out.append(s)
        else:
            s = str(v).strip()
            if s:
                out.append(s)
    return out


def get_wards(df_routes: pd.DataFrame | None = None) -> list[str]:
    """
    Lấy danh sách quận/huyện của HCMC có trong routes_geo (dựa vào cột district).
    """
    df = df_routes if df_routes is not None else load_routes_with_district()

    if df is None or df.empty or "district" not in df.columns:
        return []

    districts_col = df[df["city"] == HCMC_CITY]["district"].dropna().tolist()
    wards = _flatten_districts(districts_col)
    return sorted(set(wards))


def _classify_level(p_peak: float) -> str:
    if p_peak > HIGH_THRESHOLD:
        return "high"
    if p_peak > LOW_THRESHOLD:
        return "medium"
    return "low"


def _aggregate_forecast(df_fc: pd.DataFrame) -> Tuple[float, pd.Timestamp, float] | None:
    if df_fc is None or df_fc.empty or "ProbCongested" not in df_fc.columns:
        return None

    df_tmp = df_fc.copy()
    df_tmp["DateTime"] = pd.to_datetime(df_tmp["DateTime"], errors="coerce")
    df_tmp = df_tmp.dropna(subset=["DateTime"])

    probs = df_tmp["ProbCongested"].astype(float).clip(0.0, 1.0).dropna()
    if probs.empty:
        return None

    p_peak = float(probs.max())
    p_mean = float(probs.mean())
    idx_peak = probs.idxmax()
    t_peak = df_tmp.loc[idx_peak, "DateTime"]

    return p_peak, t_peak, p_mean


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def _compute_ward_next_2h_report_cached(
    ward: str,
    anchor_iso: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, list[str]], Dict[str, float]]:
    """
    Cacheable phần tính toán báo cáo 2h tới theo phường.
    anchor_iso dùng làm key để cache theo slot 30'.
    """
    _ = anchor_iso  # dùng để tạo cache key ổn định theo thời gian

    routes_geo = load_routes_with_district()
    df_geo = routes_geo[routes_geo["city"] == HCMC_CITY].copy()
    df_geo = df_geo[
        df_geo["district"].apply(
            lambda d: ward in _flatten_districts([d] if not isinstance(d, (list, tuple, set)) else d)
        )
    ]

    if df_geo.empty:
        empty = pd.DataFrame(
            columns=[
                "route_id",
                "route_name",
                "p_peak",
                "p_mean",
                "level",
                "t_peak",
                "t_peak_label",
                "p_peak_pct",
                "p_mean_pct",
            ]
        )
        return empty, empty, {"avoid": [], "prefer": []}, {}

    from app import forecast_hcmc_next_2h

    rows: list[dict[str, Any]] = []
    p_peak_map: dict[str, float] = {}

    for _, r in df_geo.iterrows():
        route_id = str(r["route_id"])
        out = forecast_hcmc_next_2h(route_id, routes_geo)
        if out is None:
            continue

        df_fc, full_name = out
        agg = _aggregate_forecast(df_fc)
        if agg is None:
            continue

        p_peak, t_peak, p_mean = agg
        level = _classify_level(p_peak)
        p_peak_map[route_id] = p_peak

        rows.append(
            {
                "route_id": route_id,
                "route_name": full_name or r.get("name", route_id),
                "p_peak": p_peak,
                "p_mean": p_mean,
                "level": level,
                "t_peak": pd.to_datetime(t_peak),
            }
        )

    df_res = pd.DataFrame(rows)

    if df_res.empty:
        empty = pd.DataFrame(
            columns=[
                "route_id",
                "route_name",
                "p_peak",
                "p_mean",
                "level",
                "t_peak",
                "t_peak_label",
                "p_peak_pct",
                "p_mean_pct",
            ]
        )
        return empty, empty, {"avoid": [], "prefer": []}, p_peak_map

    df_res["t_peak_label"] = df_res["t_peak"].dt.strftime("%H:%M")
    df_res["p_peak_pct"] = df_res["p_peak"] * 100.0
    df_res["p_mean_pct"] = df_res["p_mean"] * 100.0

    df_high = df_res[df_res["level"].isin(["high", "medium"])].sort_values(
        "p_peak", ascending=False
    )
    df_low = df_res[df_res["level"] == "low"].sort_values(
        "p_peak", ascending=True
    )

    suggestions = {
        "avoid": df_res[df_res["level"] == "high"]
        .sort_values("p_peak", ascending=False)["route_name"]
        .head(3)
        .tolist(),
        "prefer": df_res[df_res["level"] == "low"]
        .sort_values("p_peak", ascending=True)["route_name"]
        .head(3)
        .tolist(),
    }

    return df_high, df_low, suggestions, p_peak_map


def compute_ward_next_2h_report(
    ward: str, now_ts: pd.Timestamp
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, list[str]], Dict[str, float]]:
    """
    Tính báo cáo 2h tới cho một quận/huyện:
    - df_high: các tuyến nguy cơ kẹt (high/medium)
    - df_low: các tuyến thoáng (low)
    - suggestions: {avoid, prefer}
    - p_peak_map: route_id -> p_peak (0..1)
    """
    now_ts = pd.Timestamp(now_ts)
    if now_ts.tzinfo is None:
        now_ts = now_ts.tz_localize("Asia/Ho_Chi_Minh")
    else:
        now_ts = now_ts.tz_convert("Asia/Ho_Chi_Minh")

    anchor = now_ts.floor(f"{STEP_MINUTES}min")
    anchor_iso = anchor.isoformat()

    return _compute_ward_next_2h_report_cached(ward, anchor_iso)
