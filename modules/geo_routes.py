from functools import lru_cache
from pathlib import Path
import json
import pandas as pd

ROUTES_GEO_PATH = Path("data/routes_geo.json")
HCMC_DISTRICT_CENTERS_PATH = Path("data/geo/hcmc_district_centers.json")


def load_routes_geo() -> pd.DataFrame:
    """
    Đọc toàn bộ metadata toạ độ cho các route.
    Trả về DataFrame với các cột:
      city, zone, route_id, name, district, lat, lon
    """
    if not ROUTES_GEO_PATH.exists():
        return pd.DataFrame(
            columns=["city", "zone", "route_id", "name", "district", "lat", "lon"]
        )

    with open(ROUTES_GEO_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    # đảm bảo đủ cột
    for col in ["city", "zone", "route_id", "name", "district", "lat", "lon"]:
        if col not in df.columns:
            df[col] = None
    return df


@lru_cache(maxsize=1)
def load_hcmc_district_centers() -> pd.DataFrame:
    """
    Đọc danh sách toạ độ trung tâm cho từng quận tại HCMC.
    """
    if not HCMC_DISTRICT_CENTERS_PATH.exists():
        return pd.DataFrame(columns=["district", "district_lat", "district_lon"])

    with open(HCMC_DISTRICT_CENTERS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data.get("district_centers", []) if isinstance(data, dict) else []
    df = pd.DataFrame(items)

    for col in ["district", "district_lat", "district_lon"]:
        if col not in df.columns:
            df[col] = None

    return df[["district", "district_lat", "district_lon"]]


def get_routes_geo_for_city_zone(df_geo: pd.DataFrame, city: str, zone: str | None):
    """
    Lọc metadata toạ độ theo city/zone nếu cần.
    - Nếu zone = None hoặc '(All)' → chỉ lọc theo city.
    """
    if df_geo is None or df_geo.empty:
        return df_geo

    mask = df_geo["city"].astype(str) == str(city)
    if zone and zone != "(All)":
        mask &= df_geo["zone"].astype(str) == str(zone)

    return df_geo[mask].copy()
