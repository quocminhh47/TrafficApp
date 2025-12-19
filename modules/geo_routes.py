from pathlib import Path
import json
import pandas as pd

ROUTES_GEO_PATH = Path("data/routes_geo.json")


def load_routes_geo() -> pd.DataFrame:
    """
    Đọc toàn bộ metadata toạ độ cho các route.
    Trả về DataFrame với các cột:
      city, zone, route_id, name, lat, lon
    """
    if not ROUTES_GEO_PATH.exists():
        return pd.DataFrame(columns=["city", "zone", "route_id", "name", "lat", "lon"])

    with open(ROUTES_GEO_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    # đảm bảo đủ cột
    for col in ["city", "zone", "route_id", "name", "lat", "lon"]:
        if col not in df.columns:
            df[col] = None
    return df


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
