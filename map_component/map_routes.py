import os
import streamlit.components.v1 as components

# When developing, set MAP_COMPONENT_DEV=1 to point to the Vite dev server.
# In production (default), serve the built or source frontend directly from disk.
_DEV_MODE = os.getenv("MAP_COMPONENT_DEV", "0") == "1"

if _DEV_MODE:
    _component_func = components.declare_component(
        "map_routes",
        url="http://localhost:5173",
    )
else:
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    dist_dir = os.path.join(parent_dir, "frontend", "dist")
    fallback_dir = os.path.join(parent_dir, "frontend")
    static_path = dist_dir if os.path.exists(dist_dir) else fallback_dir

    _component_func = components.declare_component(
        "map_routes",
        path=static_path,
    )


def map_routes(
    routes_data,
    selected_route_id,
    all_routes=None,
    *,
    focus_bounds=None,
    district_center=None,
    district_centers_geojson=None,
    district_capacity=None,
    selected_district=None,
    key=None,
):
    """
    Hiển thị bản đồ các tuyến đường (routes) với custom Leaflet frontend.

    Parameters
    ----------
    routes_data : list[dict]
        Danh sách route (đã lọc theo city/zone hiện tại) để vẽ marker.
        Mỗi item thường có: {city, zone, route_id, lat, lon, ...}

    selected_route_id : str or None
        Route đang được chọn ở sidebar (Streamlit), dùng để highlight/zoom.

    all_routes : list[dict] or None
        Danh sách tất cả route của toàn project (mọi city),
        dùng để tính GLOBAL bounds cho nút "Reset view".

    key : str or None
        Key cho component trong Streamlit.

    Returns
    -------
    str or None
        route_id của marker vừa được click trên map, hoặc None nếu chưa click.
    """

    component_value = _component_func(
        data=routes_data,
        selected_route_id=selected_route_id,
        all_routes=all_routes,  # << gửi thêm xuống frontend
        focus_bounds=focus_bounds,
        district_center=district_center,
        district_centers_geojson=district_centers_geojson,
        district_capacity=district_capacity,
        selected_district=selected_district,
        key=key,
        default=None,
    )

    return component_value
