import os
import streamlit.components.v1 as components

# Create a _RELEASE constant. We'll set this to False while developing
# and True when we're ready to package it up.
_RELEASE = False

if not _RELEASE:
    # DEVELOPMENT MODE:
    # In this mode, we rely on the vite dev server running on port 5173.
    _component_func = components.declare_component(
        "map_routes",
        url="http://localhost:5173",
    )
else:
    # PRODUCTION MODE:
    # In this mode, we point to the build directory.
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "frontend", "dist")
    _component_func = components.declare_component(
        "map_routes",
        path=build_dir
    )


def map_routes(routes_data, selected_route_id, all_routes=None, key=None):
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
        key=key,
        default=None,
    )

    return component_value
