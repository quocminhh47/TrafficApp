import streamlit.components.v1 as components
import os

_component_dir = os.path.dirname(os.path.abspath(__file__))
_map_click_component = components.declare_component(
    "map_click", path=_component_dir + "/frontend"
)

def map_click(data, key=None):
    """
    data: dict chứa {"lat":..., "lon":..., "route_id": ...}
    return: route_id được click (hoặc None)
    """
    return _map_click_component(data=data, key=key, default=None)
