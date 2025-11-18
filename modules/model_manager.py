from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import pickle
from typing import Dict, List, Optional

import tensorflow as tf


@dataclass
class ModelContext:
    """Thông tin model cho 1 'family' (I-94, Fremont, ...)."""
    gru_model: Optional[tf.keras.Model]
    meta: dict
    scaler: object
    routes_model: List[str]
    rid2idx: Dict[str, int]
    lookback: int
    horizon: int
    family_name: str
    model_dir: Path


def _detect_model_dir(city: str, zone: Optional[str]) -> Path:
    """
    Chọn thư mục model tương ứng với (city, zone).

    Ưu tiên các folder con trong `model/` nếu có file seq_meta.json:
      - model/{City}_{Zone}/seq_meta.json
      - model/{City}/seq_meta.json

    Nếu không thấy gì → fallback về thư mục gốc `model/`
    (giữ nguyên behaviour cũ cho I-94).
    """
    base = Path("model")

    candidates = []

    # Ví dụ: model/Seattle_FremontBridge/
    if zone and zone != "(All)":
        norm = f"{city}_{zone}".replace(" ", "_")
        candidates.append(base / norm)

    # Ví dụ: model/Seattle/
    norm_city = city.replace(" ", "_")
    candidates.append(base / norm_city)

    # Cuối cùng: thư mục gốc model/
    candidates.append(base)

    for d in candidates:
        meta_path = d / "seq_meta.json"
        if meta_path.exists():
            return d

    # Nếu ngay cả model/ cũng không có meta -> trả về base,
    # để phần load báo lỗi dễ hiểu.
    return base


def load_model_context(city: str, zone: Optional[str]) -> ModelContext:
    """
    Load GRU + meta + scaler cho city/zone.

    - Nếu không có model riêng cho city/zone, sẽ dùng model/ gốc.
    - Không động vào behaviour cũ: I-94 vẫn đọc từ model/.
    """
    model_dir = _detect_model_dir(city, zone)
    family_name = model_dir.name if model_dir.name != "model" else "default"

    meta_path = model_dir / "seq_meta.json"
    scaler_path = model_dir / "vehicles_scaler.pkl"
    model_path = model_dir / "traffic_seq.keras"

    if not meta_path.exists():
        raise FileNotFoundError(
            f"⚠️ Không tìm thấy meta seq_meta.json cho city={city}, zone={zone} "
            f"(đã kiểm tra trong {model_dir})."
        )

    with open(meta_path, "r") as f:
        meta = json.load(f)

    routes_model = meta.get("routes", [])
    if not routes_model:
        raise FileNotFoundError(
            f"⚠️ seq_meta.json trong {model_dir} không có key 'routes'."
        )

    lookback = int(meta.get("LOOKBACK", 168))
    horizon = int(meta.get("HORIZON", 24))
    rid2idx = {rid: idx for idx, rid in enumerate(routes_model)}

    if not scaler_path.exists():
        raise FileNotFoundError(
            f"⚠️ Thiếu scaler vehicles_scaler.pkl trong {model_dir}. Hãy train trước."
        )
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    gru_model = None
    if model_path.exists():
        gru_model = tf.keras.models.load_model(model_path)
    else:
        # Không có GRU cũng được, khi đó forecast_gru có thể không được dùng
        gru_model = None

    return ModelContext(
        gru_model=gru_model,
        meta=meta,
        scaler=scaler,
        routes_model=routes_model,
        rid2idx=rid2idx,
        lookback=lookback,
        horizon=horizon,
        family_name=family_name,
        model_dir=model_dir,
    )
