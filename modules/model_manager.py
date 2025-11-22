from dataclasses import dataclass
from pathlib import Path
import json

import joblib
from tensorflow.keras.models import load_model


@dataclass
class ModelContext:
    # GRU seq model (bắt buộc, vì _detect_model_dir đang chọn theo file GRU)
    gru_model: object | None
    # RNN seq model (tuỳ chọn – có thể None nếu chưa train)
    rnn_model: object | None
    # Meta & các thông tin đi kèm
    meta: dict
    scaler: object
    routes_model: list
    rid2idx: dict
    lookback: int
    horizon: int
    family_name: str


def _detect_model_dir(city: str, zone: str | None) -> Path:
    """
    Chọn thư mục model tương ứng với (city, zone).

    Ưu tiên:
      1) Special-case I94: model/I94
      2) model/<City>_<Zone>  (ví dụ Seattle_FremontBridge)
      3) model/<City>
      4) model/ (fallback cuối cùng, nếu còn dùng)
    """
    base = Path("model")
    candidates = []

    city_str = (city or "").strip()
    zone_str = (zone or "").strip() if zone else None

    # 1) Special-case I94
    if city_str.lower() == "minneapolis" or (zone_str and zone_str.upper() == "I94"):
        candidates.append(base / "I94")

    # 2) City_Zone
    if zone_str and zone_str != "(All)":
        norm = f"{city_str}_{zone_str}".replace(" ", "_")
        candidates.append(base / norm)

    # 3) City-only
    if city_str:
        candidates.append(base / city_str.replace(" ", "_"))

    # 4) Fallback root
    candidates.append(base)

    for d in candidates:
        meta_path = d / "seq_meta.json"
        scaler_path = d / "vehicles_scaler.pkl"
        model_path = d / "traffic_seq.keras"  # GRU chính
        if meta_path.exists() and scaler_path.exists() and model_path.exists():
            print(f"[ModelManager] Using model dir: {d}")
            return d

    raise FileNotFoundError(
        f"Không tìm thấy model dir hợp lệ cho city={city_str}, zone={zone_str}. "
        f"Đã thử: {', '.join(str(c) for c in candidates)}"
    )


def load_model_context(city: str, zone: str | None = None) -> ModelContext:
    """
    Load context model cho (city, zone):

    - GRU model:  traffic_seq.keras
    - RNN model:  traffic_rnn_seq.keras (optional)
    - meta:       seq_meta.json
    - scaler:     vehicles_scaler.pkl
    - routes_model, rid2idx, lookback, horizon, family_name
    """
    model_dir = _detect_model_dir(city, zone)

    # Meta
    meta_path = model_dir / "seq_meta.json"
    with open(meta_path, "r") as f:
        meta = json.load(f)

    lookback = int(meta.get("LOOKBACK", 168))
    horizon = int(meta.get("HORIZON", 24))
    routes_model = list(meta.get("routes", []))
    rid2idx = {rid: idx for idx, rid in enumerate(routes_model)}

    # Scaler (joblib.load đọc được cả pickle lẫn joblib)
    scaler_path = model_dir / "vehicles_scaler.pkl"
    scaler = joblib.load(scaler_path)

    # GRU model (chính) – file cũ: traffic_seq.keras
    gru_path = model_dir / "traffic_seq.keras"
    if gru_path.exists():
        gru_model = load_model(gru_path)
    else:
        gru_model = None
        print(f"[ModelManager] WARNING: Không tìm thấy {gru_path}, gru_model=None.")

    # RNN model (mới, optional) – file: traffic_rnn_seq.keras
    rnn_path = model_dir / "traffic_rnn_seq.keras"
    if rnn_path.exists():
        rnn_model = load_model(rnn_path)
        print(f"[ModelManager] Loaded RNN model from {rnn_path}")
    else:
        rnn_model = None
        print(f"[ModelManager] INFO: Không có RNN model tại {rnn_path} (optional).")

    family_name = model_dir.name

    print(
        f"[ModelManager] Loaded family='{family_name}' "
        f"(LOOKBACK={lookback}, HORIZON={horizon}, routes={routes_model})"
    )

    return ModelContext(
        gru_model=gru_model,
        rnn_model=rnn_model,
        meta=meta,
        scaler=scaler,
        routes_model=routes_model,
        rid2idx=rid2idx,
        lookback=lookback,
        horizon=horizon,
        family_name=family_name,
    )
