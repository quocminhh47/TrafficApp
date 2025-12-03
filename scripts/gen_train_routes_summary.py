import os
import unicodedata
from typing import Dict, Optional

import numpy as np
import pandas as pd
from tensorflow import keras  # type: ignore

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

TRAIN_CSV = os.path.join(BASE_DIR, "data", "raw", "hcmc", "train.csv")
MODELS_DIR = os.path.join(BASE_DIR, "model", "hcmc")
OUT_CSV = os.path.join(BASE_DIR, "data", "raw", "hcmc", "train_routes_summary.csv")


def slugify(name: str) -> str:
    """
    Chuyển 'Lý Thường Kiệt' -> 'ly_thuong_kiet'
    (bỏ dấu, lower, chỉ giữ a-z0-9 + _)
    """
    name = str(name).strip()
    # bỏ dấu
    nfkd = unicodedata.normalize("NFKD", name)
    name_ascii = "".join(c for c in nfkd if not unicodedata.combining(c))
    # lower + replace space -> _
    name_ascii = name_ascii.lower().replace(" ", "_")
    # giữ lại chữ số + chữ cái + _
    cleaned = []
    for ch in name_ascii:
        if ch.isalnum() or ch == "_":
            cleaned.append(ch)
    return "".join(cleaned)


def build_streetname_to_slug(train_csv: str) -> Dict[str, str]:
    df = pd.read_csv(train_csv)
    if "street_name" not in df.columns:
        raise KeyError("train.csv không có cột 'street_name'")

    street_names = sorted(df["street_name"].dropna().unique())
    mapping = {}
    for s in street_names:
        mapping[s] = slugify(s)
    return mapping


def infer_timesteps_from_model(model_path: str) -> Optional[int]:
    """
    Load model .keras và đọc input_shape để lấy timesteps.
    Giả định input shape ~ (None, T, 1) hoặc (None, T, D).
    """
    try:
        model = keras.models.load_model(model_path)
    except Exception as ex:
        print(f"[WARN] Không load được model {model_path}: {ex}")
        return None

    in_shape = model.input_shape  # vd: (None, 164, 1)
    if isinstance(in_shape, (list, tuple)):
        # Keras đôi khi trả về list nếu multiple-input, ở đây giả định single
        if isinstance(in_shape, list):
            in_shape = in_shape[0]
    try:
        timesteps = int(in_shape[1])
        return timesteps
    except Exception:
        print(f"[WARN] Không đọc được timesteps từ input_shape={in_shape} của {model_path}")
        return None


def main():
    if not os.path.exists(TRAIN_CSV):
        raise FileNotFoundError(f"Không tìm thấy TRAIN_CSV: {TRAIN_CSV}")
    if not os.path.exists(MODELS_DIR):
        raise FileNotFoundError(f"Không tìm thấy MODELS_DIR: {MODELS_DIR}")

    print(f"[INFO] Đọc train.csv từ: {TRAIN_CSV}")
    street_to_slug = build_streetname_to_slug(TRAIN_CSV)

    # Đảo map: slug -> street_name (ưu tiên match duy nhất)
    slug_to_street: Dict[str, str] = {}
    for street, sl in street_to_slug.items():
        if sl not in slug_to_street:
            slug_to_street[sl] = street
        else:
            # nếu 2 street cùng slugify ra 1 slug -> log cảnh báo
            print(f"[WARN] slug trùng nhau '{sl}' cho '{slug_to_street[sl]}' và '{street}'")

    rows = []

    # Duyệt tất cả file model trong model/hcmc/
    for fname in sorted(os.listdir(MODELS_DIR)):
        if not fname.endswith(".keras"):
            continue
        fpath = os.path.join(MODELS_DIR, fname)

        # pattern: gru_congestion_<slug>.keras
        base = fname[:-6]  # bỏ '.keras'
        if base.startswith("gru_congestion_"):
            slug = base[len("gru_congestion_") :]
            model_type = "GRU"
        else:
            # nếu sau này bạn có LSTM/GRNN đặt tên khác thì xử lý thêm ở đây
            print(f"[WARN] Bỏ qua model không match pattern 'gru_congestion_': {fname}")
            continue

        street_name = slug_to_street.get(slug)
        if street_name is None:
            print(f"[WARN] Không tìm thấy street_name tương ứng slug='{slug}' (file {fname})")
            street_name = slug  # fallback: dùng slug làm tên

        timesteps = infer_timesteps_from_model(fpath)

        row = {
            "street_name": street_name,
            "slug": slug,
            "timesteps": timesteps if timesteps is not None else np.nan,
            "n_train_seq": np.nan,  # có thể để trống, eval script sẽ lo phần test sau
            "n_test_seq": np.nan,
            "acc": np.nan,
            "f1": np.nan,
            "status": "trained",
            "model_path": os.path.relpath(fpath, BASE_DIR),  # ví dụ: model/hcmc/...
        }
        rows.append(row)

    df_out = pd.DataFrame(rows)
    df_out = df_out.sort_values("street_name")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df_out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print(f"[INFO] Đã ghi train_routes_summary.csv với {len(df_out)} tuyến:")
    print(f"       {OUT_CSV}")


if __name__ == "__main__":
    main()
