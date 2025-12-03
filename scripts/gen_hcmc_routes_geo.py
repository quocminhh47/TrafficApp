# scripts/gen_hcmc_routes_geo.py

import os
import json
import unicodedata
import pandas as pd

# ===================== CONFIG =====================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CSV_PATH = os.path.join(BASE_DIR, "data", "raw", "hcmc", "train.csv")
METRICS_PATH = os.path.join(BASE_DIR, "model", "hcmc", "gru_congestion_metrics.csv")

# file JSON output – bạn merge vào routes.json hoặc load riêng
OUTPUT_JSON = os.path.join(BASE_DIR, "data", "routes_hcmc.json")

CITY_NAME = "HoChiMinh"
ZONE_NAME = "HCMC"          # tuỳ bạn, dùng làm zone cho HCMC

# chỉ lấy các tuyến đã train GRU thành công
MIN_STATUS = "trained"

# nếu muốn lọc thêm theo F1 (ví dụ >= 0.5), set ngưỡng ở đây
MIN_F1 = 0.0   # tạm thời lấy hết; đổi thành 0.5 nếu muốn loại model tệ


def slugify(text: str) -> str:
    """Giống logic bạn dùng khi train (loại dấu, về lowercase, thay space=underscore)."""
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower().replace(" ", "_")
    for ch in ["/", "\\", ",", ".", ":", ";", "(", ")", "'", '"']:
        text = text.replace(ch, "")
    return text


def main():
    print(f"[INFO] Đọc train: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    print(f"[INFO] Đọc metrics: {METRICS_PATH}")
    m = pd.read_csv(METRICS_PATH)

    # chỉ chọn những tuyến status=trained và F1 >= MIN_F1
    m = m[m["status"] == "trained"].copy()
    m = m[m["f1"].fillna(0) >= MIN_F1].copy()

    if m.empty:
        print("[WARN] Không có tuyến nào đủ điều kiện theo metrics.")
        return

    # build DateTime để group theo street_name
    df["date"] = pd.to_datetime(df["date"])
    period_num = df["period"].str.extract(r"period_(\d+)_(\d+)", expand=True).astype(int)
    df["hour"] = period_num[0]
    df["minute"] = period_num[1]
    df["DateTime"] = (
        df["date"]
        + pd.to_timedelta(df["hour"], unit="h")
        + pd.to_timedelta(df["minute"], unit="m")
    )

    routes = []

    for _, row in m.iterrows():
        street_name = row["street_name"]
        slug = row["slug"]

        df_st = df[df["street_name"] == street_name]
        if df_st.empty:
            print(f"[WARN] Không tìm thấy dữ liệu geo cho '{street_name}' – skip")
            continue

        # trung bình toạ độ node đầu / cuối để ra 1 điểm trung tâm
        lat = (df_st["lat_snode"].mean() + df_st["lat_enode"].mean()) / 2.0
        lon = (df_st["long_snode"].mean() + df_st["long_enode"].mean()) / 2.0

        route = {
            "city": CITY_NAME,
            "zone": ZONE_NAME,
            "route_id": slug,  # dùng slug làm route_id cho đồng bộ với model
            "name": f"{street_name} (HCMC)",
            "lat": float(lat),
            "lon": float(lon),
        }
        routes.append(route)

    print(f"[INFO] Số tuyến HCMC xuất ra: {len(routes)}")
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(routes, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Đã lưu routes HCMC vào: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
