# scripts/prep_data.py
import sys
from pathlib import Path
import pandas as pd
import glob

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def find_i94_csv():
    # Tìm file I-94 trong thư mục raw (tên file trên Kaggle có thể khác nhau)
    candidates = []
    for pat in [
        "i94/*.csv",
        "i94/*/*.csv",
        "*metro*interstate*traffic*volume*.csv",
        "*Metro_Interstate_Traffic_Volume*.csv",
        "*.csv",
    ]:
        candidates += glob.glob(str(RAW_DIR / pat))
    # Ưu tiên file có cột 'date_time' & 'traffic_volume'
    for f in candidates:
        try:
            head = pd.read_csv(f, nrows=5)
            if {"date_time", "traffic_volume"}.issubset({c.lower() for c in head.columns}):
                return Path(f)
        except Exception:
            continue
    return None

def main():
    src = find_i94_csv()
    if src is None:
        print("Không tìm thấy CSV I-94 trong data/raw/. Hãy tải Kaggle dataset về data/raw/i94/", file=sys.stderr)
        sys.exit(1)

    print(f"Đọc: {src}")
    df = pd.read_csv(src)

    # Chuẩn tên cột (lowercase)
    df.columns = [c.strip() for c in df.columns]
    # Tìm cột date_time & traffic_volume (không phân biệt hoa thường)
    date_col = next((c for c in df.columns if c.lower() == "date_time"), None)
    vol_col  = next((c for c in df.columns if c.lower() == "traffic_volume"), None)
    if not date_col or not vol_col:
        raise ValueError("Không thấy cột date_time hoặc traffic_volume trong file.")

    # Parse datetime về UTC để đồng nhất
    dt = pd.to_datetime(df[date_col], utc=True, errors="coerce")
    out = pd.DataFrame({
        "DateTime": dt,
        "City": "Minneapolis-StPaul",
        "ZoneName": "I-94 Corridor",
        "RouteId": "I-94-WB",
        "Vehicles": pd.to_numeric(df[vol_col], errors="coerce"),
    })
    out = out.dropna(subset=["DateTime", "Vehicles"]).sort_values("DateTime")

    # I-94 đã là hourly; nếu nguồn khác là 5 phút, bạn có thể resample tại đây

    dst = OUT_DIR / "i94_hourly.parquet"
    out.to_parquet(dst, index=False)
    print(f"Saved: {dst.resolve()} (rows={len(out)})")

if __name__ == "__main__":
    main()
