# scripts/prep_data_hcmc.py
from pathlib import Path
import pandas as pd
import numpy as np

RAW = Path("data/raw/hcmc")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEGMENTS_CSV = RAW / "segments.csv"
STATUS_CSV   = RAW / "segment_status.csv"
TRAIN_CSV    = RAW / "train.csv"

def _read_csv(fp):
    return pd.read_csv(fp)

def _load_segments_lookup():
    """
    segments.csv: dùng để tra street_name theo segment (_id).
    Columns: ['_id','created_at','updated_at','s_node_id','e_node_id','length','street_id',
              'max_velocity','street_level','street_name','street_type']
    """
    seg = _read_csv(SEGMENTS_CSV)
    # map: segment_id -> street_name (fallback 'HCMC' nếu thiếu)
    look = seg[["_id", "street_name"]].rename(columns={"_id": "segment_id"})
    look["street_name"] = look["street_name"].fillna("HCMC").astype(str)
    return look

def _resample_hourly(df: pd.DataFrame, how: str):
    """
    df: DateTime, City, ZoneName, RouteId, Vehicles
    how: 'mean' | 'sum'
    """
    df = df.dropna(subset=["DateTime", "Vehicles"]).sort_values("DateTime")
    rule = {"Vehicles": how}
    out = (
        df.set_index("DateTime")
          .groupby(["City", "ZoneName", "RouteId"])
          .resample("1H")
          .agg(rule)
          .reset_index()
    )
    return out

def build_speed_from_status():
    """
    segment_status.csv -> speed (km/h)
    Columns: ['_id','updated_at','segment_id','velocity']
    """
    df = _read_csv(STATUS_CSV)
    # thời gian
    df["DateTime"] = pd.to_datetime(df["updated_at"], utc=True, errors="coerce")
    # giá trị
    df["Vehicles"] = pd.to_numeric(df["velocity"], errors="coerce")  # dùng chung tên cột 'Vehicles'
    df["RouteId"]  = df["segment_id"].astype(str)
    df["City"]     = "HoChiMinhCity"

    # Join street_name làm ZoneName
    look = _load_segments_lookup()
    df = df.merge(look, on="segment_id", how="left")
    df["ZoneName"] = df["street_name"].fillna("HCMC").astype(str)
    df = df[["DateTime", "City", "ZoneName", "RouteId", "Vehicles"]]

    # Resample hourly: speed -> mean
    return _resample_hourly(df, "mean")

def build_los_from_train():
    """
    train.csv -> LOS (A..F) theo ô 30 phút
    Columns: ['_id','segment_id','date','weekday','period','LOS', ...]
    'period' ví dụ: 'period_0_30', 'period_23_30', 'period_0_00'...
    """
    df = _read_csv(TRAIN_CSV)
    # map LOS A..F -> 5..0 (nếu là số, giữ số)
    s = df["LOS"].astype(str).str.strip().str.upper()
    map_af = {"A":5, "B":4, "C":3, "D":2, "E":1, "F":0}
    los_num = s.map(map_af)
    los_num = los_num.where(los_num.notna(), pd.to_numeric(df["LOS"], errors="coerce"))

    # reconstruct datetime từ date + period_* (30 phút)
    # Lấy số phút từ chuỗi period: 'period_<hour>_<min>'
    per = df["period"].astype(str).str.extract(r"period_(\d+)_(\d+)", expand=True)
    per.columns = ["hour", "minute"]
    per["hour"] = pd.to_numeric(per["hour"], errors="coerce")
    per["minute"] = pd.to_numeric(per["minute"], errors="coerce")
    base = pd.to_datetime(df["date"], utc=True, errors="coerce")
    dt = base + pd.to_timedelta(per["hour"].fillna(0), unit="h") + pd.to_timedelta(per["minute"].fillna(0), unit="m")

    out = pd.DataFrame({
        "DateTime": dt,
        "City": "HoChiMinhCity",
        "ZoneName": "HCMC",  # có thể thay bằng street_name nếu bạn muốn: join theo segment_id với segments.csv
        "RouteId": df["segment_id"].astype(str),
        "Vehicles": los_num.astype(float)  # LOS số hoá (0..5)
    })

    # Resample hourly: LOS -> mean (trung bình mức độ phục vụ)
    return _resample_hourly(out, "mean")

def main():
    outs = []

    if STATUS_CSV.exists():
        try:
            speed_hourly = build_speed_from_status()
            outs.append(("speed", speed_hourly))
            print("OK: segment_status -> speed_hourly", len(speed_hourly))
        except Exception as e:
            print("Skip segment_status:", e)

    if TRAIN_CSV.exists():
        try:
            los_hourly = build_los_from_train()
            outs.append(("los", los_hourly))
            print("OK: train -> los_hourly", len(los_hourly))
        except Exception as e:
            print("Skip train.csv:", e)

    if not outs:
        raise SystemExit("Không tạo được dữ liệu HCMC (thiếu segment_status.csv hoặc train.csv hợp lệ).")

    # Ghi riêng 2 file để app biết metric nào
    for kind, df in outs:
        if kind == "speed":
            df.to_parquet(OUT_DIR / "hcmc_speed_hourly.parquet", index=False)
        elif kind == "los":
            df.to_parquet(OUT_DIR / "hcmc_los_hourly.parquet", index=False)

    # (tuỳ chọn) Gộp 2 file thành 1 nếu bạn vẫn muốn
    # all_df = pd.concat([d for _, d in outs], ignore_index=True)
    # all_df.to_parquet(OUT_DIR / "hcmc_hourly.parquet", index=False)

    print("Saved:", (OUT_DIR / "hcmc_speed_hourly.parquet").resolve(), "exists=", (OUT_DIR / "hcmc_speed_hourly.parquet").exists())
    print("Saved:", (OUT_DIR / "hcmc_los_hourly.parquet").resolve(), "exists=", (OUT_DIR / "hcmc_los_hourly.parquet").exists())

if __name__ == "__main__":
    main()
