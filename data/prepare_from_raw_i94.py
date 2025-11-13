#!/usr/bin/env python
# data/prepare_from_raw_i94.py — chuyển raw CSV -> parquet huấn luyện
import pandas as pd
from pathlib import Path

RAW_FILE = Path("data/raw/i94/Metro_Interstate_Traffic_Volume.csv")
OUT_DIR = Path("data/processed_ds/Minneapolis/I94")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"[INFO] Reading raw CSV: {RAW_FILE}")
df = pd.read_csv(RAW_FILE)

# Chuyển cột thời gian sang UTC
df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce", utc=True)
df = df.rename(columns={"date_time": "DateTime", "traffic_volume": "Vehicles"})
df = df[["DateTime", "Vehicles"]].dropna()

# Thêm cột RouteId cố định
df["RouteId"] = "I-94-WB"

# Chuyển sang parquet
df.to_parquet(OUT_DIR / "I94_WB.parquet")
print(f"✅ Saved: {OUT_DIR / 'I94_WB.parquet'} ({len(df)} rows)")
print("Example:")
print(df.head())
