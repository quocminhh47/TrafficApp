# scripts/ingest_metro_i94.py
import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw/Minneapolis/Metro_Interstate_Traffic_Volume.csv")
OUT_DIR = Path("data/processed_ds/Minneapolis/I94")
OUT_FILE = OUT_DIR / "i94_main.parquet"


def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw file not found: {RAW_PATH}")

    print("[INFO] Loading raw CSV...")
    df = pd.read_csv(RAW_PATH)

    # Chuẩn hóa cột
    if "date_time" in df.columns:
        df = df.rename(columns={"date_time": "DateTime"})
    if "traffic_volume" in df.columns:
        df = df.rename(columns={"traffic_volume": "Vehicles"})

    if "DateTime" not in df.columns or "Vehicles" not in df.columns:
        raise ValueError("Raw CSV must contain 'date_time' and 'traffic_volume' columns.")

    df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
    df = df.dropna(subset=["DateTime", "Vehicles"])
    df = df.sort_values("DateTime")

    # RouteId cố định (I-94 West Bound)
    df["RouteId"] = "I-94-WB"

    # Resample về 1h cho chắc chắn
    g = (
        df.set_index("DateTime")
        .resample("1h")["Vehicles"]
        .mean()
        .interpolate(limit=3)
        .reset_index()
    )
    g["RouteId"] = "I-94-WB"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    g.to_parquet(OUT_FILE, index=False)
    print(f"[INFO] Saved parquet to {OUT_FILE}")
    print(f"[INFO] Time range: {g['DateTime'].min()} → {g['DateTime'].max()} rows={len(g)}")


if __name__ == "__main__":
    main()
