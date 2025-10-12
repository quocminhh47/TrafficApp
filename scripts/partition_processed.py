# scripts/partition_processed.py (v2 - verbose & safe)
from pathlib import Path
import glob
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

SRC = Path("data/processed")
DST = Path("data/processed_ds")
DST.mkdir(parents=True, exist_ok=True)

def guess_city_zone(fp: str):
    l = fp.lower()
    if "hcmc" in l or "hochiminh" in l or "hcm" in l:
        return "HoChiMinhCity", "HCMC"
    if "i94" in l or "minneapolis" in l or "stpaul" in l:
        return "Minneapolis-StPaul", "I-94 Corridor"
    if "berlin" in l:
        return "Berlin", "Berlin"
    return "Unknown", "Unknown"

def normalize_df(df: pd.DataFrame, src_path: str) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    # DateTime
    if "datetime" not in cols and "date_time" not in cols and "timestamp" not in cols and "time" not in cols and "updated_at" not in cols:
        raise ValueError(f"{src_path}: missing time column")
    time_col = (
        cols.get("datetime") or cols.get("date_time") or cols.get("timestamp")
        or cols.get("time") or cols.get("updated_at")
    )
    df["DateTime"] = pd.to_datetime(df[time_col], utc=True, errors="coerce")

    # Vehicles
    vcol = None
    for key in ["vehicles","flow","volume","count","speed","velocity","los","level_of_service","level"]:
        for c in df.columns:
            if key in c.lower():
                vcol = c; break
        if vcol: break
    if vcol is None:
        raise ValueError(f"{src_path}: missing Vehicles/flow/speed/LOS")
    if "los" in vcol.lower() or "level" in vcol.lower():
        s = df[vcol].astype(str).str.strip().str.upper()
        map_af = {"A":5,"B":4,"C":3,"D":2,"E":1,"F":0}
        df["Vehicles"] = s.map(map_af)
        if df["Vehicles"].isna().all():
            df["Vehicles"] = pd.to_numeric(df[vcol], errors="coerce")
    else:
        df["Vehicles"] = pd.to_numeric(df[vcol], errors="coerce")

    # City/ZoneName/RouteId
    city, zone = guess_city_zone(src_path)
    if "City" not in df.columns:
        df["City"] = city
    if "ZoneName" not in df.columns:
        df["ZoneName"] = zone
    if "RouteId" not in df.columns:
        route = None
        for cand in ["routeid","segment_id","segmentid","road_id","link_id","edge_id","segment","link","road","id","zone"]:
            if cand in cols:
                route = cols[cand]; break
        df["RouteId"] = df[route].astype(str) if route else f"{city}-ALL"

    out = df[["DateTime","City","ZoneName","RouteId","Vehicles"]].copy()
    out = out.dropna(subset=["DateTime","Vehicles"])
    return out

def main():
    files = sorted(set(glob.glob(str(SRC / "*.parquet"))))
    if not files:
        raise SystemExit("No parquet in data/processed/")

    dfs = []
    for f in files:
        try:
            df0 = pd.read_parquet(f)
            print(f"[READ] {f} -> rows={len(df0)} cols={list(df0.columns)}")
            if len(df0)==0:
                print(f"[SKIP EMPTY] {f}"); continue
            df = normalize_df(df0, f)
            print(f"[OK] {f} -> normalized rows={len(df)}")
            dfs.append(df)
        except Exception as e:
            print(f"[SKIP] {f} -> {e}")

    if not dfs:
        raise SystemExit("No valid frames to write.")

    big = pd.concat(dfs, ignore_index=True)
    if len(big)==0:
        raise SystemExit("Concatenated is empty.")

    # Arrow types
    for c in ["City","ZoneName","RouteId"]:
        big[c] = big[c].astype("string")

    table = pa.Table.from_pandas(big, preserve_index=False)
    print("[WRITE] rows:", table.num_rows, "cols:", table.schema)

    ds.write_dataset(
        table,
        base_dir=str(DST),
        format="parquet",
        partitioning=["City","ZoneName"],
        existing_data_behavior="overwrite_or_ignore",
        # đảm bảo hợp lệ:
        max_rows_per_group=100_000,
        max_rows_per_file=500_000,
    )
    print("[DONE] ->", DST.resolve())

if __name__ == "__main__":
    main()
