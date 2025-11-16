# modules/data_loader.py

from pathlib import Path
import glob
import pandas as pd

DATA_ROOT = Path("data/processed_ds")


def _files_for(city: str, zone: str | None):
    """
    Trả về list path parquet cho 1 city + optional zone.
    """
    if zone in (None, "(All)"):
        pat = DATA_ROOT / city / "**" / "*.parquet"
    else:
        pat = DATA_ROOT / city / zone / "**" / "*.parquet"
    return sorted(glob.glob(str(pat), recursive=True))


def load_slice(
    city: str,
    zone: str | None = None,
    routes: list[str] | None = None,
    start_dt=None,
    end_dt=None,
) -> pd.DataFrame:
    """
    Đọc parquet theo City / Zone / RouteId / time window.
    Trả về DataFrame với cột: DateTime (UTC-naive), RouteId, Vehicles.
    - city: tên folder con dưới data/processed_ds (vd: "i94")
    - zone: None hoặc "(All)" để lấy toàn bộ, hoặc tên zone cụ thể
    - routes: list route_id (string). Nếu None → không filter.
    - start_dt, end_dt: datetime / string, có thể tz-aware hoặc naive.
      Nếu None → không giới hạn.
    """

    files = _files_for(city, zone)
    frames: list[pd.DataFrame] = []

    for f in files:
        try:
            df = pd.read_parquet(f)
        except Exception:
            continue

        # Kiểm tra cột bắt buộc
        if df.empty or "DateTime" not in df or "Vehicles" not in df:
            continue

        # Chuẩn hoá DateTime về UTC-naive
        dt = pd.to_datetime(df["DateTime"], utc=True, errors="coerce")
        dt = dt.dt.tz_convert(None)  # bỏ tz, giữ UTC-naive
        df["DateTime"] = dt

        # Vehicles numeric
        df["Vehicles"] = pd.to_numeric(df["Vehicles"], errors="coerce")

        # RouteId là string nếu có
        if "RouteId" not in df:
            continue
        df["RouteId"] = df["RouteId"].astype(str)

        # Bỏ NaN DateTime / Vehicles
        df = df.dropna(subset=["DateTime", "Vehicles"])
        if df.empty:
            continue

        # Filter routes
        if routes:
            routes_str = list(map(str, routes))
            df = df[df["RouteId"].isin(routes_str)]
            if df.empty:
                continue

        frames.append(df[["DateTime", "RouteId", "Vehicles"]])

    if not frames:
        return pd.DataFrame(columns=["DateTime", "RouteId", "Vehicles"])

    df_all = pd.concat(frames, ignore_index=True)

    # Filter time window (start_dt, end_dt)
    if start_dt is not None:
        s = pd.Timestamp(start_dt)
        if s.tzinfo is not None:
            s = s.tz_convert("UTC").tz_localize(None)
        df_all = df_all[df_all["DateTime"] >= s]

    if end_dt is not None:
        e = pd.Timestamp(end_dt)
        if e.tzinfo is not None:
            e = e.tz_convert("UTC").tz_localize(None)
        df_all = df_all[df_all["DateTime"] < e]

    return df_all.sort_values("DateTime").reset_index(drop=True)


def list_cities() -> list[str]:
    """
    Liệt kê các city trong data/processed_ds
    """
    if not DATA_ROOT.is_dir():
        return []
    return sorted(
        [p.name for p in DATA_ROOT.iterdir() if p.is_dir() and not p.name.startswith(".")]
    )


def list_zones(city: str) -> list[str]:
    """
    Liệt kê zone cho 1 city.
    Trả về list đã include "(All)" ở đầu.
    """
    base = DATA_ROOT / city
    if not base.is_dir():
        return []
    zones = [
        p.name for p in base.iterdir() if p.is_dir() and not p.name.startswith(".")
    ]
    return ["(All)"] + sorted(zones)


def list_routes(city: str, zone: str | None) -> list[str]:
    """
    Đọc nhanh RouteId unique từ parquet (không filter theo thời gian).
    """
    df = load_slice(city, None if zone in (None, "(All)") else zone,
                    routes=None, start_dt=None, end_dt=None)
    if df.empty:
        return []
    return sorted(df["RouteId"].astype(str).unique().tolist())
