from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from filelock import FileLock
except ModuleNotFoundError:  # pragma: no cover - fallback for environments missing dependency
    import fcntl

    class FileLock:  # type: ignore[misc]
        """Minimal POSIX file lock used when filelock is unavailable."""

        def __init__(self, lock_file):
            self.lock_file = Path(lock_file)
            self._handle = None

        def acquire(self, timeout=None):
            self.lock_file.parent.mkdir(parents=True, exist_ok=True)
            self._handle = open(self.lock_file, "w")
            fcntl.flock(self._handle, fcntl.LOCK_EX)
            return True

        def release(self):
            if self._handle:
                fcntl.flock(self._handle, fcntl.LOCK_UN)
                self._handle.close()
                self._handle = None

        def __enter__(self):
            self.acquire()
            return self

        def __exit__(self, exc_type, exc, tb):
            self.release()

from modules.data_loader import list_zones
from modules.model_manager import load_model_context
from modules.model_utils import forecast_gru, forecast_rnn

DATA_ROOT = Path("data/processed_ds")
STEP_DAYS = 7
MODEL_TYPE = "GRU"

# Mapping known routes to the corresponding parquet stem (without _original)
ROUTE_FILE_MAP: dict[tuple[str, str, str], str] = {
    ("Minneapolis", "I94", "I-94-WB"): "i94_main",
    ("Seattle", "FremontBridge", "Fremont-East"): "Fremont_East",
    ("Seattle", "FremontBridge", "Fremont-Total"): "Fremont_Total",
    ("Seattle", "FremontBridge", "Fremont-West"): "Fremont_West",
}


def files_for(city: str, zone: str | None, file_name: str) -> Path | list[Path] | None:
    """Return parquet path(s) for a city/zone/file stem.

    If multiple files are found, the sorted list is returned so the caller can
    decide how to proceed.
    """

    file = f"{file_name}_original.parquet"
    if zone in (None, "(All)"):
        pat = DATA_ROOT / city / "**" / file
    else:
        pat = DATA_ROOT / city / zone / "**" / file

    matches = sorted(Path(p) for p in glob.glob(str(pat), recursive=True))
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    return matches


def _load_model_context_with_fallback(city: str, zone: str | None):
    zone_for_model = None if zone == "(All)" else zone
    try:
        ctx = load_model_context(city, zone_for_model)
        return ctx
    except FileNotFoundError as e:
        if zone != "(All)":
            raise

        zones_all = list_zones(city)
        for z in zones_all:
            if z == "(All)":
                continue
            try:
                ctx = load_model_context(city, z)
                return ctx
            except FileNotFoundError:
                continue

        raise e


def _normalize_zone_for_path(real_path: Path | None, city: str, zone: str | None) -> str:
    if real_path is not None:
        parent = real_path.parent
        if parent.name and parent.name != city:
            return parent.name
    if zone in (None, "(All)"):
        return "(All)" if zone is None else zone
    return zone


def build_forecast_cache(city: str, zone: str | None, route_id: str, file_name: str):
    real_path_obj = files_for(city, zone, file_name)
    if real_path_obj is None:
        print(f"❌ Không tìm thấy file dữ liệu gốc cho {city=}, {zone=}, {file_name=}")
        return None

    if isinstance(real_path_obj, list):
        real_path: Path | list[Path] = real_path_obj
        base_real = real_path[0] if real_path else None
    else:
        real_path = real_path_obj
        base_real = real_path_obj

    ext_zone = _normalize_zone_for_path(base_real, city, zone)
    ext_path = DATA_ROOT / city / ext_zone / f"{file_name}.parquet"

    try:
        ctx = _load_model_context_with_fallback(city, zone)
    except FileNotFoundError as e:
        print(str(e))
        return None

    df_real = pd.read_parquet(real_path)
    df_real = df_real[df_real["RouteId"] == route_id]

    if df_real.empty:
        print("❌ Không có data thật")
        return None

    if ext_path.exists():
        df_ext = pd.read_parquet(ext_path)
        df_ext = df_ext[(df_ext["RouteId"] == route_id)]
        hist = pd.concat([df_real, df_ext], ignore_index=True)
    else:
        hist = df_real.copy()

    today = pd.Timestamp.today().normalize()
    all_new_fc = []

    last_ext_day = hist["DateTime"].max()
    if pd.isna(last_ext_day):
        print("❌ Không có dữ liệu DateTime hợp lệ")
        return None
    last_ext_day = pd.to_datetime(last_ext_day).normalize()

    while last_ext_day < today:
        remaining_days = (today - last_ext_day).days
        n_days = STEP_DAYS
        if remaining_days <= STEP_DAYS:
            n_days = max(remaining_days - 1, 0)

        df_fc, anchor = forecast_week_from_history(
            hist=hist,
            route_id=route_id,
            ctx=ctx,
            n_days=n_days,
            model_type=MODEL_TYPE,
        )

        if df_fc.empty:
            break

        df_fc["is_forecast"] = True

        tmp = df_fc.rename(columns={"PredictedVehicles": "Vehicles"})
        hist = pd.concat(
            [hist, tmp[["DateTime", "Vehicles", "RouteId"]]],
            ignore_index=True,
        )
        all_new_fc.append(tmp)
        last_ext_day = pd.to_datetime(hist["DateTime"].max()).normalize()

    if not all_new_fc:
        print("Không có forecast mới.")
        return ext_path

    df_base = df_real.copy()
    df_add = pd.concat(all_new_fc, ignore_index=True)

    df_new = (
        pd.concat([df_base, df_add], ignore_index=True)
        .drop_duplicates(subset=["DateTime", "RouteId"], keep="last")
        .sort_values("DateTime")
    )

    mask = df_new["is_forecast"] == True
    noise = np.random.uniform(-0.03, 0.03, mask.sum())
    df_new.loc[mask, "Vehicles"] = (
        df_new.loc[mask, "Vehicles"] * (1 + noise)
    ).clip(lower=0).round().astype(int)

    if ext_path.exists():
        df_old = pd.read_parquet(ext_path)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new

    df_all = df_all.drop_duplicates(
        subset=["DateTime", "RouteId"], keep="last"
    )

    ext_path.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_parquet(ext_path, index=False)
    print(f"✅ Saved forecast → {ext_path}")
    return ext_path


def forecast_week_from_history(
    hist: pd.DataFrame,
    route_id,
    ctx,
    n_days=7,
    model_type="GRU",
):
    if hist.empty:
        return pd.DataFrame(), None

    hist = hist.copy()
    hist["DateTime"] = pd.to_datetime(hist["DateTime"], errors="coerce")
    hist = hist.dropna(subset=["DateTime", "Vehicles"])
    hist = hist.sort_values("DateTime")

    last_dt = hist["DateTime"].max()
    if pd.isna(last_dt):
        return pd.DataFrame(), None
    anchor_day_raw = last_dt.normalize()

    model_type_norm = (model_type or "GRU").upper()
    use_rnn = (
        model_type_norm == "RNN"
        and getattr(ctx, "rnn_model", None) is not None
    )

    if model_type_norm == "RNN" and not use_rnn:
        print(
            "[forecast_week_from_history] RNN selected but ctx.rnn_model is None → fallback GRU"
        )

    all_fc = []

    for k in range(1, n_days + 1):
        base_date = anchor_day_raw + pd.Timedelta(days=k)

        hist_start = base_date - pd.Timedelta(hours=ctx.lookback)
        df_hist = hist[
            (hist["DateTime"] >= hist_start)
            & (hist["DateTime"] < base_date)
        ]

        if len(df_hist) < ctx.lookback:
            print(
                f"[forecast_week_from_history] thiếu history cho {base_date.date()}, dừng."
            )
            break

        if use_rnn:
            df_fc_day, model_used = forecast_rnn(
                route_id=route_id,
                base_date=base_date,
                model=ctx.rnn_model,
                meta=ctx.meta,
                scaler=ctx.scaler,
                routes_model=ctx.routes_model,
                rid2idx=ctx.rid2idx,
                df_hist=df_hist,
            )
        else:
            df_fc_day, model_used = forecast_gru(
                route_id=route_id,
                base_date=base_date,
                model=ctx.gru_model,
                meta=ctx.meta,
                scaler=ctx.scaler,
                routes_model=ctx.routes_model,
                rid2idx=ctx.rid2idx,
                df_hist=df_hist,
            )

        if df_fc_day is None or df_fc_day.empty:
            break

        all_fc.append(df_fc_day)

        tmp = df_fc_day.rename(columns={"PredictedVehicles": "Vehicles"})
        hist = pd.concat(
            [hist, tmp[["DateTime", "Vehicles", "RouteId"]]],
            ignore_index=True,
        )

    if not all_fc:
        return pd.DataFrame(), anchor_day_raw

    df_fc_raw = pd.concat(all_fc, ignore_index=True)
    return df_fc_raw, anchor_day_raw


def get_default_file_name(city: str, zone: str | None, route_id: str) -> str | None:
    zone_norm = "(All)" if zone is None else zone
    key = (str(city), str(zone_norm), str(route_id))
    return ROUTE_FILE_MAP.get(key)


def _read_max_dt_for_route(path: Path, route_id: str) -> pd.Timestamp | None:
    try:
        df_meta = pd.read_parquet(path, columns=["DateTime", "RouteId"])
    except Exception:
        return None

    if df_meta.empty or "DateTime" not in df_meta or "RouteId" not in df_meta:
        return None

    df_meta["RouteId"] = df_meta["RouteId"].astype(str)
    df_meta = df_meta[df_meta["RouteId"] == str(route_id)]
    if df_meta.empty:
        return None

    dt = pd.to_datetime(df_meta["DateTime"], errors="coerce")
    dt = dt.dropna()
    if dt.empty:
        return None
    return dt.max()


def ensure_forecast_cache(
    city: str,
    zone: str | None,
    route_id: str,
    file_name: str,
    *,
    st_module=None,
) -> Path:
    st_mod = st_module
    try:
        if st_mod is None:
            import streamlit as st  # type: ignore

            st_mod = st
    except Exception:
        st_mod = None

    orig_paths = files_for(city, zone, file_name)
    resolved_zone = _normalize_zone_for_path(
        (orig_paths[0] if isinstance(orig_paths, list) and orig_paths else orig_paths),
        city,
        zone,
    )
    ext_path = DATA_ROOT / city / resolved_zone / f"{file_name}.parquet"
    session_key = f"forecast_cache::{city}::{zone}::{route_id}::{file_name}"

    if st_mod is not None and st_mod.session_state.get(session_key):
        return ext_path

    lock = FileLock(str(ext_path) + ".lock")
    today_norm = pd.Timestamp.today().normalize()

    with lock:
        needs_build = False
        if not ext_path.exists():
            needs_build = True
        else:
            max_dt = _read_max_dt_for_route(ext_path, route_id)
            if max_dt is None or pd.to_datetime(max_dt).normalize() < today_norm:
                needs_build = True

        if needs_build:
            if st_mod is not None:
                with st_mod.spinner("Đang cập nhật dữ liệu dự báo…"):
                    build_forecast_cache(city, zone, route_id, file_name)
            else:
                build_forecast_cache(city, zone, route_id, file_name)
        if st_mod is not None and ext_path.exists():
            st_mod.session_state[session_key] = True

    return ext_path

