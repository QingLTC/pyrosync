# Portions of this code were generated with assistance from Claude (Anthropic), an SI tool.
"""Read NASA FIRMS archive CSVs into one standardized table, and save/load tables."""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

from . import config as C

STD_COLS = ["lat", "lon", "date", "sensor", "daynight", "frp", "src", "type"]

# FIRMS archive file names carry a product token, e.g.
#   fire_archive_M-C61_846654.csv   (MODIS, Terra + Aqua together)
#   fire_archive_SV-C2_846665.csv   (VIIRS S-NPP)
#   fire_archive_J1V-C2_846666.csv  (VIIRS NOAA-20)
#   fire_nrt_J2V-C2_846668.csv      (VIIRS NOAA-21, near real-time)
_VIIRS_TOKENS = {"J2V": C.N21, "J1V": C.N20, "SV": C.SNPP}


def _family_and_viirs_sensor(path: str, cols: list[str]) -> tuple[str, str | None]:
    name = os.path.basename(path).upper()
    if "_M-C6" in name or "MODIS" in name:
        return "MODIS", None
    for token, sensor in _VIIRS_TOKENS.items():   # J2V/J1V checked before SV
        if f"_{token}-" in name or f"_{token}_" in name:
            return "VIIRS", sensor
    # Fall back on columns: MODIS files have bright_t31, VIIRS files bright_ti4.
    if "bright_t31" in cols or "brightness" in cols:
        return "MODIS", None
    return "VIIRS", None


def _viirs_sensor_from_column(sat: pd.Series) -> pd.Series:
    s = sat.astype(str).str.upper().str.strip()
    out = pd.Series(C.SNPP, index=s.index)
    out[s.isin(["N20", "1", "J1", "NOAA-20", "NOAA20"])] = C.N20
    out[s.isin(["N21", "2", "J2", "NOAA-21", "NOAA21"])] = C.N21
    return out


def load_firms_csv(path: str) -> pd.DataFrame:
    """Read one FIRMS CSV and return the standard columns, already quality-filtered."""
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]
    family, viirs_sensor = _family_and_viirs_sensor(path, list(df.columns))

    out = pd.DataFrame({
        "lat": df["latitude"].astype(float),
        "lon": df["longitude"].astype(float),
        "date": pd.to_datetime(df["acq_date"]),
        "daynight": df["daynight"].astype(str).str.upper().str[0],
        "frp": pd.to_numeric(df.get("frp"), errors="coerce").fillna(0.0),
        "src": np.where(os.path.basename(path).lower().startswith("fire_nrt")
                        | df.get("version", pd.Series("", index=df.index)).astype(str).str.contains("NRT"),
                        "NRT", "SP"),
        "type": pd.to_numeric(df["type"], errors="coerce") if "type" in df else np.nan,
    })

    if family == "MODIS":
        first = df["satellite"].astype(str).str.upper().str[0]
        out["sensor"] = np.where(first == "A", C.AQUA, C.TERRA)
        conf = pd.to_numeric(df["confidence"], errors="coerce").fillna(0)
        keep = conf >= C.MODIS_MIN_CONF
    else:
        out["sensor"] = viirs_sensor if viirs_sensor else _viirs_sensor_from_column(df["satellite"])
        conf = df["confidence"].astype(str).str.lower().str[0]
        keep = ~conf.isin(C.VIIRS_DROP_CONF)

    return out.loc[keep, STD_COLS].reset_index(drop=True)


def load_raw_dir(raw_dir: str = C.RAW_DIR) -> pd.DataFrame:
    """Load every CSV under raw_dir (sub-folders included) and clean it up.

    - Standard-processing (SP) rows win; NRT rows are kept only after the last SP date
      of the same sensor (FIRMS replaces NRT with SP after ~5 months).
    - Static heat sources: cells where SP data ever flagged type 1/2/3 (volcano, other
      static land source, offshore) are removed from all rows, NRT included, because
      NRT files carry no type column.
    """
    paths = sorted(glob.glob(os.path.join(raw_dir, "**", "*.csv"), recursive=True))
    if not paths:
        raise FileNotFoundError(f"No CSV files found under {raw_dir!r}")
    frames = []
    for p in paths:
        part = load_firms_csv(p)
        print(f"  read {os.path.basename(p):45s} {len(part):>9,} rows "
              f"{sorted(part['sensor'].unique())}")
        frames.append(part)
    df = pd.concat(frames, ignore_index=True)

    # SP beats NRT
    sp_last = df[df["src"] == "SP"].groupby("sensor")["date"].max()
    nrt = df["src"] == "NRT"
    cutoff = df["sensor"].map(sp_last)
    df = df[~nrt | cutoff.isna() | (df["date"] > cutoff)]
    df = df.drop_duplicates(subset=["sensor", "lat", "lon", "date", "daynight"])

    # Static heat sources
    ix = np.floor(df["lon"] / C.CELL_FINE).astype("int64")
    iy = np.floor(df["lat"] / C.CELL_FINE).astype("int64")
    key = ix * 100_000 + (iy + 50_000)          # one integer per fine cell
    static_keys = np.unique(key[df["type"].isin([1, 2, 3])])
    if len(static_keys):
        is_static = key.isin(static_keys)
        print(f"  removed {int(is_static.sum()):,} detections in "
              f"{len(static_keys):,} static-source cells")
        df = df[~is_static]

    return df.reset_index(drop=True)


# --- table storage (parquet when pyarrow is installed, else gzipped CSV) ------
def _has_parquet() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


def save_table(df: pd.DataFrame, stem: str) -> str:
    os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
    if _has_parquet():
        path = stem + ".parquet"
        df.to_parquet(path, index=False)
    else:
        path = stem + ".csv.gz"
        df.to_csv(path, index=False)
    return path


def load_table(stem: str) -> pd.DataFrame:
    if os.path.exists(stem + ".parquet"):
        return pd.read_parquet(stem + ".parquet")
    return pd.read_csv(stem + ".csv.gz")
