# Portions of this code were generated with assistance from Claude (Anthropic), an SI tool.
"""Turn detections into comparable counting units.

One fire can light up 1 MODIS pixel but 4-5 VIIRS pixels. Counting raw detections
therefore inflates VIIRS. We count *fire cell-days* instead: a ~1 km grid cell with
at least one detection on a given day (per sensor, per day/night pass).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def assign_region(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    conds = [(lon >= x0) & (lon <= x1) & (lat >= y0) & (lat <= y1)
             for _, x0, x1, y0, y1 in C.REGIONS]
    names = [r[0] for r in C.REGIONS]
    return np.select(conds, names, default=C.OTHER_REGION)


def to_celldays(det: pd.DataFrame) -> pd.DataFrame:
    """Detections -> one row per (sensor, day/night, date, fine cell)."""
    d = det.assign(
        ix=np.floor(det["lon"] / C.CELL_FINE).astype("int64"),
        iy=np.floor(det["lat"] / C.CELL_FINE).astype("int64"),
    )
    cd = (d.groupby(["sensor", "daynight", "date", "ix", "iy"], observed=True)
            .agg(n_det=("frp", "size"), frp_sum=("frp", "sum"))
            .reset_index())
    return cd


def to_monthly_grid(cd: pd.DataFrame) -> pd.DataFrame:
    """Cell-days -> monthly totals on the coarse (~11 km) grid, per sensor and pass."""
    ratio = int(round(C.CELL_COARSE / C.CELL_FINE))
    cd = cd.assign(
        cx=cd["ix"] // ratio,
        cy=cd["iy"] // ratio,
        year=cd["date"].dt.year.astype("int16"),
        month=cd["date"].dt.month.astype("int8"),
        low=(cd["frp_sum"] < C.LOW_FRP_MW).astype("int32"),
    )
    g = (cd.groupby(["sensor", "daynight", "year", "month", "cx", "cy"], observed=True)
           .agg(celldays=("n_det", "size"), n_det=("n_det", "sum"),
                frp_sum=("frp_sum", "sum"), low_cd=("low", "sum"))
           .reset_index())
    lon_c = (g["cx"] + 0.5) * C.CELL_COARSE
    lat_c = (g["cy"] + 0.5) * C.CELL_COARSE
    g["region"] = assign_region(lon_c.to_numpy(), lat_c.to_numpy())
    return g


def in_window(df: pd.DataFrame, window: tuple[str, str]) -> pd.Series:
    key = df["year"].astype(int) * 100 + df["month"].astype(int)
    lo = int(window[0].replace("-", ""))
    hi = int(window[1].replace("-", ""))
    return (key >= lo) & (key <= hi)
