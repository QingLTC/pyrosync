# Portions of this code were generated with assistance from Claude (Anthropic), an SI tool.
"""Area-of-interest queries on the harmonized grid: monthly series, calendar, anomalies."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def select_bbox(grid: pd.DataFrame, bbox) -> pd.DataFrame:
    lon0, lon1, lat0, lat1 = bbox
    lon_c = (grid["cx"] + 0.5) * C.CELL_COARSE
    lat_c = (grid["cy"] + 0.5) * C.CELL_COARSE
    return grid[(lon_c >= lon0) & (lon_c <= lon1) & (lat_c >= lat0) & (lat_c <= lat1)]


def monthly_series(sub: pd.DataFrame, last_ym: str, method: str = "A") -> pd.DataFrame:
    """One row per month from CALENDAR_START_YEAR to last_ym.

    final = Aqua before SWITCH_MONTH, harmonized VIIRS from SWITCH_MONTH on.
    """
    cols = ["aqua_cd", "terra_cd", "snpp_cd", "n20_cd", "snpp_det",
            "harm_A", "harm_A_lo", "harm_A_hi", "harm_B"]
    m = sub.groupby(["year", "month"])[cols].sum()
    start = pd.Period(f"{C.CALENDAR_START_YEAR}-01", "M")
    end = pd.Period(last_ym, "M")
    idx = pd.period_range(start, end, freq="M")
    full = pd.MultiIndex.from_arrays([idx.year, idx.month], names=["year", "month"])
    m = m.reindex(full, fill_value=0).reset_index()

    if method == "B":
        rel_lo = np.where(m["harm_A"] > 0, m["harm_A_lo"] / m["harm_A"].where(m["harm_A"] > 0, 1), 1)
        rel_hi = np.where(m["harm_A"] > 0, m["harm_A_hi"] / m["harm_A"].where(m["harm_A"] > 0, 1), 1)
        m["harm"], m["harm_lo"], m["harm_hi"] = m["harm_B"], m["harm_B"] * rel_lo, m["harm_B"] * rel_hi
    else:
        m["harm"], m["harm_lo"], m["harm_hi"] = m["harm_A"], m["harm_A_lo"], m["harm_A_hi"]

    switch = int(C.SWITCH_MONTH.replace("-", ""))
    use_viirs = (m["year"] * 100 + m["month"]) >= switch
    m["final"] = np.where(use_viirs, m["harm"], m["aqua_cd"])
    m["final_lo"] = np.where(use_viirs, m["harm_lo"], m["aqua_cd"])
    m["final_hi"] = np.where(use_viirs, m["harm_hi"], m["aqua_cd"])
    m["source"] = np.where(use_viirs, "VIIRS, harmonized", "MODIS Aqua")
    m["date"] = pd.to_datetime(dict(year=m["year"], month=m["month"], day=1))
    return m


def calendar_matrix(ms: pd.DataFrame, col: str = "final") -> pd.DataFrame:
    """Years (rows) x months (columns)."""
    return ms.pivot(index="year", columns="month", values=col).sort_index()


def month_anomaly(ms: pd.DataFrame, year: int, month: int) -> dict:
    """Compare one month with the same calendar month in all earlier years."""
    cur = ms[(ms["year"] == year) & (ms["month"] == month)]
    base = ms[(ms["month"] == month) & (ms["year"] < year)
              & (ms["year"] >= C.CALENDAR_START_YEAR)]
    if cur.empty or base.empty:
        return {}
    value = float(cur["final"].iloc[0])
    med = float(base["final"].median())
    allyears = pd.concat([base[["year", "final"]], cur[["year", "final"]]])
    rank = int((allyears["final"] > value).sum()) + 1
    return {
        "value": value,
        "lo": float(cur["final_lo"].iloc[0]),
        "hi": float(cur["final_hi"].iloc[0]),
        "median": med,
        "ratio": value / med if med > 0 else np.nan,
        "percentile": float((base["final"] < value).mean() * 100),
        "rank": rank,
        "n_years": len(allyears),
    }


def annual(ms: pd.DataFrame) -> pd.DataFrame:
    cols = ["final", "final_lo", "final_hi", "aqua_cd", "harm", "harm_lo", "harm_hi",
            "snpp_cd", "snpp_det"]
    a = ms.groupby("year")[cols].sum().reset_index()
    a["months"] = ms.groupby("year")["month"].count().to_numpy()
    return a
