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
    cols = ["aqua_cd", "terra_cd", "snpp_cd", "n20_cd", "aqua_det", "terra_det", "snpp_det", "n20_det",
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


SEASON_MONTHS = (8, 9, 10)


def season_story(ms: pd.DataFrame, end: pd.Timestamp, months=SEASON_MONTHS) -> dict:
    """Rank the latest fire season (default Aug-Oct) against every earlier year."""
    year = end.year if end.month >= months[0] else end.year - 1
    s = (ms[ms["month"].isin(months)].groupby("year")
           .agg(total=("final", "sum"), lo=("final_lo", "sum"), hi=("final_hi", "sum")))
    if year not in s.index or s.loc[year, "total"] <= 0:
        return {}
    cur = float(s.loc[year, "total"])
    earlier = s[s.index < year]
    above = earlier[earlier["total"] >= cur]
    season_end = pd.Timestamp(year=year, month=months[-1], day=1) + pd.offsets.MonthEnd(0)
    return {
        "year": year,
        "months": months,
        "total": cur,
        "lo": float(s.loc[year, "lo"]),
        "hi": float(s.loc[year, "hi"]),
        "rank": int(len(above)) + 1,
        "n_years": int(len(earlier)) + 1,
        "highest_since": int(above.index.max()) if len(above) else None,
        "higher_years": [int(y) for y in above.sort_values("total", ascending=False).index],
        "partial": bool(end < season_end),
        "table": s,
    }


def naive_vs_harmonized(ms: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Annual naive record vs. harmonized record, both indexed to their 2003-2011 mean = 100.

    Naive = raw MODIS hotspot detections (Terra + Aqua, as FIRMS distributes them) until
    2011, then raw VIIRS detections (S-NPP, then NOAA-20 once S-NPP ends): what you get by
    gluing FIRMS counts together. Indexing removes the unit difference, so only the
    shape of each record is compared.
    Returns the table and the inflation factor: VIIRS detections / MODIS detections over
    the years both flew (window A).
    """
    key = ms["year"] * 100 + ms["month"]
    first_viirs = int(C.WIN_A[0][:4]) * 100 + 1
    last_snpp = int(C.SNPP_LAST_MONTH.replace("-", ""))
    modis_det = ms["aqua_det"] + ms["terra_det"]
    naive = np.where(key < first_viirs, modis_det,
                     np.where(key <= last_snpp, ms["snpp_det"], ms["n20_det"]))
    a = (ms.assign(naive=naive).groupby("year")[["naive", "final", "final_lo", "final_hi"]]
           .sum().reset_index())
    base = a[(a["year"] >= C.CALENDAR_START_YEAR) & (a["year"] < first_viirs // 100)]
    for col, ref in (("naive", "naive"), ("final", "final"), ("final_lo", "final"), ("final_hi", "final")):
        denom = base[ref].mean()
        a[col + "_idx"] = a[col] / denom * 100 if denom > 0 else np.nan
    win = (key >= int(C.WIN_A[0].replace("-", ""))) & (key <= int(C.WIN_A[1].replace("-", "")))
    m = modis_det[win].sum()
    inflation = float(ms.loc[win, "snpp_det"].sum() / m) if m > 0 else np.nan
    return a, inflation


def annual(ms: pd.DataFrame) -> pd.DataFrame:
    cols = ["final", "final_lo", "final_hi", "aqua_cd", "harm", "harm_lo", "harm_hi",
            "snpp_cd", "snpp_det"]
    a = ms.groupby("year")[cols].sum().reset_index()
    a["months"] = ms.groupby("year")["month"].count().to_numpy()
    return a