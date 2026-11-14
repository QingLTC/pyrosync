# Portions of this code were generated with assistance from Claude (Anthropic), an SI tool.
"""Write FAKE FIRMS-format CSVs for testing the pipeline before real data arrives.

    python scripts/make_synthetic.py data/synthetic_raw

The numbers are invented. Never use this output in the project's results.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
OUT = sys.argv[1] if len(sys.argv) > 1 else "data/synthetic_raw"
os.makedirs(OUT, exist_ok=True)

SEASON = np.array([0.3, 0.3, 0.4, 0.4, 0.6, 0.9, 1.5, 3.0, 4.5, 4.0, 1.2, 0.4])
YEAR_MULT = {2002: 2.0, 2004: 1.8, 2006: 3.0, 2009: 2.0, 2014: 1.8, 2015: 5.0,
             2019: 3.5, 2023: 2.8, 2026: 4.0, 2010: 0.4, 2016: 0.4, 2017: 0.6,
             2020: 0.6, 2021: 0.6, 2022: 0.5}
REGIONS = [  # name, lon0, lon1, lat0, lat1, weight, peat share
    ("Sumatra", 99.0, 105.5, -4.5, 2.0, 0.35, 0.5),
    ("Kalimantan", 109.5, 117.0, -3.5, 1.5, 0.40, 0.5),
    ("Papua", 132.0, 140.5, -8.5, -2.0, 0.10, 0.1),
    ("Other", 119.5, 125.0, -5.5, 1.0, 0.15, 0.1),
]
START, END = pd.Timestamp("2000-11-01"), pd.Timestamp("2026-11-13")
SP_END = pd.Timestamp("2026-06-30")
NRT_START = pd.Timestamp("2026-06-01")

days = pd.date_range(START, END, freq="D")
mult = (SEASON[days.month - 1] * np.array([YEAR_MULT.get(y, 1.0) for y in days.year]))
n_events = rng.poisson(4.0 * mult)
ev_day = np.repeat(np.arange(len(days)), n_events)
n_ev = len(ev_day)

reg_idx = rng.choice(len(REGIONS), size=n_ev, p=[r[5] for r in REGIONS])
lon = np.array([rng.uniform(REGIONS[i][1], REGIONS[i][2]) for i in reg_idx])
lat = np.array([rng.uniform(REGIONS[i][3], REGIONS[i][4]) for i in reg_idx])
peat = rng.random(n_ev) < np.array([REGIONS[i][6] for i in reg_idx])
size = rng.lognormal(0.0, 1.0, n_ev)
dur = rng.geometric(1 / 3, n_ev)

# expand events to event-days
rep = dur
e = np.repeat(np.arange(n_ev), rep)
offset = np.concatenate([np.arange(d) for d in dur])
day_i = ev_day[e] + offset
keep = day_i < len(days)
e, day_i = e[keep], day_i[keep]
ED = pd.DataFrame({"date": days[day_i], "lon": lon[e], "lat": lat[e],
                   "peat": peat[e], "size": size[e]})
print(f"{n_ev:,} events, {len(ED):,} event-days")


def detections(ed, n, jitter, frp_scale, day_prob):
    i = np.repeat(np.arange(len(ed)), n)
    d = ed.iloc[i].reset_index(drop=True)
    d["lon"] = d["lon"] + rng.normal(0, jitter, len(d))
    d["lat"] = d["lat"] + rng.normal(0, jitter, len(d))
    frp_mean = np.where(d["peat"], frp_scale * 0.35, frp_scale)
    d["frp"] = np.round(rng.lognormal(np.log(frp_mean), 0.8), 2)
    dp = np.where(d["peat"], day_prob - 0.15, day_prob)
    d["daynight"] = np.where(rng.random(len(d)) < dp, "D", "N")
    return d


def viirs(start, end, factor, sat_code):
    ed = ED[(ED["date"] >= start) & (ED["date"] <= end)]
    n = rng.poisson(factor * (1 + 3 * ed["size"].to_numpy()))
    d = detections(ed, n, 0.004, 8.0, 0.6)
    d["confidence"] = rng.choice(["n", "l", "h"], len(d), p=[0.85, 0.08, 0.07])
    d["satellite"] = sat_code
    d["instrument"] = "VIIRS"
    d["bright_ti4"] = np.round(rng.uniform(300, 367), 2)
    d["bright_ti5"] = np.round(rng.uniform(280, 310), 2)
    d["scan"], d["track"] = 0.4, 0.4
    return d


def modis(start, sat, drift_after="2021-01-01"):
    ed = ED[ED["date"] >= start]
    p = 1 - np.exp(-0.5 * ed["size"].to_numpy())
    p = p * np.where(ed["peat"], 0.6, 1.0)
    if sat == "Terra":
        p = p * 0.8
    p = p * np.where(ed["date"] >= pd.Timestamp(drift_after), 0.8, 1.0)
    seen = rng.random(len(ed)) < p
    ed = ed[seen]
    n = 1 + rng.poisson(0.6 * ed["size"].to_numpy())
    d = detections(ed, n, 0.006, 15.0, 0.7)
    d["confidence"] = rng.integers(0, 101, len(d))
    d["satellite"] = sat
    d["instrument"] = "MODIS"
    d["brightness"] = np.round(rng.uniform(310, 400), 1)
    d["bright_t31"] = np.round(rng.uniform(290, 310), 1)
    d["scan"], d["track"] = 1.0, 1.0
    return d


def gas_flares(start, end, instrument):
    dts = pd.date_range(start, end, freq="D")
    pts = [(117.5, -1.0), (108.0, -6.5), (131.0, -2.5)]
    rows = [{"date": t, "lon": x, "lat": y, "frp": 20.0, "daynight": "N"} for t in dts for x, y in pts]
    d = pd.DataFrame(rows)
    d["peat"], d["size"] = False, 1.0
    if instrument == "MODIS":
        d["confidence"], d["satellite"], d["instrument"] = 80, "Aqua", "MODIS"
        d["brightness"], d["bright_t31"] = 330.0, 300.0
        d["scan"] = d["track"] = 1.0
    else:
        d["confidence"], d["instrument"] = "n", "VIIRS"
        d["bright_ti4"], d["bright_ti5"] = 340.0, 300.0
        d["scan"] = d["track"] = 0.4
    d["type"] = 2
    return d


def write(df, name, family, nrt):
    df = df.copy()
    df["acq_date"] = df["date"].dt.strftime("%Y-%m-%d")
    df["acq_time"] = np.where(df["daynight"] == "D", 530, 1730)
    df["latitude"], df["longitude"] = df["lat"].round(5), df["lon"].round(5)
    if "type" not in df:
        df["type"] = 0
    df["type"] = df["type"].fillna(0).astype(int)
    if family == "MODIS":
        df["version"] = "6.1NRT" if nrt else "6.1"
        cols = ["latitude", "longitude", "brightness", "scan", "track", "acq_date", "acq_time",
                "satellite", "instrument", "confidence", "version", "bright_t31", "frp",
                "daynight", "type"]
    else:
        df["version"] = "2.0NRT" if nrt else 2
        cols = ["latitude", "longitude", "bright_ti4", "scan", "track", "acq_date", "acq_time",
                "satellite", "instrument", "confidence", "version", "bright_ti5", "frp",
                "daynight", "type"]
    if nrt:
        cols = [c for c in cols if c != "type"]
    df[cols].to_csv(os.path.join(OUT, name), index=False)
    print(f"  wrote {name}: {len(df):,} rows")


def split_write(df, token, family, idn, sat_fill=None):
    if sat_fill is not None:
        df = df.assign(satellite=sat_fill)
    write(df[df["date"] <= SP_END], f"fire_archive_{token}_{idn}.csv", family, nrt=False)
    write(df[df["date"] >= NRT_START], f"fire_nrt_{token}_{idn}.csv", family, nrt=True)


m = pd.concat([modis("2000-11-01", "Terra"), modis("2002-07-04", "Aqua"),
               gas_flares("2010-01-01", "2026-06-30", "MODIS")], ignore_index=True)
split_write(m, "M-C61", "MODIS", 846654)
sv = pd.concat([viirs("2012-01-20", "2026-11-02", 1.0, "N"),
                gas_flares("2012-01-20", "2026-06-30", "VIIRS").assign(satellite="N")])
split_write(sv, "SV-C2", "VIIRS", 846665)
split_write(viirs("2018-04-01", END, 1.05, "1"), "J1V-C2", "VIIRS", 846666)
split_write(viirs("2023-03-01", END, 1.05, "2"), "J2V-C2", "VIIRS", 846668)
