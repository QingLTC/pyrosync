# Portions of this code were generated with assistance from Claude (Anthropic), an SI tool.
"""Diagnose the static heat-source filter on the raw FIRMS data.

    python scripts/check_static.py

For several thresholds it shows how many cells would count as static and what
would be removed. Real vegetation fires are seasonal (Aug-Oct in Indonesia);
static sources (gas flares, mills, volcanoes) burn all year. So if the type-0
detections a rule removes are as seasonal as fires overall, that rule is
deleting real fires.
"""
from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyrosync import config as C, io  # noqa: E402

RAW = sys.argv[1] if len(sys.argv) > 1 else C.RAW_DIR
det = io.load_raw_dir(RAW, remove_static=False)
det["key"] = io.cell_key(det)
det["season"] = det["date"].dt.month.isin([8, 9, 10])
counts = io.static_flag_counts(det)

print("\nFlag types in SP data (all sensors):")
print(det["type"].value_counts(dropna=False).rename("detections").to_string())

all_season = det.loc[det["type"].eq(0), "season"].mean() * 100
print(f"\nShare of ALL type-0 detections in Aug-Oct: {all_season:.0f}%")

rows = []
for n in [1, 2, 3, 5, 10, 20, 50]:
    keys = counts[counts >= n].index
    in_cells = det["key"].isin(keys)
    t0 = in_cells & det["type"].eq(0)
    rows.append({
        "min_flags": n,
        "static_cells": len(keys),
        "removed_total": int(in_cells.sum()),
        "removed_%": round(in_cells.mean() * 100, 1),
        "flagged_1_2_3": int((in_cells & det["type"].isin([1, 2, 3])).sum()),
        "type0_removed": int(t0.sum()),
        "type0_in_AugOct_%": round(det.loc[t0, "season"].mean() * 100, 0) if t0.any() else None,
        "NRT_removed": int((in_cells & det["src"].eq("NRT")).sum()),
    })
print("\nWhat each threshold would remove:")
print(pd.DataFrame(rows).to_string(index=False))

print("\nTop 15 static cells (most type 1/2/3 flags):")
top = counts.head(15).rename("flags").reset_index().rename(columns={"index": "key"})
first = det.drop_duplicates("key").set_index("key")[["lat", "lon"]]
top = top.join(first, on="key")
top["all_dets"] = top["key"].map(det["key"].value_counts())
print(top[["lat", "lon", "flags", "all_dets"]].round(3).to_string(index=False))

print("\nCells by number of flags:")
bins = pd.cut(counts, [0, 1, 2, 4, 9, 19, 49, 10**9],
              labels=["1", "2", "3-4", "5-9", "10-19", "20-49", "50+"])
print(bins.value_counts().sort_index().rename("cells").to_string())
