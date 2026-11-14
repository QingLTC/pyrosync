# Portions of this code were generated with assistance from Claude (Anthropic), an SI tool.
"""Build everything the web app needs from the raw FIRMS CSVs.

    python scripts/build_dataset.py            # reads data/raw, writes data/processed

Outputs (data/processed):
    grid            harmonized monthly table on the ~11 km grid (what the app loads)
    factors_A.csv   S-NPP -> Aqua ratio factors with 5-95 % range
    factors_B.csv   NOAA-20 -> S-NPP ratio factors
    validation.csv  hold-out scores of each method
    validation_ts.csv  hold-out monthly series (Indonesia)
    meta.json       data end date, row counts, settings used
"""
from __future__ import annotations

import json
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyrosync import calibrate, config as C, grid, io  # noqa: E402


def main(raw_dir=C.RAW_DIR, out_dir=C.PROCESSED_DIR):
    t0 = time.time()
    print("1/6 reading raw FIRMS files")
    det = io.load_raw_dir(raw_dir)
    print(f"    {len(det):,} detections kept, {det['date'].min().date()} .. {det['date'].max().date()}")
    print(det.groupby("sensor").size().rename("detections").to_string())

    print("2/6 detections -> fire cell-days (~1 km)")
    cd = grid.to_celldays(det)
    print(f"    {len(cd):,} cell-days")

    print("3/6 cell-days -> monthly ~11 km grid")
    mg = grid.to_monthly_grid(cd)
    print(f"    {len(mg):,} rows")

    print("4/6 Layer A ratio factors + chain")
    fac_a = calibrate.fit_ratio(mg, C.AQUA, C.SNPP, C.WIN_A)
    fac_b = calibrate.fit_ratio(mg, C.SNPP, C.N20, C.WIN_B, n_boot=50)
    harm_a = calibrate.apply_ratio_chain(mg, fac_a, fac_b)

    print("5/6 Layer B model")
    feat = calibrate.build_features(mg)
    model, corr = calibrate.fit_ml(feat, C.WIN_A)
    harm_b = calibrate.predict_ml(model, corr, feat)

    print("6/6 validation + writing outputs")
    table, ts = calibrate.validate(mg)
    print(table.to_string(index=False))

    raw = (mg.groupby(calibrate.CELL + ["sensor"], observed=True)[["celldays", "n_det"]].sum()
             .unstack("sensor", fill_value=0))
    names = {C.AQUA: "aqua", C.TERRA: "terra", C.SNPP: "snpp", C.N20: "n20", C.N21: "n21"}
    out = pd.DataFrame(index=raw.index)
    for s, short in names.items():
        out[f"{short}_cd"] = raw["celldays"][s] if s in raw["celldays"] else 0
        out[f"{short}_det"] = raw["n_det"][s] if s in raw["n_det"] else 0
    out = out.reset_index()
    out = out.merge(harm_a, on=calibrate.CELL, how="left").merge(harm_b, on=calibrate.CELL, how="left")
    # Months where only N20 exists (after S-NPP ends): Layer B falls back to Layer A.
    out["harm_B"] = out["harm_B"].fillna(out["harm_A"])
    out = out.fillna(0)
    for c in out.columns:
        if c.endswith(("_cd", "_det")):
            out[c] = out[c].astype("int32")
        elif c.startswith("harm"):
            out[c] = out[c].astype("float32").round(3)

    os.makedirs(out_dir, exist_ok=True)
    path = io.save_table(out, os.path.join(out_dir, "grid"))
    fac_a.to_csv(os.path.join(out_dir, "factors_A.csv"), index=False)
    fac_b.to_csv(os.path.join(out_dir, "factors_B.csv"), index=False)
    table.to_csv(os.path.join(out_dir, "validation.csv"), index=False)
    ts.to_csv(os.path.join(out_dir, "validation_ts.csv"), index=False)
    meta = {
        "data_end_date": str(det["date"].max().date()),
        "data_start_date": str(det["date"].min().date()),
        "detections": int(len(det)),
        "detections_by_sensor": det.groupby("sensor").size().astype(int).to_dict(),
        "grid_rows": int(len(out)),
        "settings": {k: getattr(C, k) for k in
                     ["CELL_FINE", "CELL_COARSE", "WIN_A", "WIN_B", "SWITCH_MONTH",
                      "MODIS_MIN_CONF", "MIN_GROUP_COUNT"]},
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"done in {time.time() - t0:.0f}s -> {path}")


if __name__ == "__main__":
    main(*sys.argv[1:])
