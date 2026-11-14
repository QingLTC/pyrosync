# PyroSync · Harmonized MODIS + VIIRS Burning-Activity Calendar

NASA Space Apps Challenge 2026 · Challenge: *Harmonization of MODIS and VIIRS Hot Spots*

PyroSync turns 20+ years of NASA FIRMS active-fire hotspots into one consistent
burning-activity calendar. MODIS (1 km, since 2000) and VIIRS (375 m, since 2012) see
the same fires differently, so their raw counts cannot be compared. PyroSync converts
VIIRS onto the MODIS Aqua scale, so the record runs from 2003 to today without a jump
when the satellites change. Users pick an area, see which months and years burned most,
and compare the current month with the historical median.

Case study: Indonesia, whose fires sent haze across Malaysia and Singapore in October 2026.

## How it works

1. **Clean**: keep detections of good quality (MODIS confidence ≥ 30, VIIRS not *low*).
   Remove static heat sources two ways: detections FIRMS flags as volcano / static land
   source / offshore, plus all detections in ~1 km cells flagged ≥ 50 times (volcano
   craters, smelters, gas flares), so near-real-time data without flags is treated the
   same. `scripts/check_static.py` showed why the threshold matters: a 1-flag rule
   deleted ~7 % of all detections that were as seasonal as real fires.
2. **Common unit**: count *fire cell-days* (a ~1 km cell with at least one detection on a
   day), per sensor and day/night pass. This removes most of the "one fire = many VIIRS
   pixels" effect.
3. **Layer A (ratio)**: Aqua ÷ S-NPP cell-days per region × calendar month × day/night,
   learned 2012-02 to 2020-12 (both at ~13:30 local overpass, before Aqua's orbit drift).
   Year bootstrap gives a 5-95 % range.
4. **Chain**: NOAA-20 → S-NPP (2018-05 to 2026-10), since S-NPP FIRMS data stopped on
   2 Nov 2026.
5. **Layer B (machine learning)**: gradient-boosted Poisson regression predicting Aqua
   cell-days per ~11 km cell-month from VIIRS features.
6. **Calendar**: Aqua before 2021, harmonized VIIRS from 2021.
7. **Validation**: fit on 2012-2017, score on 2018-2020 against real Aqua.

## Run it

```bash
pip install -r requirements.txt

# 1. Put the FIRMS archive CSVs (unzipped) in data/raw/
# 2. Build the harmonized tables (writes data/processed/)
python scripts/build_dataset.py
# 3. Start the app
streamlit run app.py
```

To test without real data: `python scripts/make_synthetic.py data/synthetic_raw`, then
`python scripts/build_dataset.py data/synthetic_raw data/synthetic_processed` and
`PYROSYNC_DATA=data/synthetic_processed streamlit run app.py`.
Synthetic numbers are invented and are never used in results.

## Project layout

| Path | What it does |
| --- | --- |
| `pyrosync/config.py` | All settings: grids, windows, regions, preset areas |
| `pyrosync/io.py` | Reads FIRMS CSVs, filters quality, merges SP and NRT, removes static sources |
| `pyrosync/grid.py` | Detections → fire cell-days → monthly ~11 km grid |
| `pyrosync/calibrate.py` | Layer A, chain, Layer B, hold-out validation |
| `pyrosync/series.py` | Area queries, calendar, anomaly vs. median |
| `scripts/build_dataset.py` | Runs the whole pipeline |
| `app.py` | Streamlit web app |

## Data

- NASA FIRMS: https://www.earthdata.nasa.gov/data/tools/firms
- FIRMS archive download (MODIS C6.1; VIIRS 375 m S-NPP, NOAA-20, NOAA-21):
  https://firms.modaps.eosdis.nasa.gov/download/

## Limitations

- Thick smoke and cloud hide fires from both sensors; the worst haze weeks may be under-counted.
- Smouldering peat fires are partly missed even by VIIRS.
- Harmonized values are estimates with a range, not measurements.
- The most recent months use near-real-time (NRT) data.
- Regions and preset areas are approximate rectangles.
- The calendar measures detected burning activity, not burned area.

## Use of Super Intelligence (SI)

This project used SI tools, as allowed by the NASA Space Apps rules:

- **Code**: parts of the code were generated with assistance from Claude (Anthropic) and
  then reviewed, run and edited by the team. Files containing SI-assisted code say so in
  their first line.
- **Research**: Claude helped find and summarize background sources (sensor differences,
  peat-fire detection, Indonesia's fire history).
- **Text**: Claude helped draft and edit this README and the project description.
- No SI-generated images, audio or video are used. No NASA logos or branding are used.

## Libraries

pandas, NumPy, scikit-learn, Streamlit, Plotly, PyArrow (all open source).