# PyroSync · Harmonized MODIS + VIIRS Burning-Activity Calendar

NASA Space Apps Challenge 2026 · Challenge: *Harmonization of MODIS and VIIRS Hot Spots*

PyroSync turns 20+ years of NASA FIRMS active-fire hotspots into one consistent
burning-activity calendar. MODIS (1 km, since 2000) and VIIRS (375 m, since 2012) see
the same fires differently, so their raw counts cannot be compared. PyroSync converts
VIIRS onto the MODIS Aqua scale, so the record runs from 2003 to today without a jump
when the satellites change. Users pick an area, see which months and years burned most,
and compare the current month with the historical median.

Case study: Indonesia, whose fires sent haze across Malaysia and Singapore in October 2026.

**Live app:** https://pyrosync.streamlit.app

![PyroSync web app: area selector, a headline ranking the 2026 fire season, four summary numbers and the burning calendar for Indonesia]
(<img width="1600" height="814" alt="image" src="https://github.com/user-attachments/assets/76428602-53b2-43d7-be65-6e7b2b17b01f" />)
 
*The app for Indonesia: the 2026 fire season ranks third since 2003, and every month since 2003 sits on one harmonized scale.*
 
![Line chart: gluing raw MODIS and VIIRS counts makes every year since 2012 look about 5× higher, while the PyroSync harmonized record stays continuous](<img width="1335" height="543" alt="image" src="https://github.com/user-attachments/assets/525779c7-fd8b-48d6-938e-348df4a75c2e" />
)
 
*Why harmonize: a naive splice of raw FIRMS counts (orange) invents a jump when VIIRS joins in 2012; the harmonized record (blue) does not.*

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

## Results (Indonesia, NASA FIRMS data through 2026-11-14)

Live app: https://pyrosync.streamlit.app

**Why harmonization is needed.** Over 2012–2020, VIIRS S-NPP reported 5.1× as many
detections as MODIS (Terra + Aqua) for the same area. Gluing the raw FIRMS records
together therefore makes every year since 2012 look about five times more fiery.

**Validation.** Fitted on 2012–2017, scored on monthly totals for 2018–2020 against real
MODIS Aqua fire cell-days (data the model never saw):

| Method | Monthly error (MAPE) | Correlation (r) | Total bias |
| --- | --- | --- | --- |
| Raw VIIRS detections | 700.0% | 0.997 | +701.3% |
| Raw VIIRS fire cell-days | 501.2% | 0.994 | +429.2% |
| **Layer A: ratio (default)** | **18.9%** | 0.991 | **+0.7%** |
| Layer B: machine learning | 23.0% | 0.994 | +0.8% |

The machine-learning model was not better than the simple ratio, so the app uses the
transparent ratio by default and offers the model as a check.

**Checking our own pipeline.** A first static-source filter removed 13.4% of all
detections. `scripts/check_static.py` showed that the removed "vegetation fire"
detections were as seasonal as real fires (65% in Aug–Oct vs. 69% overall), so the filter
was deleting real fires near industrial sites. The revised filter removes about 7%, and
the calibration bias fell from +4.5% to +0.7%.

**Case study: was 2026 Indonesia's worst fire season?** Total harmonized fire cell-days in
August–October:

| Year | Aug–Oct fire cell-days |
| --- | --- |
| 2015 | 73,440 |
| 2006 | 70,391 |
| **2026** | **53,697** (range 46,771–58,307) |
| 2004 | 52,630 |
| 2009 | 42,214 |
| 2019 | 38,358 |

2026 is the worst season since 2015 and third since 2003, behind 2015 and 2006; its
uncertainty range overlaps 2004. October 2026 alone was 3.8× the historical median for
October. A naive splice of raw counts would have called 2026 a record.

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

## Project Architecture

```
pyrosync/
├── data/
│   └── processed/             # Outputs the app loads
│       ├── .gitkeep
│       ├── factors_A.csv      # S-NPP → Aqua ratio factors with 5–95 % range
│       ├── factors_B.csv      # NOAA-20 → S-NPP ratio factors
│       ├── grid.parquet       # Harmonized monthly table on the ~11 km grid
│       ├── meta.json          # Data end date, row counts, settings used
│       ├── validation.csv     # Hold-out scores of each method
│       └── validation_ts.csv  # Hold-out monthly series (Indonesia)
├── pyrosync/                  # Core pipeline package
│   ├── __init__.py
│   ├── calibrate.py           # Layer A ratios, NOAA-20 → S-NPP chain, Layer B model, hold-out validation
│   ├── config.py              # All settings: grids, calibration windows, regions, preset areas
│   ├── grid.py                # Detections → ~1 km fire cell-days → monthly ~11 km grid
│   ├── io.py                  # Reads FIRMS CSVs, quality filter, SP/NRT merge, static-source removal
│   └── series.py              # Area queries, calendar, month vs. median, season ranking, naive splice
├── scripts/
│   ├── build_dataset.py       # Runs the whole pipeline: data/raw → data/processed
│   ├── check_static.py        # Diagnostic used to choose the static heat-source threshold
│   └── make_synthetic.py      # Fake FIRMS-format data for testing only (never used in results
├── .gitignore                 # Keeps raw and synthetic data out of git
├── LICENSE                    # MIT license
├── README.md                  # Project documentation
├── app.py                     # Streamlit web app: area picker, calendar, charts, map, downloads
└── requirements.txt           # Python dependencies
```

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


