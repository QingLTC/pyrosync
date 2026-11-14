# Portions of this code were generated with assistance from Claude (Anthropic), an SI tool.
"""Project-wide settings. Change values here, not inside the pipeline code."""

# --- Grids -----------------------------------------------------------------
CELL_FINE = 0.01    # ~1.1 km: unit for "fire cell-days" (about one MODIS pixel)
CELL_COARSE = 0.1   # ~11 km: unit stored for the web app (area-of-interest queries)

# --- Filtering ---------------------------------------------------------------
MODIS_MIN_CONF = 30     # MODIS confidence is 0-100; below 30 = low confidence
VIIRS_DROP_CONF = ("l",)  # VIIRS confidence l/n/h (or low/nominal/high); drop low
LOW_FRP_MW = 10.0       # cell-days with total FRP under this count as "low-intensity"
STATIC_MIN_FLAGS = 50   # a ~1 km cell is a persistent static source if SP data flagged
                        # at least this many detections in it as type 1/2/3. Chosen with
                        # scripts/check_static.py: at 1 flag, 65 % of the type-0 detections
                        # removed fell in Aug-Oct (as seasonal as real fires, 69 %);
                        # at 50 flags only 37 %.

# --- Sensors ---------------------------------------------------------------
# Internal sensor names used everywhere in the pipeline.
AQUA, TERRA = "MODIS_A", "MODIS_T"
SNPP, N20, N21 = "VIIRS_SNPP", "VIIRS_N20", "VIIRS_N21"
ALL_SENSORS = [TERRA, AQUA, SNPP, N20, N21]

# --- Harmonization windows (inclusive months, "YYYY-MM") ---------------------
# Window A: learn S-NPP -> Aqua. Ends before the Aqua orbit drift (from 2021).
WIN_A = ("2012-02", "2020-12")
# Hold-out split inside window A, used only for validation.
WIN_A_TRAIN = ("2012-02", "2017-12")
WIN_A_TEST = ("2018-01", "2020-12")
# Window B: learn NOAA-20 -> S-NPP. S-NPP FIRMS production stopped 2026-11-02.
WIN_B = ("2018-05", "2026-10")
SNPP_LAST_MONTH = "2026-10"   # last full month with S-NPP data

# From this month on, the calendar uses harmonized VIIRS instead of Aqua.
SWITCH_MONTH = "2021-01"
CALENDAR_START_YEAR = 2003    # first full year of Aqua

# --- Calibration -------------------------------------------------------------
MIN_GROUP_COUNT = 30   # min VIIRS cell-days a group needs for its own ratio
N_BOOTSTRAP = 200      # year-resampling draws for the uncertainty range
RANDOM_SEED = 0

# --- Calibration regions (approximate bounding boxes, checked in order) ------
# (name, lon_min, lon_max, lat_min, lat_max)
REGIONS = [
    ("Sumatra", 95.0, 106.2, -6.0, 6.0),
    ("Kalimantan", 108.5, 119.3, -4.3, 4.5),
    ("Papua", 130.5, 141.1, -9.2, 1.0),
]
OTHER_REGION = "Other"   # Java, Sulawesi, Nusa Tenggara, Maluku, ...

# --- Preset areas of interest for the app (approximate rectangles) -----------
# (lon_min, lon_max, lat_min, lat_max)
PRESET_AOIS = {
    "Indonesia (all)": (94.5, 141.5, -11.5, 6.5),
    "Sumatra": (95.0, 106.2, -6.0, 6.0),
    "  Riau": (100.0, 104.0, -1.2, 2.5),
    "  Jambi": (101.1, 104.5, -2.8, -0.7),
    "  South Sumatra": (102.0, 106.2, -4.9, -1.6),
    "Kalimantan": (108.5, 119.3, -4.3, 4.5),
    "  West Kalimantan": (108.5, 114.2, -3.1, 2.1),
    "  Central Kalimantan": (110.7, 116.0, -3.6, 0.8),
    "  South Kalimantan": (114.3, 116.6, -4.2, -1.3),
    "Papua": (130.5, 141.1, -9.2, 1.0),
}

# --- Paths -------------------------------------------------------------------
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"