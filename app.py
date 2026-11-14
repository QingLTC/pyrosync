# Portions of this code were generated with assistance from Claude (Anthropic), an SI tool.
"""PyroSync web app: a harmonized MODIS + VIIRS burning-activity calendar.

    streamlit run app.py
"""
from __future__ import annotations

import calendar
import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pyrosync import config as C
from pyrosync import io, series

DATA_DIR = os.environ.get("PYROSYNC_DATA", C.PROCESSED_DIR)
MONTHS = [calendar.month_abbr[m] for m in range(1, 13)]

st.set_page_config(page_title="PyroSync · Harmonized fire calendar", layout="wide")


@st.cache_data(show_spinner="Loading harmonized data…")
def load():
    grid = io.load_table(os.path.join(DATA_DIR, "grid"))
    with open(os.path.join(DATA_DIR, "meta.json")) as f:
        meta = json.load(f)
    fac_a = pd.read_csv(os.path.join(DATA_DIR, "factors_A.csv"))
    val = pd.read_csv(os.path.join(DATA_DIR, "validation.csv"))
    val_ts = pd.read_csv(os.path.join(DATA_DIR, "validation_ts.csv"))
    return grid, meta, fac_a, val, val_ts


grid, meta, fac_a, val, val_ts = load()
end = pd.Timestamp(meta["data_end_date"])
last_ym = end.strftime("%Y-%m")
month_is_partial = end.day < end.days_in_month
last_complete = (end.to_period("M") - (1 if month_is_partial else 0))

# ---------------------------------------------------------------- sidebar
st.sidebar.title("PyroSync")
st.sidebar.caption("Harmonized MODIS + VIIRS burning-activity calendar")

aoi_name = st.sidebar.selectbox("Area of interest", list(C.PRESET_AOIS) + ["Custom rectangle"])
if aoi_name == "Custom rectangle":
    c1, c2 = st.sidebar.columns(2)
    lon0 = c1.number_input("West lon", value=101.0, step=0.5)
    lon1 = c2.number_input("East lon", value=104.0, step=0.5)
    lat0 = c1.number_input("South lat", value=-3.0, step=0.5)
    lat1 = c2.number_input("North lat", value=0.0, step=0.5)
    bbox = (min(lon0, lon1), max(lon0, lon1), min(lat0, lat1), max(lat0, lat1))
else:
    bbox = C.PRESET_AOIS[aoi_name]
aoi_label = aoi_name.strip()

method_label = st.sidebar.radio(
    "Harmonization method",
    ["Layer A: ratio (transparent)", "Layer B: machine learning"],
    help="A scales VIIRS by ratios learned when both sensors flew (2012-2020). "
         "B predicts the MODIS-equivalent count from VIIRS features.")
method = "B" if method_label.startswith("Layer B") else "A"

years_all = list(range(C.CALENDAR_START_YEAR, end.year + 1))
yr0, yr1 = st.sidebar.select_slider("Years", options=years_all,
                                    value=(years_all[0], years_all[-1]))

st.sidebar.markdown("**Compare a month with history**")
cmp_year = st.sidebar.selectbox("Year", years_all[::-1],
                                index=years_all[::-1].index(last_complete.year))
cmp_month = st.sidebar.selectbox("Month", range(1, 13), index=last_complete.month - 1,
                                 format_func=lambda m: calendar.month_name[m])

# ---------------------------------------------------------------- data for AOI
sub = series.select_bbox(grid, bbox)
ms = series.monthly_series(sub, last_ym, method)
ms_view = ms[(ms["year"] >= yr0) & (ms["year"] <= yr1)]
an = series.month_anomaly(ms, cmp_year, cmp_month)

# ---------------------------------------------------------------- header
st.title(f"Burning activity calendar · {aoi_label}")
st.caption(
    f"Satellite-detected fire activity in MODIS-Aqua-equivalent **fire cell-days** "
    f"(a ~1 km cell with at least one hotspot on a day). MODIS Aqua up to "
    f"{int(C.SWITCH_MONTH[:4]) - 1}, harmonized VIIRS from {C.SWITCH_MONTH}. "
    f"Data through {end.date()}. This measures detected burning activity, not burned area.")

story = series.season_story(ms, end)
if story:
    mlab = f"{MONTHS[story['months'][0] - 1]}–{MONTHS[story['months'][-1] - 1]}"
    head = (f"**{story['year']} fire season ({mlab}{', so far' if story['partial'] else ''}): "
            f"{story['total']:,.0f} fire cell-days**, #{story['rank']} of {story['n_years']} "
            f"since {C.CALENDAR_START_YEAR}")
    if story["highest_since"]:
        head += f", the highest since {story['highest_since']}."
        if len(story["higher_years"]) <= 3:
            head += f" Only {', '.join(map(str, story['higher_years']))} burned more."
    else:
        head += ", the highest in the record."
    with st.container(border=True):
        st.markdown(head)
        st.caption(f"Range {story['lo']:,.0f} to {story['hi']:,.0f} from calibration uncertainty. "
                   "Earlier years use MODIS Aqua; recent years use VIIRS converted to the same scale.")

if an:
    sel = f"{calendar.month_name[cmp_month]} {cmp_year}"
    partial = (pd.Period(f"{cmp_year}-{cmp_month:02d}", "M") == end.to_period("M")) and month_is_partial
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"{sel}{' (partial)' if partial else ''}", f"{an['value']:,.0f}",
              help=f"Range {an['lo']:,.0f} to {an['hi']:,.0f} (calibration uncertainty)")
    m2.metric("vs. historical median", f"{an['ratio']:.1f}×" if np.isfinite(an["ratio"]) else "n/a",
              help=f"Median of {calendar.month_name[cmp_month]} in earlier years: {an['median']:,.0f}")
    m3.metric("Percentile", f"{an['percentile']:.0f}th")
    m4.metric("Rank", f"#{an['rank']} of {an['n_years']}",
              help=f"Rank among all {calendar.month_name[cmp_month]}s since {C.CALENDAR_START_YEAR}")

tab_cal, tab_why, tab_trend, tab_season, tab_map, tab_method, tab_dl = st.tabs(
    ["Calendar", "Why harmonize?", "Long-term trend", "Season vs. normal", "Map",
     "Method & validation", "Download"])

# ---------------------------------------------------------------- why harmonize
with tab_why:
    nv, inflation = series.naive_vs_harmonized(ms)
    nv = nv[(nv["year"] >= yr0) & (nv["year"] <= yr1)]
    first_viirs = int(C.WIN_A[0][:4])
    if np.isfinite(inflation):
        st.markdown(f"#### Gluing raw FIRMS counts together makes every year since {first_viirs} "
                    f"look about {inflation:.1f}× more fiery")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(nv["year"]) + list(nv["year"])[::-1],
                             y=list(nv["final_hi_idx"]) + list(nv["final_lo_idx"])[::-1],
                             fill="toself", line=dict(width=0), fillcolor="rgba(0,114,178,0.18)",
                             name="Calibration range", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=nv["year"], y=nv["naive_idx"], mode="lines+markers",
                             line=dict(color="#D55E00", dash="dash", width=2.5),
                             name="Naive: raw MODIS detections, then raw VIIRS detections"))
    fig.add_trace(go.Scatter(x=nv["year"], y=nv["final_idx"], mode="lines+markers",
                             line=dict(color="#0072B2", width=3), name="PyroSync harmonized record"))
    fig.add_hline(y=100, line_color="#BBBBBB", line_width=1)
    fig.add_vline(x=first_viirs - 0.5, line_dash="dash", line_color="#999999",
                  annotation_text="VIIRS joins", annotation_position="top")
    fig.update_layout(height=480, yaxis_title=f"Index ({C.CALENDAR_START_YEAR}–{first_viirs - 1} average = 100)",
                      legend=dict(orientation="h", y=-0.2), margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Both records are scaled so their {C.CALENDAR_START_YEAR}–{first_viirs - 1} average is 100, "
               "so only their shape is compared. The naive record jumps when VIIRS joins because VIIRS "
               "sees smaller fires and splits one fire into several 375 m pixels. "
               f"Over {C.WIN_A[0][:4]}–{C.WIN_A[1][:4]}, VIIRS S-NPP reported {inflation:.1f}× as many "
               "detections as MODIS (Terra + Aqua) for the same area.")

# ---------------------------------------------------------------- calendar
with tab_cal:
    mat = series.calendar_matrix(ms_view)
    mat.columns = MONTHS
    fig = px.imshow(mat, color_continuous_scale="cividis", aspect="auto",
                    labels=dict(x="Month", y="Year", color="Fire cell-days"),
                    text_auto=".0f")
    fig.update_traces(textfont_size=9)
    fig.update_yaxes(autorange="reversed", dtick=1)
    fig.update_layout(height=max(420, 26 * len(mat)), margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, width="stretch")
    st.caption("Colour = harmonized fire cell-days per month (colour-blind-safe *cividis* scale). "
               f"Rows from {C.SWITCH_MONTH[:4]} on use harmonized VIIRS."
               + (f" {MONTHS[end.month - 1]} {end.year} is partial." if month_is_partial else ""))

# ---------------------------------------------------------------- trend
with tab_trend:
    ann = series.annual(ms_view)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(ann["year"]) + list(ann["year"])[::-1],
                             y=list(ann["final_hi"]) + list(ann["final_lo"])[::-1],
                             fill="toself", line=dict(width=0), fillcolor="rgba(0,114,178,0.18)",
                             name="Calibration range", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=ann["year"], y=ann["final"], mode="lines+markers",
                             line=dict(color="#0072B2", width=3), name="Harmonized record"))
    fig.add_trace(go.Scatter(x=ann["year"], y=ann["aqua_cd"], mode="lines",
                             line=dict(color="#555555", dash="dot"), name="MODIS Aqua (raw)"))
    viirs_years = ann[ann["snpp_cd"] > 0]
    fig.add_trace(go.Scatter(x=viirs_years["year"], y=viirs_years["snpp_cd"], mode="lines",
                             line=dict(color="#D55E00", dash="dash"),
                             name="VIIRS S-NPP cell-days (not harmonized)"))
    fig.add_trace(go.Scatter(x=viirs_years["year"], y=viirs_years["harm"], mode="lines",
                             line=dict(color="#009E73"), name="VIIRS harmonized (overlap check)"))
    fig.add_vline(x=int(C.SWITCH_MONTH[:4]) - 0.5, line_dash="dash", line_color="#999999",
                  annotation_text="switch to VIIRS", annotation_position="top")
    fig.update_layout(height=480, yaxis_title="Fire cell-days per year",
                      legend=dict(orientation="h", y=-0.2), margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, width="stretch")
    st.caption("Where Aqua and harmonized VIIRS overlap (2012-2020) the two lines should agree; "
               "un-harmonized VIIRS sits far higher because it sees smaller fires."
               + (f" {end.year} is a partial year." if end.month < 12 or month_is_partial else ""))

# ---------------------------------------------------------------- season vs normal
with tab_season:
    base = ms[(ms["year"] < cmp_year) & (ms["year"] >= C.CALENDAR_START_YEAR)]
    q = base.groupby("month")["final"].quantile([0.25, 0.5, 0.75]).unstack()
    cur = ms[ms["year"] == cmp_year]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=MONTHS + MONTHS[::-1], y=list(q[0.75]) + list(q[0.25])[::-1],
                             fill="toself", line=dict(width=0), fillcolor="rgba(120,120,120,0.2)",
                             name=f"Middle 50 % of {C.CALENDAR_START_YEAR}-{cmp_year - 1}",
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=MONTHS, y=q[0.5], line=dict(color="#555555", dash="dot"),
                             name="Historical median"))
    fig.add_trace(go.Scatter(x=[MONTHS[m - 1] for m in cur["month"]], y=cur["final"],
                             mode="lines+markers", line=dict(color="#D55E00", width=3),
                             name=str(cmp_year)))
    fig.update_layout(height=440, yaxis_title="Fire cell-days per month",
                      legend=dict(orientation="h", y=-0.2), margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- map
with tab_map:
    map_year = st.selectbox("Year to map", years_all[::-1], key="map_year")
    col = "aqua_cd" if map_year < int(C.SWITCH_MONTH[:4]) else ("harm_B" if method == "B" else "harm_A")
    cells = (sub[sub["year"] == map_year].groupby(["cx", "cy"])[col].sum().reset_index())
    cells = cells[cells[col] > 0]
    cells["lon"] = (cells["cx"] + 0.5) * C.CELL_COARSE
    cells["lat"] = (cells["cy"] + 0.5) * C.CELL_COARSE
    if cells.empty:
        st.info("No detections in this area and year.")
    else:
        scatter = getattr(px, "scatter_map", None) or px.scatter_mapbox
        kw = {} if scatter is getattr(px, "scatter_map", None) else {"mapbox_style": "carto-positron"}
        fig = scatter(cells, lat="lat", lon="lon", color=col, size=col, size_max=12,
                      color_continuous_scale="cividis", zoom=4, height=560,
                      labels={col: "Fire cell-days"}, **kw)
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Each dot = one ~11 km cell; {map_year} total.")

# ---------------------------------------------------------------- method
with tab_method:
    st.markdown(f"""
**Problem.** MODIS (1 km pixels, since 2000) and VIIRS (375 m, since 2012) see the same fires
differently: VIIRS detects smaller and cooler fires and splits one fire into several pixels, so
raw counts jump when the record moves from MODIS to VIIRS.

**Steps.**
1. Keep good-quality vegetation-fire detections (MODIS confidence ≥ {C.MODIS_MIN_CONF}, VIIRS not
   *low*). Static heat sources are removed in two ways: detections FIRMS flags as volcano, static
   land source or offshore, plus every detection in ~1 km cells flagged at least
   {C.STATIC_MIN_FLAGS} times (volcano craters, smelters, gas flares), so recent near-real-time data,
   which carries no flag, is treated like older data.
2. Count **fire cell-days**: a ~1 km cell with ≥ 1 detection on a day, per sensor and day/night pass.
3. **Layer A**: ratio of Aqua to S-NPP cell-days per region × calendar month × day/night, learned in
   {C.WIN_A[0]} to {C.WIN_A[1]}, when Aqua and S-NPP both flew at ~13:30 and before Aqua's orbit drift.
   Uncertainty = 5-95 % range from resampling whole years.
4. **Chain**: NOAA-20 is aligned to S-NPP ({C.WIN_B[0]} to {C.WIN_B[1]}) because S-NPP stopped
   delivering FIRMS data on 2 Nov 2026.
5. **Layer B**: gradient-boosted Poisson regression predicting Aqua cell-days per ~11 km cell-month
   from VIIRS day/night counts, FRP, share of low-intensity detections, month, region and location.
6. Calendar = Aqua before {C.SWITCH_MONTH}, harmonized VIIRS after.
""")
    st.subheader("Hold-out validation")
    st.caption(f"Fitted on {C.WIN_A_TRAIN[0]} to {C.WIN_A_TRAIN[1]}, scored on monthly totals in "
               f"{C.WIN_A_TEST[0]} to {C.WIN_A_TEST[1]} against real MODIS Aqua. "
               "MAPE = mean absolute % error (months with ≥ 10 cell-days); bias = total over/under-count.")
    st.dataframe(val, hide_index=True, width="stretch")
    vt = val_ts.copy()
    vt["date"] = pd.to_datetime(dict(year=vt["year"], month=vt["month"], day=1))
    keep = [c for c in ["Aqua (actual)", "Layer A (ratio)", "Layer B (ML)", "raw VIIRS cell-days"] if c in vt]
    fig = px.line(vt, x="date", y=keep, labels={"value": "Fire cell-days", "variable": ""})
    fig.update_layout(height=380, legend=dict(orientation="h", y=-0.25), margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")
    with st.expander("Layer A factors (Aqua cell-days per S-NPP cell-day)"):
        st.dataframe(fac_a.round(3), hide_index=True, width="stretch")
    st.markdown("""
**Limitations.** Thick smoke and cloud hide fires from both sensors, so the worst haze weeks may be
under-counted. Smouldering peat fires are partly missed even by VIIRS. Harmonized values are model
estimates with a range, not measurements. The most recent months use near-real-time data.
""")

# ---------------------------------------------------------------- download
with tab_dl:
    out = ms_view[["year", "month", "source", "final", "final_lo", "final_hi",
                   "aqua_cd", "snpp_cd", "n20_cd", "snpp_det"]].rename(columns={
        "final": "harmonized_fire_celldays", "final_lo": "range_low", "final_hi": "range_high",
        "aqua_cd": "modis_aqua_celldays_raw", "snpp_cd": "viirs_snpp_celldays_raw",
        "n20_cd": "viirs_noaa20_celldays_raw", "snpp_det": "viirs_snpp_detections_raw"})
    st.download_button("Monthly series (CSV)", out.to_csv(index=False).encode(),
                       file_name=f"pyrosync_{aoi_label.replace(' ', '_')}_{yr0}-{yr1}_monthly.csv",
                       mime="text/csv")
    cells_out = sub[(sub["year"] >= yr0) & (sub["year"] <= yr1)].copy()
    cells_out["lon_center"] = ((cells_out["cx"] + 0.5) * C.CELL_COARSE).round(2)
    cells_out["lat_center"] = ((cells_out["cy"] + 0.5) * C.CELL_COARSE).round(2)
    cells_out = cells_out.drop(columns=["cx", "cy"])
    st.download_button("Hotspots per ~11 km cell and month (CSV)", cells_out.to_csv(index=False).encode(),
                       file_name=f"pyrosync_{aoi_label.replace(' ', '_')}_{yr0}-{yr1}_cells.csv",
                       mime="text/csv")
    st.caption("Raw counts per sensor plus harmonized values for every grid cell and month in your selection.")

st.divider()
st.caption("Data: NASA FIRMS active fire products (MODIS C6.1; VIIRS 375 m S-NPP, NOAA-20, NOAA-21). "
           "Built for the NASA Space Apps Challenge 2026. Parts of the code were written with SI "
           "assistance (Claude, Anthropic); see README.")