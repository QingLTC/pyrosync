# Portions of this code were generated with assistance from Claude (Anthropic), an SI tool.
"""Harmonize VIIRS fire cell-days onto the MODIS Aqua scale.

Layer A (ratio):   k = sum(Aqua cell-days) / sum(VIIRS cell-days) per region x calendar
                   month x day/night, learned in window A; sparse groups fall back to
                   region x day/night, then day/night. Uncertainty = year bootstrap.
Chain:             NOAA-20 -> S-NPP (window B), then S-NPP -> Aqua (window A).
Layer B (ML):      gradient-boosted Poisson regression predicting Aqua cell-days per
                   ~11 km cell-month from VIIRS features (day/night counts, FRP,
                   share of low-intensity cell-days, month, region, location).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from . import config as C
from .grid import in_window

KEYS = ["region", "month", "daynight"]
CELL = ["year", "month", "cx", "cy", "region"]
_LEVELS = [["region", "month", "daynight"], ["region", "daynight"], ["daynight"]]


# ----------------------------------------------------------------------------
# Layer A: ratio factors
# ----------------------------------------------------------------------------
def _yearly_totals(mg, num, den, window):
    sub = mg[in_window(mg, window) & mg["sensor"].isin([num, den])]
    t = (sub.groupby(KEYS + ["year", "sensor"], observed=True)["celldays"].sum()
            .unstack("sensor", fill_value=0))
    for s in (num, den):
        if s not in t:
            t[s] = 0
    return t[[num, den]].rename(columns={num: "num", den: "den"}).reset_index()


def _full_grid():
    regions = [r[0] for r in C.REGIONS] + [C.OTHER_REGION]
    idx = pd.MultiIndex.from_product([regions, range(1, 13), ["D", "N"]], names=KEYS)
    return idx.to_frame(index=False)


def _ratio_from_totals(tot):
    out = _full_grid()
    out["k"] = np.nan
    out["level"] = ""
    for lvl in _LEVELS:
        s = tot.groupby(lvl)[["num", "den"]].sum()
        s = s[s["den"] >= C.MIN_GROUP_COUNT]
        s["k_lvl"] = s["num"] / s["den"]
        m = out.merge(s["k_lvl"].reset_index(), on=lvl, how="left")
        fill = out["k"].isna() & m["k_lvl"].notna()
        out.loc[fill, "k"] = m.loc[fill, "k_lvl"]
        out.loc[fill, "level"] = "+".join(lvl)
    out["k"] = out["k"].fillna(1.0)   # no data at all: leave counts unchanged
    out.loc[out["level"] == "", "level"] = "none"
    return out


def fit_ratio(mg, num, den, window, n_boot=C.N_BOOTSTRAP):
    """Ratio factors with a 5-95 % range from resampling whole years."""
    tot = _yearly_totals(mg, num, den, window)
    fac = _ratio_from_totals(tot)
    years = tot["year"].unique()
    rng = np.random.default_rng(C.RANDOM_SEED)
    draws = []
    by_year = {y: g for y, g in tot.groupby("year")}
    for _ in range(n_boot):
        pick = rng.choice(years, size=len(years), replace=True)
        boot = pd.concat([by_year[y] for y in pick], ignore_index=True)
        draws.append(_ratio_from_totals(boot)["k"].to_numpy())
    draws = np.vstack(draws)
    fac["k_lo"] = np.percentile(draws, 5, axis=0)
    fac["k_hi"] = np.percentile(draws, 95, axis=0)
    return fac


def apply_ratio_chain(mg, fac_a, fac_b):
    """Harmonized (Aqua-equivalent) cell-days per coarse cell-month.

    Months up to SNPP_LAST_MONTH use S-NPP x k_A; later months use NOAA-20 x k_B x k_A.
    """
    last = int(C.SNPP_LAST_MONTH.replace("-", ""))
    key = mg["year"].astype(int) * 100 + mg["month"].astype(int)
    first = int(C.WIN_A[0].replace("-", ""))

    snpp = mg[(mg["sensor"] == C.SNPP) & (key >= first) & (key <= last)]
    snpp = snpp.merge(fac_a[KEYS + ["k", "k_lo", "k_hi"]], on=KEYS, how="left")
    snpp["h"], snpp["h_lo"], snpp["h_hi"] = (snpp["celldays"] * snpp[c]
                                             for c in ("k", "k_lo", "k_hi"))

    n20 = mg[(mg["sensor"] == C.N20) & (key > last)]
    n20 = n20.merge(fac_b[KEYS + ["k"]].rename(columns={"k": "kb"}), on=KEYS, how="left")
    n20 = n20.merge(fac_a[KEYS + ["k", "k_lo", "k_hi"]], on=KEYS, how="left")
    base = n20["celldays"] * n20["kb"]
    n20["h"], n20["h_lo"], n20["h_hi"] = (base * n20[c] for c in ("k", "k_lo", "k_hi"))

    both = pd.concat([snpp, n20], ignore_index=True)
    out = (both.groupby(CELL, observed=True)[["h", "h_lo", "h_hi"]].sum()
               .rename(columns={"h": "harm_A", "h_lo": "harm_A_lo", "h_hi": "harm_A_hi"})
               .reset_index())
    return out


# ----------------------------------------------------------------------------
# Layer B: gradient-boosted regression
# ----------------------------------------------------------------------------
_REGION_CODES = {name: i for i, name in enumerate([r[0] for r in C.REGIONS] + [C.OTHER_REGION])}
FEATURES = ["cd_D", "cd_N", "n_det", "frp_sum", "low_share", "m_sin", "m_cos",
            "region_code", "cx", "cy"]


def build_features(mg, viirs=C.SNPP):
    v = mg[mg["sensor"] == viirs]
    cd = (v.pivot_table(index=CELL, columns="daynight", values="celldays",
                        aggfunc="sum", fill_value=0)
            .reindex(columns=["D", "N"], fill_value=0)
            .rename(columns={"D": "cd_D", "N": "cd_N"}))
    other = v.groupby(CELL, observed=True)[["n_det", "frp_sum", "low_cd"]].sum()
    target = (mg[mg["sensor"] == C.AQUA].groupby(CELL, observed=True)["celldays"].sum()
                .rename("aqua"))
    f = cd.join(other, how="outer").join(target, how="outer").fillna(0).reset_index()
    tot = f["cd_D"] + f["cd_N"]
    f["low_share"] = np.where(tot > 0, f["low_cd"] / tot.where(tot > 0, 1), 0.0)
    f["m_sin"] = np.sin(2 * np.pi * f["month"] / 12)
    f["m_cos"] = np.cos(2 * np.pi * f["month"] / 12)
    f["region_code"] = f["region"].map(_REGION_CODES).astype(int)
    f["has_viirs"] = tot > 0
    return f


def fit_ml(feat, window):
    train = feat[in_window(feat, window)]
    model = HistGradientBoostingRegressor(
        loss="poisson", learning_rate=0.05, max_iter=400, max_leaf_nodes=31,
        min_samples_leaf=40, l2_regularization=1.0,
        categorical_features=[FEATURES.index("region_code")],
        random_state=C.RANDOM_SEED)
    model.fit(train[FEATURES], train["aqua"])
    # Aqua cell-days in cells VIIRS never saw cannot be predicted cell by cell;
    # a per-region correction puts that mass back.
    pred = model.predict(train.loc[train["has_viirs"], FEATURES])
    seen = train[train["has_viirs"]].assign(pred=pred)
    corr = (train.groupby("region")["aqua"].sum()
            / seen.groupby("region")["pred"].sum()).fillna(1.0)
    return model, corr


def predict_ml(model, corr, feat):
    f = feat[feat["has_viirs"]].copy()
    f["harm_B"] = model.predict(f[FEATURES]) * f["region"].map(corr).fillna(1.0)
    return f[CELL + ["harm_B"]]


# ----------------------------------------------------------------------------
# Validation on held-out years
# ----------------------------------------------------------------------------
def _metrics(pred, actual):
    ok = actual >= 10
    mape = float(np.mean(np.abs(pred[ok] - actual[ok]) / actual[ok]) * 100) if ok.any() else np.nan
    r = float(np.corrcoef(pred, actual)[0, 1]) if len(actual) > 2 else np.nan
    bias = float((pred.sum() - actual.sum()) / actual.sum() * 100)
    return {"MAPE_%": round(mape, 1), "r": round(r, 3), "bias_%": round(bias, 1)}


def validate(mg):
    """Fit on WIN_A_TRAIN, score monthly totals on WIN_A_TEST (national and by region)."""
    fac_a = fit_ratio(mg, C.AQUA, C.SNPP, C.WIN_A_TRAIN, n_boot=20)
    fac_b = _ratio_from_totals(_yearly_totals(mg, C.SNPP, C.N20, C.WIN_B))
    harm_a = apply_ratio_chain(mg, fac_a, fac_b)

    feat = build_features(mg)
    model, corr = fit_ml(feat, C.WIN_A_TRAIN)
    harm_b = predict_ml(model, corr, feat)

    test = mg[in_window(mg, C.WIN_A_TEST)]
    month_keys = ["year", "month"]

    def monthly(df, col, by=()):
        d = df[in_window(df, C.WIN_A_TEST)]
        return d.groupby(month_keys + list(by), observed=True)[col].sum()

    rows = {}
    series = {}
    for scope, by in (("Indonesia", ()), ("by region", ("region",))):
        actual = monthly(test[test["sensor"] == C.AQUA], "celldays", by)
        cand = {
            "raw VIIRS detections": monthly(test[test["sensor"] == C.SNPP], "n_det", by),
            "raw VIIRS cell-days": monthly(test[test["sensor"] == C.SNPP], "celldays", by),
            "Layer A (ratio)": monthly(harm_a, "harm_A", by),
            "Layer B (ML)": monthly(harm_b, "harm_B", by),
        }
        for name, s in cand.items():
            a, p = actual.align(s, fill_value=0)
            rows[(scope, name)] = _metrics(p.to_numpy(float), a.to_numpy(float))
            if scope == "Indonesia":
                series[name] = p
        if scope == "Indonesia":
            series["Aqua (actual)"] = actual
    table = pd.DataFrame(rows).T.reset_index().rename(columns={"level_0": "scope",
                                                               "level_1": "method"})
    ts = pd.DataFrame(series).fillna(0).reset_index()
    return table, ts
