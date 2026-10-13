"""Re-runnable checks of the pipeline outputs; writes the evidence to reports/verification.md.

Run:  python scripts/verify_project.py     (after run_all.py; about 1 minute)

Each check prints PASS or FAIL with the numbers behind it. The report records
only what this script measured on the current files. Tests (pytest) and the
notebook are run separately; see README -> Verification.
"""

import datetime as dt
import sys

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from lahore_gw.config import FT_TO_M, PROCESSED_DIR, RAW_DIR, REPORTS_DIR, ROOT
from lahore_gw.features import FEATURES, build_samples, chronological_split
from lahore_gw.modeling import make_preprocessor
from lahore_gw.prediction import forecast_next_season, load_models, model_version
from lahore_gw.weather import window_bounds

TRAIN_LAST_YEAR, VAL_LAST_YEAR = 2012, 2015
TRACE_FILE, TRACE_COLUMN = "Lahore WL  PRE-20.xlsx", "Pre-20"
TRACE_WELLS = {"LHR-108": 108, "LHR-155": 155}       # read in Pre-2020; LHR-095 has no reading then
results = []


def check(name, ok, detail):
    results.append({"check": name, "result": "PASS" if ok else "FAIL", "evidence": detail})
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")


def main():
    long = pd.read_csv(PROCESSED_DIR / "lahore_groundwater_long.csv")
    wells = pd.read_csv(PROCESSED_DIR / "lahore_wells.csv")
    weather = pd.read_csv(PROCESSED_DIR / "weather_seasonal_by_well.csv")

    # ---- units and missing values
    rd = long[long.status == "reading"]
    check("Depth units", np.allclose(rd.depth_m, rd.depth_ft * FT_TO_M) and (rd.depth_ft > 0).all(),
          f"{len(rd)} readings; depth_m = depth_ft x 0.3048; range {rd.depth_ft.min():g}-{rd.depth_ft.max():g} ft")
    na = long[long.status != "reading"]
    check("Unavailable readings stay missing", na.depth_ft.isna().all(),
          f"{len(na)} closed/dead/blank/not-listed records, none with a depth value; "
          f"counts {na.status.value_counts().to_dict()}")
    unres = wells[wells.elevation_status.str.startswith("unresolved")]
    check("Conflicting elevations left unresolved", unres.ground_elevation_m.isna().all(),
          "; ".join(f"{r.well_id} ({r.elevation_provenance})" for r in unres.itertuples()) or "none")

    # ---- pairing, timing and splits
    S = build_samples(long, wells, weather)
    piv = long.pivot(index="well_id", columns="season_index", values="depth_ft")
    origin_ok = all(abs(piv.loc[r.well_id, r.origin_si] - r.depth_t_ft) < 1e-9 for r in S.itertuples())
    check("Consecutive-season pairing", bool((S.target_si == S.origin_si + 1).all()
                                              and S.target_depth_ft.notna().all() and origin_ok),
          f"{len(S)} samples; every target is the next season and a real reading; latest-depth feature = origin reading")
    starts = [window_bounds(int(y), s)[0].isoformat() for y, s in zip(S.target_year, S.target_season)]
    w = weather.assign(origin_si=weather.year * 2 + (weather.season == "post").astype(int))
    ends = S.merge(w[["well_id", "origin_si", "window_end"]], on=["well_id", "origin_si"], how="left")["window_end"]
    check("Predictors available at the forecast origin", bool((ends.values < np.array(starts)).all()),
          "every origin weather window ends before the target window starts; target-window weather is not a feature")
    tr, va, te = chronological_split(S, TRAIN_LAST_YEAR, VAL_LAST_YEAR)
    check("Chronological split", tr.target_si.max() < va.target_si.min() and va.target_si.max() < te.target_si.min(),
          f"train {len(tr)} (targets to {TRAIN_LAST_YEAR}), validation {len(va)} ({TRAIN_LAST_YEAR + 1}-"
          f"{VAL_LAST_YEAR}), test {len(te)} ({VAL_LAST_YEAR + 1}+)")

    # ---- saved artefacts
    models, info = load_models()
    trval = pd.concat([tr, va])
    saved = models["nn"].prep_.transform(te[FEATURES])
    fresh = make_preprocessor().fit(trval[FEATURES]).transform(te[FEATURES])
    leaky = make_preprocessor().fit(pd.concat([trval, te])[FEATURES]).transform(te[FEATURES])
    check("Test rows never reached preprocessing", np.allclose(saved, fresh) and not np.allclose(saved, leaky),
          "saved preprocessor equals one fitted on train+validation only, and differs from one fitted with test rows")
    val_mae = info["ridge_alpha_validation_mae_ft"]
    best_alpha = float(min(val_mae, key=val_mae.get))
    med_epochs = int(np.median(info["nn_best_epochs_per_member"]))
    check("Hyper-parameters chosen on validation",
          float(info["ridge_alpha"]) == best_alpha and info["nn_final_epochs"] == med_epochs,
          f"ridge alpha {info['ridge_alpha']} has the lowest validation MAE; network epochs "
          f"{info['nn_final_epochs']} = median early-stopping epoch on validation {info['nn_best_epochs_per_member']}")
    tp = pd.read_csv(PROCESSED_DIR / "test_predictions.csv")
    key = ["well_id", "target_year", "target_season"]
    m = te.merge(tp, on=key, suffixes=("", "_saved"))
    ridge_col = f"Ridge regression (alpha={info['ridge_alpha']})"
    dn = float(np.max(np.abs(models["nn"].predict(m) - m["TensorFlow NN ensemble"])))
    dr = float(np.max(np.abs(models["ridge"].predict(m) - m[ridge_col])))
    check("Saved models reproduce saved test predictions", len(m) == len(te) and max(dn, dr) < 1e-6,
          f"{len(m)} rows; largest difference network {dn:.1e} ft, ridge {dr:.1e} ft")
    fwd = pd.read_csv(PROCESSED_DIR / "forward_forecast.csv")
    fc = forecast_next_season(long, wells, weather, models, version=model_version())
    fc = fc[fc.nn_forecast_depth_ft.notna()].merge(fwd, on="well_id", suffixes=("", "_saved"))
    df = float(np.max(np.abs(fc.nn_forecast_depth_ft - fc.nn_forecast_depth_ft_saved)))
    check("Saved models reproduce the forward forecast", df < 1e-6, f"{len(fc)} wells; largest difference {df:.1e} ft")

    # ---- raw cell -> processed -> forecast -> dashboard
    raw = pd.read_excel(RAW_DIR / TRACE_FILE, header=1)
    raw = raw[raw["District"].astype(str).str.strip().str.lower() == "lahore"]
    from streamlit.testing.v1 import AppTest
    for wid, gis in TRACE_WELLS.items():
        cell = float(raw.loc[raw["GIS. No."] == gis, TRACE_COLUMN].iloc[0])
        proc = long[(long.well_id == wid) & (long.year == 2020) & (long.season == "pre")].depth_ft.iloc[0]
        f = fwd[fwd.well_id == wid].iloc[0]
        at = AppTest.from_file(str(ROOT / "app" / "dashboard.py"), default_timeout=180).run()
        at.selectbox(key="well").set_value(wid).run()
        shown = " ".join(x.value for x in at.markdown) + " ".join(x.value for x in at.info)
        ok = (cell == proc and not at.exception and f"{proc:.2f} ft" in shown
              and f"{f.nn_forecast_depth_ft:.2f} ft" in shown)
        check(f"Trace {wid}", ok, f"raw {TRACE_FILE} {TRACE_COLUMN} = {cell:g} -> processed {proc:g} ft -> "
              f"network forecast post-2020 {f.nn_forecast_depth_ft:.2f} ft -> both shown in the dashboard")
    at = AppTest.from_file(str(ROOT / "app" / "dashboard.py"), default_timeout=180).run()
    at.selectbox(key="well").set_value("LHR-095").run()
    shown = " ".join(x.value for x in at.markdown) + " ".join(x.value for x in at.info) + \
        " ".join(x.value for x in at.caption)
    check("Unsupported forecast explained (LHR-095)", not at.exception and "No next-season model output" in shown,
          "no reading in the latest season, so the dashboard explains why instead of showing a number")

    write(results)
    return all(r["result"] == "PASS" for r in results)


def write(rows):
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    body = "\n".join(f"| {r['check']} | {r['result']} | {r['evidence']} |" for r in rows)
    (REPORTS_DIR / "verification.md").write_text(
        f"# Verification record (generated by `scripts/verify_project.py`, {stamp})\n\n"
        "Measured on the current processed data and saved models. Tests and the notebook are run "
        "separately (see README).\n\n| Check | Result | Evidence |\n|---|---|---|\n" + body + "\n")


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
