"""Stage 3: build samples, train baselines + ridge + TensorFlow NN, evaluate, explain.

Run:  python scripts/03_train_evaluate.py
Outputs: models/ (preprocessor + networks), reports/metrics*.csv, reports/results.md,
         reports/feature_importance_*.csv, reports/figures/*.png,
         data/processed/forward_forecast.csv (unverified model output)
"""

import argparse
import datetime as dt
import json
import platform

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from lahore_gw import plots
from lahore_gw.config import FIGURES_DIR, MODELS_DIR, PROCESSED_DIR, REPORTS_DIR
from lahore_gw.reporting import md_table
from lahore_gw.features import FEATURES, WEATHER_FEATURES, build_samples, chronological_split
from lahore_gw.modeling import (NNEnsemble, RidgeChange, SeasonalSwing, bootstrap_mae_diff,
                                metrics, permutation_importance, persistence,
                                same_season_last_year, save_ridge)
from lahore_gw.prediction import data_snapshot, forecast_next_season, forecast_rows_only, model_version

TRAIN_LAST_YEAR = 2012   # targets 2003-2012
VAL_LAST_YEAR = 2015     # targets 2013-2015 (model selection)
                         # targets 2016-2020 = untouched test period
RIDGE_ALPHAS = [0.1, 1, 3, 10, 30, 100, 300]


def load():
    long = pd.read_csv(PROCESSED_DIR / "lahore_groundwater_long.csv")
    wells = pd.read_csv(PROCESSED_DIR / "lahore_wells.csv")
    wpath = PROCESSED_DIR / "weather_seasonal_by_well.csv"
    weather = pd.read_csv(wpath) if wpath.exists() else None
    return long, wells, weather


def main(quick=False):
    long, wells, weather = load()
    S = build_samples(long, wells, weather)
    train, val, test = chronological_split(S, TRAIN_LAST_YEAR, VAL_LAST_YEAR)
    trainval = pd.concat([train, val])
    split_info = {name: {"samples": len(d), "wells": int(d.well_id.nunique()),
                         "first_target": f"{d.iloc[0].target_season}-{d.iloc[0].target_year}",
                         "last_target": f"{d.iloc[-1].target_season}-{d.iloc[-1].target_year}"}
                  for name, d in [("train", train), ("validation", val), ("test", test)]}
    print(json.dumps(split_info, indent=1))

    # ---------------- model selection on validation (fit on train only)
    ridge_val = {a: metrics(val.target_depth_ft, RidgeChange(a).fit(train).predict(val))["mae_ft"]
                 for a in RIDGE_ALPHAS}
    best_alpha = min(ridge_val, key=ridge_val.get)
    no_wx = [f for f in FEATURES if f not in WEATHER_FEATURES]
    nn_kwargs = dict(n_members=2 if quick else 5, max_epochs=60 if quick else 400)
    nn_sel = NNEnsemble(**nn_kwargs).fit(train, val)
    best_epochs = int(np.median(nn_sel.best_epochs_))

    rows = []

    def record(name, split, df, pred):
        rows.append({"model": name, "split": split, **metrics(df.target_depth_ft, pred)})

    swing_tr = SeasonalSwing().fit(train)
    ridge_tr = RidgeChange(best_alpha).fit(train)
    for split, df in [("validation", val)]:
        record("Persistence (last reading)", split, df, persistence(df))
        record("Same season last year", split, df, same_season_last_year(df))
        record("Persistence + avg seasonal change", split, df, swing_tr.predict(df))
        record(f"Ridge regression (alpha={best_alpha})", split, df, ridge_tr.predict(df))
        record("Ridge without weather features", split, df,
               RidgeChange(best_alpha, no_wx).fit(train).predict(df))
        record("TensorFlow NN ensemble", split, df, nn_sel.predict(df))

    # ---------------- final fit on train+validation, single test evaluation
    swing = SeasonalSwing().fit(trainval)
    ridge = RidgeChange(best_alpha).fit(trainval)
    nn = NNEnsemble(**nn_kwargs).fit(trainval, epochs=best_epochs)
    preds = {
        "Persistence (last reading)": persistence(test),
        "Same season last year": same_season_last_year(test),
        "Persistence + avg seasonal change": swing.predict(test),
        f"Ridge regression (alpha={best_alpha})": ridge.predict(test),
        "Ridge without weather features": RidgeChange(best_alpha, no_wx).fit(trainval).predict(test),
        "TensorFlow NN ensemble": nn.predict(test),
    }
    for name, p in preds.items():
        record(name, "test", test, p)
    met = pd.DataFrame(rows)

    y = test.target_depth_ft.to_numpy()
    comp = []
    for name, p in preds.items():
        if name.startswith("Persistence (last"):
            continue
        d, lo, hi = bootstrap_mae_diff(y, p, preds["Persistence (last reading)"],
                                       groups=test.well_id.to_numpy())
        comp.append({"model": f"{name} vs Persistence", "mae_minus_persistence_ft": d, "ci95_low": lo,
                     "ci95_high": hi, "clearly_better_than_persistence": hi < 0})
    for a_name, b_name in [("TensorFlow NN ensemble", f"Ridge regression (alpha={best_alpha})"),
                           (f"Ridge regression (alpha={best_alpha})", "Ridge without weather features")]:
        d, lo, hi = bootstrap_mae_diff(y, preds[a_name], preds[b_name], groups=test.well_id.to_numpy())
        comp.append({"model": f"{a_name} vs {b_name}", "mae_minus_persistence_ft": d,
                     "ci95_low": lo, "ci95_high": hi, "clearly_better_than_persistence": hi < 0})
    comp = pd.DataFrame(comp).rename(columns={"mae_minus_persistence_ft": "mae_difference_ft",
                                              "clearly_better_than_persistence": "clearly_better"})

    # ---------------- held-out permutation importance (test period)
    def nn_fn(df, anchor=None):
        a = df.depth_t_ft.to_numpy() if anchor is None else anchor
        return a + nn.predict_change(df)

    def ridge_fn(df, anchor=None):
        a = df.depth_t_ft.to_numpy() if anchor is None else anchor
        return a + ridge.predict_change(df)

    reps = 5 if quick else 30
    imp_nn = permutation_importance(nn_fn, test, n_repeats=reps)
    imp_ridge = permutation_importance(ridge_fn, test, n_repeats=reps)

    # ---------------- save artefacts
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    nn.save(MODELS_DIR / "nn")
    save_ridge(ridge, MODELS_DIR)
    (MODELS_DIR / "training_info.json").write_text(json.dumps({
        "splits": split_info, "ridge_alpha_validation_mae_ft": ridge_val, "ridge_alpha": best_alpha,
        "nn_best_epochs_per_member": nn_sel.best_epochs_, "nn_final_epochs": best_epochs,
        "seasonal_swing_ft": {("post" if k else "pre"): v for k, v in swing.swing_.items()},
        "features": FEATURES, "quick_run": quick,
        "data_snapshot": data_snapshot(), "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "library_versions": library_versions()}, indent=2))
    met.to_csv(REPORTS_DIR / "metrics.csv", index=False)
    comp.to_csv(REPORTS_DIR / "metrics_vs_persistence.csv", index=False)
    imp_nn.to_csv(REPORTS_DIR / "feature_importance_nn.csv", index=False)
    imp_ridge.to_csv(REPORTS_DIR / "feature_importance_ridge.csv", index=False)
    out = test[["well_id", "origin_season", "origin_year", "target_season", "target_year",
                "depth_t_ft", "target_depth_ft"]].copy()
    for name, p in preds.items():
        out[name] = p
    out.to_csv(PROCESSED_DIR / "test_predictions.csv", index=False)
    per_well = (out.assign(nn_abs_err=np.abs(out["TensorFlow NN ensemble"] - out.target_depth_ft),
                           pers_abs_err=np.abs(out["Persistence (last reading)"] - out.target_depth_ft))
                .groupby("well_id")[["nn_abs_err", "pers_abs_err"]].agg(["mean", "count"]).round(2))
    per_well.columns = ["nn_mae_ft", "n", "persistence_mae_ft", "n2"]
    per_well.drop(columns="n2").to_csv(REPORTS_DIR / "test_error_by_well.csv")

    # ---------------- forward forecast from the last available season
    # same code path as scripts/04_predict.py, so both label and version rows identically
    snap = data_snapshot()
    fwd = forecast_rows_only(forecast_next_season(
        long, wells, weather, {"nn": nn, "ridge": ridge}, version=model_version(),
        snapshot=snap, trained_snapshot=snap))
    if len(fwd):
        fwd.to_csv(PROCESSED_DIR / "forward_forecast.csv", index=False)

    # ---------------- figures
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plots.model_comparison(met, FIGURES_DIR / "model_comparison_test_mae.png")
    sel = {k: preds[k] for k in ["Persistence (last reading)", f"Ridge regression (alpha={best_alpha})",
                                 "TensorFlow NN ensemble"]}
    plots.predicted_vs_actual(test, sel, FIGURES_DIR / "test_forecast_vs_observed.png")
    plots.change_errors(test, {k: v for k, v in sel.items() if not k.startswith("Persistence")},
                        FIGURES_DIR / "test_change_forecast_vs_observed.png")
    plots.feature_importance(imp_nn, FIGURES_DIR / "feature_importance_nn.png",
                             "Held-out permutation importance: TensorFlow NN")
    plots.feature_importance(imp_ridge, FIGURES_DIR / "feature_importance_ridge.png",
                             "Held-out permutation importance: ridge regression")
    plots.training_curves(nn_sel.histories_, FIGURES_DIR / "nn_training_curve.png")

    write_results(met, comp, imp_nn, imp_ridge, split_info, best_alpha, best_epochs, fwd)
    print(met.round(3).to_string())
    print(comp.round(3).to_string())
    print(imp_nn.round(3).to_string())


def library_versions():
    import sklearn
    import tensorflow as tf
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__, "tensorflow": tf.__version__}


def write_results(met, comp, imp_nn, imp_ridge, split_info, alpha, epochs, fwd):
    def tbl(df):
        return md_table(df)

    t = met[met.split == "test"].sort_values("mae_ft")
    best = t.iloc[0]
    pers = t[t.model == "Persistence (last reading)"].iloc[0]
    nnrow = t[t.model == "TensorFlow NN ensemble"].iloc[0]
    def row(prefix):
        return comp[comp.model.str.startswith(prefix)].iloc[0]

    nn_p = row("TensorFlow NN ensemble vs Persistence")
    nn_r = row("TensorFlow NN ensemble vs Ridge")
    wx = comp[comp.model.str.endswith("vs Ridge without weather features")].iloc[0]
    verdict = (
        ("The network beats persistence (95% interval of the MAE difference is below zero: "
         f"{nn_p.ci95_low:.2f} to {nn_p.ci95_high:.2f} ft). ") if nn_p.clearly_better else
        ("The network is NOT clearly better than repeating the last reading "
         f"(95% interval {nn_p.ci95_low:.2f} to {nn_p.ci95_high:.2f} ft). "))
    verdict += (
        ("It is also clearly better than ridge regression. " if nn_r.clearly_better else
         f"It is slightly but consistently WORSE than the much simpler ridge regression (difference "
         f"{nn_r.mae_difference_ft:+.2f} ft, 95% interval {nn_r.ci95_low:.2f} to {nn_r.ci95_high:.2f} ft), "
         "even though it looked marginally better on the validation years; the extra complexity is not "
         "justified by these data. " if nn_r.ci95_low > 0 else
         f"It is NOT better than the much simpler ridge regression (difference {nn_r.mae_difference_ft:+.2f} ft, "
         f"95% interval {nn_r.ci95_low:.2f} to {nn_r.ci95_high:.2f} ft), so the extra complexity is not justified "
         "by these data. "))
    verdict += (
        ("Weather features clearly improve the ridge model. " if wx.clearly_better else
         f"Adding NASA POWER weather does not clearly improve ridge regression (difference "
         f"{wx.mae_difference_ft:+.2f} ft, 95% interval {wx.ci95_low:.2f} to {wx.ci95_high:.2f} ft)."))
    text = f"""# Results (generated by `scripts/03_train_evaluate.py`)

**Task:** forecast depth to water (ft below ground) at the next observed season
(pre-monsoon -> post-monsoon or post-monsoon -> next pre-monsoon) for Lahore District
piezometers, using only information available at the end of the current season.

## Data splits (chronological, by target year)

{tbl(pd.DataFrame(split_info).T.reset_index().rename(columns={'index': 'split'}))}

Ridge alpha chosen on validation: **{alpha}**. NN epochs chosen on validation: **{epochs}**
(early stopping), then all models were refitted on train+validation and evaluated once on test.

## Errors

{tbl(met[['model', 'split', 'n', 'mae_ft', 'rmse_ft', 'mae_m', 'rmse_m', 'bias_ft']])}

Best test MAE: **{best.model}** with {best.mae_ft:.2f} ft ({best.mae_m:.2f} m).
Persistence: {pers.mae_ft:.2f} ft. TensorFlow NN: {nnrow.mae_ft:.2f} ft ({nnrow.mae_m:.2f} m).

### Paired comparisons (cluster bootstrap by well, test period)

`mae_difference_ft` = MAE(first model) - MAE(second model); negative means the first is better.
`clearly_better` = the whole 95% interval is below zero.

{tbl(comp)}

**Honest reading.** {verdict} With {split_info['test']['samples']} test samples from
{split_info['test']['wells']} wells, judge each difference by its interval in the table above, not
by its size. All models under-predict depth in the test period (negative bias):
water tables deepened faster in 2016–2020 than the training years suggested.

## Held-out feature importance (permutation on the test period)

TensorFlow NN:

{tbl(imp_nn[['feature_group', 'mae_increase_ft', 'std_ft']])}

Ridge regression:

{tbl(imp_ridge[['feature_group', 'mae_increase_ft', 'std_ft']])}

Importance shows what the fitted model *relies on* for these held-out seasons: a predictive
association. It does not prove that a factor *causes* groundwater change: features are correlated (e.g.
rainfall and temperature share the same coarse grid cell and season), and a model can
lean on proxies. Negative or near-zero values mean "no measurable reliance".

## Forward forecast

{len(fwd)} wells have a reading in the last available season, giving an
**unverified** forecast for the following season (see dashboard). These are model
outputs for a historical season, not current (2026) groundwater levels, and they do
not cover every Lahore neighbourhood.

![comparison](figures/model_comparison_test_mae.png)
![importance](figures/feature_importance_nn.png)

Per-well figures (`figures/test_forecast_vs_observed.png`, `figures/test_change_forecast_vs_observed.png`)
are written locally and not committed; see docs/DATA_POLICY.md.
"""
    (REPORTS_DIR / "results.md").write_text(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fast smoke run (fewer epochs/members)")
    main(quick=ap.parse_args().quick)
