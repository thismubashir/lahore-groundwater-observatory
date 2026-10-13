"""Shared Streamlit pieces for the local and public dashboard views."""

import altair as alt
import pandas as pd
import streamlit as st

from lahore_gw import water_risk as wr
from lahore_gw.config import FT_TO_M
from lahore_gw.dashboard_data import season_label

NAVY, TEAL, SLATE, GREY = "#14325c", "#0f8b8d", "#5f7d95", "#b8c0cc"
BANNER = "Historical research — current groundwater levels unverified"
RIDGE_PREFIX, NN_NAME = "Ridge regression", "TensorFlow NN ensemble"
TABS = ["Overview", "Explore a Well", "Model Results", "Data Quality"]


def page_header(public: bool):
    st.set_page_config(page_title="Lahore Groundwater Observatory", layout="wide")
    st.markdown(f"""<style>
h1, h2, h3, h4 {{ color: {NAVY}; }}
[data-testid="stMetricValue"] {{ color: {NAVY}; }}
.lgo-banner {{ border-left: 5px solid {TEAL}; background: #eef7f7; color: {NAVY}; padding: .55rem .9rem;
               border-radius: 4px; font-weight: 600; margin: .2rem 0 1rem 0; }}
.lgo-banner small {{ display: block; font-weight: 400; margin-top: .2rem; }}
.lgo-chip {{ display: inline-block; padding: 1px 10px; border-radius: 10px; color: #fff; font-size: .85rem;
             font-weight: 600; }}
.lgo-muted {{ color: #52606d; font-size: .92rem; }}
</style>""", unsafe_allow_html=True)
    st.title("Lahore Groundwater Observatory")
    st.markdown('<p class="lgo-muted">Depth to groundwater at Lahore District piezometers, 2003 to pre-monsoon '
                "2020: readings, historical trends, and a comparison of next-season forecasting models. A research "
                "prototype; about 30 monitoring wells cannot represent every neighbourhood.</p>",
                unsafe_allow_html=True)
    extra = ("<small>Public view: aggregate results and a synthetic example only. Per-well readings, locations and "
             "model outputs are withheld until redistribution permission for the source data is confirmed.</small>"
             if public else "")
    st.markdown(f'<div class="lgo-banner">{BANNER}{extra}</div>', unsafe_allow_html=True)


def footer():
    st.caption("Data: Punjab Irrigation Department / Directorate of Land Reclamation piezometer workbooks (Lahore "
               "Irrigation Zone); weather: NASA POWER. Inspired by the WAFER project (github.com/WAFERed); "
               "independent implementation, not affiliated. Code: MIT licence, which does not cover the source data. "
               f"Depths in feet below ground; 1 ft = {FT_TO_M} m.")


def chip(level: str) -> str:
    return f'<span class="lgo-chip" style="background:{wr.CONCERN_COLORS[level]}">{level}</span>'


def concern_cards(counts: dict, rules: dict):
    for lvl in wr.CONCERN_ORDER:
        with st.container(border=True):
            st.markdown(f"{chip(lvl)} &nbsp; **{counts.get(lvl, 0)} wells**", unsafe_allow_html=True)
            st.caption(rules[lvl][0].upper() + rules[lvl][1:] + ".")


def overview_cards(n_wells, n_readings, period, ridge):
    c = st.columns(4)
    c[0].metric("Monitoring wells", n_wells, help="Lahore District piezometers in the source files.")
    c[1].metric("Valid readings", n_readings,
                help="Real depth readings. Closed, dead, blank and not-listed seasons are not counted.")
    c[2].metric("Observation period", f"{period[0][-4:]}–{period[1][-4:]}",
                help=f"{period[0]} to {period[1]}. Readings carry a season and year; exact survey dates were "
                     "not recorded.")
    if ridge is not None:
        c[3].metric("Held-out Ridge MAE", f"{ridge.mae_ft:.2f} ft",
                    help=f"Historical evaluation: average absolute error on {int(ridge.n)} held-out next-season "
                         f"forecasts (2016–2020 targets), {ridge.mae_m:.2f} m. Not an accuracy percentage.")
    else:
        c[3].metric("Held-out Ridge MAE", "n/a", help="Run scripts/03_train_evaluate.py.")


def model_results(metrics: pd.DataFrame, comparisons: pd.DataFrame, imp_ridge: pd.DataFrame,
                  imp_nn: pd.DataFrame, test_rows: pd.DataFrame | None = None):
    """Held-out comparison. test_rows (per-sample observations) is shown only in the local view."""
    if metrics.empty:
        st.info("No model results yet. Run `python scripts/03_train_evaluate.py`.")
        return
    test = metrics[metrics.split == "test"].sort_values("mae_ft")
    n = int(test.n.iloc[0])
    st.markdown(f"**Historical evaluation.** All models are scored on the **same {n} held-out next-season "
                "targets** (pre-monsoon 2016 to pre-monsoon 2020, 20 wells). Ridge alpha and the network's training "
                "length were chosen on 2013–2015 validation targets; the test seasons were used once, for these "
                "numbers.")
    st.markdown("**MAE** (mean absolute error) is the average size of a forecast's miss, in feet or metres of "
                "depth. **RMSE** weights large misses more. **Bias** is the average signed error; negative means "
                "forecasts were too shallow. No accuracy percentage or prediction confidence is reported, because "
                "neither is defined or validated for these data.")
    tbl = test[["model", "mae_ft", "mae_m", "rmse_ft", "bias_ft"]].rename(columns={
        "mae_ft": "MAE (ft)", "mae_m": "MAE (m)", "rmse_ft": "RMSE (ft)", "bias_ft": "Bias (ft)"})
    hl = tbl.style.format(precision=2).apply(
        lambda row: ["background-color: #e3f2f2; font-weight: 600"
                     if row.model.startswith(RIDGE_PREFIX) or row.model == NN_NAME else "" for _ in row], axis=1)
    st.dataframe(hl, hide_index=True, width="stretch")
    bars = test.assign(kind=["Ridge regression" if m.startswith(RIDGE_PREFIX) else
                             "TensorFlow network" if m == NN_NAME else "Baseline or variant" for m in test.model])
    st.altair_chart(alt.Chart(bars).mark_bar().encode(
        x=alt.X("mae_ft:Q", title="Held-out MAE (ft) – lower is better"),
        y=alt.Y("model:N", sort="x", title=None, axis=alt.Axis(labelLimit=320, labelOverlap=False)),
        color=alt.Color("kind:N", title=None, scale=alt.Scale(
            domain=["Ridge regression", "TensorFlow network", "Baseline or variant"], range=[TEAL, NAVY, GREY]),
            legend=alt.Legend(orient="bottom")),
        tooltip=["model", alt.Tooltip("mae_ft:Q", format=".3f"), alt.Tooltip("rmse_ft:Q", format=".3f")]
    ).properties(height=alt.Step(30)), width="stretch")

    if not comparisons.empty:
        st.markdown("**Paired comparisons** (bootstrap that resamples whole wells; difference = first minus "
                    "second, negative = first is better)")
        comp = comparisons.assign(reading=["first clearly better" if hi < 0 else "first clearly worse" if lo > 0
                                           else "no clear difference"
                                           for lo, hi in zip(comparisons.ci95_low, comparisons.ci95_high)])
        st.dataframe(comp[["model", "mae_difference_ft", "ci95_low", "ci95_high", "reading"]].round(3).rename(
            columns={"model": "comparison", "mae_difference_ft": "MAE difference (ft)",
                     "ci95_low": "95% low", "ci95_high": "95% high"}), hide_index=True, width="stretch")

    if test_rows is not None and not test_rows.empty:
        st.markdown("**Observed versus model output on the held-out targets**")
        ridge_col = next(c for c in test_rows.columns if c.startswith(RIDGE_PREFIX))
        choice = st.radio("Model", [ridge_col, NN_NAME, "Persistence (last reading)"], horizontal=True,
                          key="avp_model")
        sc = test_rows.assign(predicted=test_rows[choice], observed=test_rows.target_depth_ft,
                              season=[season_label(y, s) for y, s in zip(test_rows.target_year,
                                                                         test_rows.target_season)])
        lim = [float(min(sc.observed.min(), sc.predicted.min())) - 2,
               float(max(sc.observed.max(), sc.predicted.max())) + 2]
        pts = alt.Chart(sc).mark_circle(size=40, color=TEAL, opacity=0.75).encode(
            x=alt.X("observed:Q", title="Observed depth (ft)", scale=alt.Scale(domain=lim)),
            y=alt.Y("predicted:Q", title="Model output (ft)", scale=alt.Scale(domain=lim)),
            tooltip=["well_id", "season", alt.Tooltip("observed:Q", format=".2f"),
                     alt.Tooltip("predicted:Q", format=".2f")])
        one = alt.Chart(pd.DataFrame({"x": lim, "y": lim})).mark_line(color=GREY, strokeDash=[4, 4]).encode(
            x="x:Q", y="y:Q")
        st.altair_chart((one + pts).properties(height=380), width="stretch")
        st.caption("Points on the dashed line are perfect forecasts. Points above it were forecast deeper than "
                   "observed.")
    else:
        st.caption("The observed-versus-output scatter uses individual readings, so it is shown only in the local "
                   "view.")

    st.markdown("**Held-out permutation importance** (how much MAE grows when a feature group is shuffled in the "
                "test seasons). These are predictive associations the fitted models rely on, not causes of "
                "groundwater change.")
    ic = st.columns(2, gap="large")
    for col, imp, title, colour in [(ic[0], imp_ridge, "Ridge regression", TEAL),
                                    (ic[1], imp_nn, "TensorFlow network", NAVY)]:
        if imp.empty:
            continue
        col.altair_chart(alt.Chart(imp).mark_bar(color=colour).encode(
            x=alt.X("mae_increase_ft:Q", title="MAE increase (ft)"),
            y=alt.Y("feature_group:N", sort="-x", title=None, axis=alt.Axis(labelLimit=320, labelOverlap=False)),
            tooltip=["feature_group", alt.Tooltip("mae_increase_ft:Q", format=".3f")]
        ).properties(title=title, height=alt.Step(24)), width="stretch")
    with st.expander("Validation-period results (2013–2015 targets, used for model choices)"):
        st.dataframe(metrics[metrics.split == "validation"][["model", "n", "mae_ft", "rmse_ft", "bias_ft"]]
                     .round(3), hide_index=True, width="stretch")


def weather_limits(n_wells: int, nseries):
    st.subheader("Weather data limitations")
    st.markdown(f"Weather comes from NASA POWER daily data on a coarse grid (about 0.5° × 0.625°, roughly 50–60 km). "
                f"All {n_wells} wells fall into **{nseries if nseries else 'a few'} distinct weather series**, so "
                "neighbouring wells share the same rainfall and temperature. Weather varies over time, not between "
                "neighbourhoods. Each reading is matched to a window ending 31 May (pre-monsoon) or 31 Oct "
                "(post-monsoon), because the workbooks give no measurement dates.")


def sensitivity_table(sens: pd.DataFrame):
    if sens.empty:
        return
    st.markdown("**Sensitivity of held-out results** (a fixed list of variants, all reported; none used to choose "
                "the model):")
    st.dataframe(sens[["variant", "test_rows", "ridge_mae_ft", "nn_mae_ft", "persistence_mae_ft",
                       "ridge_minus_persistence_ft", "ci95_low", "ci95_high"]].round(3).rename(columns={
                           "test_rows": "test rows", "ridge_mae_ft": "Ridge MAE (ft)",
                           "nn_mae_ft": "Network MAE (ft)", "persistence_mae_ft": "Persistence MAE (ft)",
                           "ridge_minus_persistence_ft": "Ridge − persistence (ft)", "ci95_low": "95% low",
                           "ci95_high": "95% high"}), hide_index=True, width="stretch")


def elevation_policy_text():
    st.markdown("Where source files disagree, no file is assumed correct: the elevation is left unresolved, recorded "
                "with the files that report each value, and not used by the models (they fill in the training "
                "median). The older file's elevation column is labelled \"ft\" but its values match the metre "
                "columns, so all elevations are read as metres above sea level.")
