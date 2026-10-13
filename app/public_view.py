"""Public dashboard view: reviewed aggregates and a synthetic example only.

Used when the restricted local data is absent, or when LGO_PUBLIC_MODE=1. It
reads only publishable files (reports/public_summary.json, metrics,
comparisons, feature importance, sensitivity) and never per-well data.
"""

import altair as alt
import pandas as pd
import streamlit as st

import ui
from lahore_gw import water_risk as wr
from lahore_gw.dashboard_data import headline_model_row, to_units, well_history
from lahore_gw.synthetic import SYNTHETIC_LABEL, synthetic_well


def render(pub: dict | None, metrics, comparisons, imp_ridge, imp_nn, sensitivity):
    if pub is None:
        st.error("No public summary found (`reports/public_summary.json`). Obtain the original workbooks "
                 "(README → Data sources) and run `python scripts/run_all.py` to build the local view.")
        st.stop()
    tab_over, tab_well, tab_model, tab_dq = st.tabs(ui.TABS)

    with tab_over:
        ui.overview_cards(pub["wells"], pub["readings"], pub["observation_period"],
                          headline_model_row(metrics, ui.RIDGE_PREFIX))
        left, right = st.columns([3, 2], gap="large")
        with left:
            st.subheader("Readings per survey season")
            cov = pd.DataFrame(pub["coverage_by_season"])
            long = cov.melt(id_vars="season", var_name="status", value_name="wells")
            long["status"] = long.status.str.replace("_", " ")
            st.altair_chart(alt.Chart(long).mark_bar().encode(
                x=alt.X("season:N", sort=list(cov.season), title=None,
                        axis=alt.Axis(labelAngle=-60, labelFontSize=9, labelOverlap=False)),
                y=alt.Y("wells:Q", title="Wells"),
                color=alt.Color("status:N", title="Status", scale=alt.Scale(
                    domain=["reading", "closed", "dead", "blank", "not listed"],
                    range=[ui.TEAL, "#8a96a3", "#4b5563", "#d5dbe1", "#eceff2"]), legend=alt.Legend(orient="bottom")),
                tooltip=["season", "status", "wells"]).properties(height=330), width="stretch")
            st.caption("Counts across all monitored wells. The well map is shown only in the local view, because "
                       "well locations come from the restricted source files.")
        with right:
            c = pub["concern"]
            st.subheader(c["title"])
            ui.concern_cards(c["counts"], c["rules"])
            st.markdown(f"From 2003 to 2020, **{c['deepening']} of {c['with_trend']}** wells with enough readings "
                        f"got deeper; the median trend was **{c['median_trend_ft_per_year']:+.2f} ft/year**. These are "
                        "past readings at monitored wells, not current or citywide conditions.")
            st.caption(wr.rainfall_text(pub["rainfall"]))

    with tab_well:
        st.warning(f"**{SYNTHETIC_LABEL}.** This tab shows how a well is presented, using generated numbers. Real "
                   "well histories and model outputs are available only in the local view.")
        long, wells = synthetic_well()
        r = wr.well_indicators(long, wells).iloc[0]
        r["concern"] = str(r["concern"])
        units = st.radio("Depth units", ["ft", "m"], horizontal=True, key="units",
                         format_func=lambda u: "feet" if u == "ft" else "metres")
        st.markdown(f"#### {r.well_id} – synthetic &nbsp; {ui.chip(r.concern)}", unsafe_allow_html=True)
        k = st.columns(3)
        k[0].metric(f"Last value ({r.last_reading})", f"{to_units(r.last_depth_ft, units):.2f} {units}")
        k[1].metric("Values", f"{int(r.n_readings)} of {long.season_index.nunique()}")
        k[2].metric(f"Trend ({int(r.trend_from)}–{int(r.trend_to)})",
                    f"{to_units(r.trend_ft_per_year, units):+.2f} {units}/yr")
        hist = well_history(long, r.well_id)
        last = hist.iloc[-1]
        pdf = pd.concat([
            hist.assign(depth=to_units(hist.depth_ft, units), series="Synthetic values")[
                ["time", "depth", "season_label", "series"]],
            pd.DataFrame({"time": [last.time, wr.HORIZON_YEAR + 0.4],
                          "depth": [to_units(last.depth_ft, units), to_units(r.projected_depth_ft, units)],
                          "season_label": [last.season_label, f"{wr.HORIZON_YEAR} scenario"],
                          "series": [f"{wr.HORIZON_YEAR} straight-line trend scenario"] * 2})])
        domain = ["Synthetic values", f"{wr.HORIZON_YEAR} straight-line trend scenario"]
        base = alt.Chart(pdf).encode(
            x=alt.X("time:Q", title="Year", axis=alt.Axis(format="d")),
            y=alt.Y("depth:Q", title=f"Depth below ground ({'feet' if units == 'ft' else 'metres'})",
                    scale=alt.Scale(reverse=True, zero=False)),
            color=alt.Color("series:N", title=None, scale=alt.Scale(domain=domain, range=[ui.NAVY, ui.SLATE]),
                            legend=alt.Legend(orient="bottom", direction="vertical", labelLimit=400)),
            tooltip=["series", "season_label", alt.Tooltip("depth:Q", format=".2f")])
        st.altair_chart((base.transform_filter(alt.datum.series == domain[0]).mark_line(strokeWidth=2)
                         + base.transform_filter(alt.datum.series == domain[0]).mark_circle(size=45)
                         + base.transform_filter(alt.datum.series == domain[1]).mark_line(strokeDash=[6, 4]))
                        .properties(height=340), width="stretch")
        st.caption("SYNTHETIC. Depth axis points down. Two seasons are deliberately missing to show how gaps are "
                   "handled: they are not plotted and never treated as zero. The dashed line is a straight-line "
                   "historical trend scenario, not a validated forecast. No model output is shown for this example.")
        a, b = st.columns(2, gap="large")
        a.markdown("**What the values show** (synthetic)\n\n"
                   + "\n".join(f"* {x}" for x in wr.explain_why(r, pub["rainfall"])))
        b.markdown("**What could follow if the pattern continued** (synthetic)\n\n"
                   + "\n".join(f"* {x}" for x in wr.future_risks(r, pub["latest_survey"])))

    with tab_model:
        ui.model_results(metrics, comparisons, imp_ridge, imp_nn, test_rows=None)

    with tab_dq:
        st.subheader("Coverage and missing readings")
        sc = pub["status_counts"]
        st.markdown(f"{pub['readings']} real readings out of {pub['well_seasons']} well-seasons. Closed "
                    f"({sc.get('closed', 0)}), dead ({sc.get('dead', 0)}), blank ({sc.get('blank', 0)}) and not-listed "
                    f"({sc.get('not_listed', 0)}) seasons are kept as missing; they are never treated as zero or as "
                    "evidence of decline, and no values are interpolated. Per-season counts are on the Overview tab.")
        st.subheader("Coordinates and elevation")
        co, el = pub["coordinates"], pub["elevation"]
        st.markdown(f"* **{co['files_disagree']} wells** have coordinates that differ between source files (up to "
                    f"{co['max_difference_km']:.1f} km); {co['disagree_below_threshold']} differ by less than "
                    f"{co['flag_km']:g} km and are not flagged.\n"
                    f"* **{co['same_village_only']} wells** are flagged only because they lie more than 5 km from a "
                    "well with the same village name.\n"
                    f"* Together {co['disagree_over_threshold']} + {co['same_village_only']} = **{co['flagged']} "
                    "coordinate-flagged wells**. Nothing is edited by hand.\n"
                    f"* **{el['unresolved']} wells** have conflicting ground elevations (left unresolved) and "
                    f"**{el['missing']}** have none.")
        ui.elevation_policy_text()
        st.subheader("Source provenance")
        st.dataframe(pd.DataFrame(pub["sources"]).rename(columns={"sha256_16": "sha256 (first 16)"}),
                     hide_index=True, width="stretch")
        st.caption("Punjab Irrigation Department, Directorate of Land Reclamation (L.R.O., UCC Lahore) workbooks. "
                   "Their redistribution terms are unverified, so they are not in the repository; the MIT code "
                   "licence does not cover them.")
        ui.weather_limits(pub["wells"], pub.get("weather_series"))
        ui.sensitivity_table(sensitivity)
