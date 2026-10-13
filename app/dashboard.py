"""Lahore Groundwater Observatory: Streamlit dashboard for the historical research prototype.

Run:  streamlit run app/dashboard.py
Tabs: Overview, Explore a Well, Model Results, Data Quality.

Two views:
* local  - per-well histories, map and model outputs from the restricted local data;
* public - aggregates and a synthetic example only (app/public_view.py). Used when
           the local data is absent or LGO_PUBLIC_MODE=1. Use it for anything shared.
"""

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import pydeck as pdk  # noqa: E402
import streamlit as st  # noqa: E402

import ui  # noqa: E402
from lahore_gw import water_risk as wr  # noqa: E402
from lahore_gw.config import RAW_DIR  # noqa: E402
from lahore_gw.dashboard_data import (FORECAST_LABEL, coordinate_flag_breakdown, data_available,  # noqa: E402
                                      headline_model_row, load_dashboard_data, load_public_data,
                                      observation_period, public_mode_requested, season_label, to_units,
                                      well_history, well_summary)
from ui import NAVY, SLATE, TEAL, chip  # noqa: E402

PUBLIC = public_mode_requested() or not data_available()
ui.page_header(PUBLIC)

if PUBLIC:
    import public_view
    pub = load_public_data()
    public_view.render(pub["summary"], pub["metrics"], pub["comparisons"], pub["importance_ridge"],
                       pub["importance_nn"], pub["sensitivity"])
    ui.footer()
    st.stop()


@st.cache_data
def _load():
    data = load_dashboard_data()
    risk = wr.well_indicators(data["long"], data["wells"])
    risk["concern"] = risk["concern"].astype(str)
    return data, risk, wr.rainfall_context(data["long"], data["weather"])


data, risk, rain = _load()
wells, long, fwd, metrics = data["wells"], data["long"], data["forecast"], data["metrics"]
summary = well_summary(wells, long)
latest = long.loc[long["season_index"].idxmax()]
latest_label = season_label(latest.year, latest.season)
first_period, last_period = observation_period(long)
info = wells.set_index("well_id")
risk_i = risk.set_index("well_id")


def hex_rgb(h: str) -> list[int]:
    return [int(h[i:i + 2], 16) for i in (1, 3, 5)]


def well_map(df: pd.DataFrame, key: str | None = None, highlight: str | None = None, height: int = 380):
    """pydeck map coloured by concern; selectable when a key is given."""
    d = df.dropna(subset=["lat", "lon"]).copy()
    d["color"] = d["concern"].map(lambda c: hex_rgb(wr.CONCERN_COLORS[c]) + [215])
    d["trend"] = d["trend_ft_per_year"].map(lambda v: "n/a" if pd.isna(v) else f"{v:+.2f} ft/yr")
    d["depth"] = d["last_depth_ft"].map(lambda v: "n/a" if pd.isna(v) else f"{v:.1f} ft")
    d = d[["well_id", "village", "tehsil", "concern", "trend", "depth", "last_reading", "lat", "lon", "color"]]
    layers = [pdk.Layer("ScatterplotLayer", data=d, id="wells", get_position=["lon", "lat"], get_fill_color="color",
                        get_line_color=[255, 255, 255], stroked=True, line_width_min_pixels=1, get_radius=550,
                        radius_min_pixels=6, radius_max_pixels=16, pickable=True)]
    if highlight is not None:
        h = d[d.well_id == highlight]
        layers.append(pdk.Layer("ScatterplotLayer", data=h, id="selected", get_position=["lon", "lat"],
                                filled=False, stroked=True, get_line_color=hex_rgb(NAVY), line_width_min_pixels=3,
                                get_radius=900, radius_min_pixels=11, radius_max_pixels=24))
    deck = pdk.Deck(layers=layers, map_style=None,
                    initial_view_state=pdk.ViewState(latitude=31.45, longitude=74.33, zoom=9.1),
                    tooltip={"text": "{well_id} – {village} ({tehsil})\n{concern}\nTrend {trend}\n"
                                     "Last reading {depth}, {last_reading}"})
    if key:
        return st.pydeck_chart(deck, on_select="rerun", selection_mode="single-object", key=key, height=height)
    return st.pydeck_chart(deck, height=height)


def legend():
    st.markdown(" ".join(chip(lvl) for lvl in wr.CONCERN_ORDER), unsafe_allow_html=True)


tab_over, tab_well, tab_model, tab_dq = st.tabs(ui.TABS)

# ===================================================================== Overview
with tab_over:
    ui.overview_cards(len(wells), int((long["status"] == "reading").sum()), (first_period, last_period),
                      headline_model_row(metrics, ui.RIDGE_PREFIX))

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("Monitoring locations")
        well_map(risk, height=420)
        legend()
        n_flag = int((wells.coord_quality != "consistent").sum())
        st.caption("Colour = historical decline concern. Coordinates are from the source files; "
                   f"{n_flag} wells carry a coordinate warning (see Data Quality).")
    with right:
        st.subheader(wr.CONCERN_TITLE)
        s = wr.network_summary(risk)
        ui.concern_cards(s["counts"], wr.CONCERN_RULES)
        st.markdown(f"From 2003 to 2020, **{s['deepening']} of {s['with_trend']}** wells with enough readings got "
                    f"deeper; the median trend was **{s['median_trend_ft_per_year']:+.2f} ft/year**. These are past "
                    "readings at monitored wells, not current or citywide conditions.")
        st.caption(wr.rainfall_text(rain))

# ============================================================== Explore a Well
with tab_well:
    # a click on the map (previous run) selects that well
    pick = st.session_state.get("well_map")
    try:
        objs = pick["selection"]["objects"].get("wells", []) if pick else []
    except (KeyError, TypeError, AttributeError):
        objs = []
    if objs and objs[0]["well_id"] != st.session_state.get("_last_map_pick"):
        st.session_state["_last_map_pick"] = objs[0]["well_id"]
        st.session_state["well"] = objs[0]["well_id"]
        st.session_state["tehsil"] = "All tehsils"

    tehsils = ["All tehsils"] + sorted(wells["tehsil"].dropna().unique())
    c1, c2, c3 = st.columns([2, 3, 2])
    tehsil = c1.selectbox("Tehsil", tehsils, key="tehsil")
    pool = summary if tehsil == "All tehsils" else summary[summary["tehsil"] == tehsil]
    options = pool.sort_values("well_id")["well_id"].tolist()
    if st.session_state.get("well") not in options:
        st.session_state["well"] = "LHR-108" if "LHR-108" in options else options[0]
    wid = c2.selectbox("Well", options, key="well",
                       format_func=lambda w: f"{w} – {info.loc[w, 'village']} ({risk_i.loc[w, 'concern']})")
    units = c3.radio("Depth units", ["ft", "m"], horizontal=True, key="units",
                     format_func=lambda u: "feet" if u == "ft" else "metres")
    r, w = risk_i.loc[wid], info.loc[wid]
    hist = well_history(long, wid)
    fc = fwd[fwd["well_id"] == wid] if not fwd.empty else fwd

    mcol, dcol = st.columns([2, 3], gap="large")
    with mcol:
        well_map(risk, key="well_map", highlight=wid, height=330)
        st.caption("Click a well on the map to select it.")
    with dcol:
        st.markdown(f"#### {wid} – {w.village}, {w.tehsil} &nbsp; {chip(r.concern)}", unsafe_allow_html=True)
        k = st.columns(3)
        if pd.notna(r.get("last_depth_ft")):
            k[0].metric(f"Last observed depth ({r.last_reading})", f"{to_units(r.last_depth_ft, units):.2f} {units}")
        else:
            k[0].metric("Last observed depth", "none")
        k[1].metric("Observations", f"{int(r.n_readings)} of {long.season_index.nunique()}",
                    help="Real readings out of the seasons covered by the source files.")
        if pd.notna(r.trend_ft_per_year):
            k[2].metric(f"Historical trend ({int(r.trend_from)}–{int(r.trend_to)})",
                        f"{to_units(r.trend_ft_per_year, units):+.2f} {units}/yr",
                        help="Theil-Sen slope of pre-monsoon depths; positive = getting deeper.")
        else:
            k[2].metric("Historical trend", "not enough data")
        st.caption(f"Location {w.lat:.4f}° N, {w.lon:.4f}° E (from {w.coord_source}). "
                   f"Concern rule: {wr.CONCERN_RULES[r.concern]}.")
        warn = []
        if w.coord_quality != "consistent":
            warn.append(f"Coordinates: {w.coord_quality}.")
        if str(w.elevation_status).startswith("unresolved"):
            warn.append(f"Ground elevation unresolved ({w.elevation_provenance}); not used by the models.")
        elif str(w.elevation_status).startswith("missing"):
            warn.append("Ground elevation missing in all files.")
        if r.get("years_without_reading", 0) >= wr.STALE_YEARS:
            warn.append(f"No reading since {r.last_reading}; the later condition of this well is unknown.")
        for x in warn:
            st.warning(x)

    # ---- chart: observed, next-season model output, 2030 scenario
    unit_lbl = "feet" if units == "ft" else "metres"
    obs = hist.assign(depth=to_units(hist["depth_ft"], units), series="Observed reading")[
        ["time", "depth", "season_label", "series"]]
    parts = [obs]
    if len(fc):
        f0 = fc.iloc[0]
        parts.append(pd.DataFrame({"time": [f0.target_year + (0.8 if f0.target_season == "post" else 0.4)],
                                   "depth": [to_units(f0.nn_forecast_depth_ft, units)],
                                   "season_label": [season_label(f0.target_year, f0.target_season)],
                                   "series": ["Next-season model output (unverified)"]}))
    if pd.notna(r.get("projected_depth_ft")):
        last = hist.iloc[-1]
        parts.append(pd.DataFrame({"time": [last.time, wr.HORIZON_YEAR + 0.4],
                                   "depth": [to_units(last.depth_ft, units), to_units(r.projected_depth_ft, units)],
                                   "season_label": [last.season_label, f"{wr.HORIZON_YEAR} scenario"],
                                   "series": [f"{wr.HORIZON_YEAR} straight-line trend scenario"] * 2}))
    pdf = pd.concat(parts, ignore_index=True)
    domain = ["Observed reading", "Next-season model output (unverified)",
              f"{wr.HORIZON_YEAR} straight-line trend scenario"]
    enc = dict(x=alt.X("time:Q", title="Year", axis=alt.Axis(format="d")),
               y=alt.Y("depth:Q", title=f"Depth below ground ({unit_lbl})",
                       scale=alt.Scale(reverse=True, zero=False)),
               color=alt.Color("series:N", title=None, scale=alt.Scale(domain=domain, range=[NAVY, TEAL, SLATE]),
                               legend=alt.Legend(orient="bottom", direction="vertical", labelLimit=400)),
               tooltip=[alt.Tooltip("series:N", title="Type"), alt.Tooltip("season_label:N", title="Season"),
                        alt.Tooltip("depth:Q", title=f"Depth ({units})", format=".2f")])
    base = alt.Chart(pdf).encode(**enc)
    chart = (base.transform_filter(alt.datum.series == domain[0]).mark_line(strokeWidth=2)
             + base.transform_filter(alt.datum.series == domain[0]).mark_circle(size=45)
             + base.transform_filter(alt.datum.series == domain[2]).mark_line(strokeDash=[6, 4], strokeWidth=2)
             + base.transform_filter(alt.datum.series == domain[1]).mark_point(shape="diamond", size=160,
                                                                                filled=True))
    st.altair_chart(chart.properties(height=360), width="stretch")

    gaps = long[(long.well_id == wid) & (long.status != "reading")]["status"].value_counts().to_dict()
    gap_txt = ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in gaps.items()) or "none"
    st.caption(f"Observation dates: each reading carries a survey season and year only; exact survey dates were "
               f"not recorded, and points are placed at mid-season for display. Depth axis points down: lower on the "
               f"chart = deeper water. Seasons without a reading (not plotted, never treated as zero): {gap_txt}. "
               "Lines join readings only for display.")
    if len(fc):
        st.info(f"**{FORECAST_LABEL}.** From the {season_label(f0.origin_year, f0.origin_season)} reading of "
                f"{to_units(f0.depth_t_ft, units):.2f} {units}, the neural network gives "
                f"**{to_units(f0.nn_forecast_depth_ft, units):.2f} {units}** for "
                f"{season_label(f0.target_year, f0.target_season)} (ridge "
                f"{to_units(f0.ridge_forecast_depth_ft, units):.2f} {units}; persistence "
                f"{to_units(f0.persistence_depth_ft, units):.2f} {units}). No observation exists to check it.")
    else:
        status = long[(long.well_id == wid) & (long.season_index == long.season_index.max())]["status"]
        st.info(f"**No next-season model output for this well.** It has no reading in the latest survey "
                f"({latest_label}: {status.iloc[0].replace('_', ' ') if len(status) else 'not listed'}). The models "
                "only forecast the season after a reading in the latest survey, so no value is shown.")
    if pd.notna(r.get("projected_depth_ft")):
        st.caption(f"The dashed {wr.HORIZON_YEAR} line is a straight-line historical trend scenario, not a "
                   f"validated forecast: {to_units(r.projected_depth_ft, units):.1f} {units} (range "
                   f"{to_units(r.projected_low_ft, units):.1f}–{to_units(r.projected_high_ft, units):.1f}).")

    a, b = st.columns(2, gap="large")
    a.markdown("**What the readings show**\n\n" + "\n".join(f"* {x}" for x in wr.explain_why(r, rain)))
    b.markdown("**What could follow if the pattern continued**\n\n"
               + "\n".join(f"* {x}" for x in wr.future_risks(r, latest_label)))
    with st.expander("All seasons for this well, with source files"):
        t = long[long.well_id == wid].sort_values("season_index")
        t = t.assign(season=[season_label(y, s) for y, s in zip(t.year, t.season)],
                     depth=to_units(t["depth_ft"], units).round(2))
        st.dataframe(t[["season", "status", "depth", "sources", "chosen_from"]].rename(
            columns={"depth": f"depth ({units})", "chosen_from": "value taken from"}), hide_index=True, width="stretch")

# =============================================================== Model Results
with tab_model:
    ui.model_results(metrics, data["comparisons"], data["importance_ridge"], data["importance_nn"],
                     test_rows=data["test"])

# ================================================================ Data Quality
with tab_dq:
    st.subheader("Coverage and missing readings")
    order = sorted(long.season_index.unique())
    cov = long.assign(season=[season_label(y, s) for y, s in zip(long.year, long.season)],
                      status=long.status.str.replace("_", " "))
    season_sort = [season_label(si // 2, "pre" if si % 2 == 0 else "post") for si in order]
    st.altair_chart(alt.Chart(cov).mark_rect().encode(
        x=alt.X("season:N", sort=season_sort, title=None,
                axis=alt.Axis(labelAngle=-60, labelFontSize=9, labelOverlap=False)),
        y=alt.Y("well_id:N", title=None, axis=alt.Axis(labelFontSize=10, labelOverlap=False)),
        color=alt.Color("status:N", title="Status", scale=alt.Scale(
            domain=["reading", "closed", "dead", "blank", "not listed"],
            range=[TEAL, "#8a96a3", "#4b5563", "#d5dbe1", "#f1f3f5"]), legend=alt.Legend(orient="bottom")),
        tooltip=["well_id", "season", "status"]).properties(height=alt.Step(15)), width="stretch")
    counts = long.status.value_counts()
    st.markdown(f"{counts.get('reading', 0)} real readings out of {len(long)} well-seasons. Closed "
                f"({counts.get('closed', 0)}), dead ({counts.get('dead', 0)}), blank ({counts.get('blank', 0)}) and "
                f"not-listed ({counts.get('not_listed', 0)}) seasons are kept as missing; they are never treated as "
                "zero or as evidence of decline, and no values are interpolated.")

    st.subheader("Coordinates")
    br = coordinate_flag_breakdown(wells, data["coord_discrepancies"])
    st.markdown(f"* **{len(br['files_disagree'])} wells** have coordinates that differ between source files. "
                f"{len(br['disagree_below_threshold'])} of them ({', '.join(br['disagree_below_threshold'])}) differ "
                f"by less than {br['flag_km']:g} km and are not flagged.\n"
                f"* **{len(br['same_village_only'])} wells** ({', '.join(br['same_village_only'])}) are flagged only "
                "because they lie more than 5 km from a well with the same village name.\n"
                f"* Together: {len(br['disagree_over_threshold'])} + {len(br['same_village_only'])} = "
                f"**{len(br['flagged'])} coordinate-flagged wells**. The latest file's coordinates are shown; nothing "
                "is edited by hand.")
    flagged = wells[wells.coord_quality != "consistent"][["well_id", "village", "coord_quality", "coord_source"]]
    st.dataframe(flagged.rename(columns={"coord_quality": "coordinate warning", "coord_source": "coordinates from"}),
                 hide_index=True, width="stretch")

    st.subheader("Ground elevation")
    el = wells[wells.elevation_status != "consistent"][["well_id", "village", "elevation_status",
                                                        "elevation_provenance", "ground_elevation_m"]]
    ui.elevation_policy_text()
    el = el.assign(ground_elevation_m=el.ground_elevation_m.map(lambda v: "not used" if pd.isna(v) else f"{v:g}"),
                   elevation_provenance=el.elevation_provenance.fillna("—").replace("", "—"))
    st.dataframe(el.rename(columns={"ground_elevation_m": "value used (m)", "elevation_status": "status",
                                    "elevation_provenance": "values reported, by file"}),
                 hide_index=True, width="stretch")

    st.subheader("Source provenance")

    @st.cache_data
    def file_hash(name: str) -> str:
        p = RAW_DIR / name
        return hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else "file not present"

    src = long[long.status == "reading"].groupby("chosen_from").agg(
        readings=("well_id", "size"), first=("season_index", "min"), last=("season_index", "max")).reset_index()
    src["seasons taken"] = [f"{season_label(a // 2, 'pre' if a % 2 == 0 else 'post')} – "
                            f"{season_label(b // 2, 'pre' if b % 2 == 0 else 'post')}" for a, b in zip(src["first"],
                                                                                                    src["last"])]
    src["sha256 (first 16)"] = src.chosen_from.map(file_hash)
    st.dataframe(src.sort_values("first")[["chosen_from", "seasons taken", "readings", "sha256 (first 16)"]].rename(
        columns={"chosen_from": "source file"}), hide_index=True, width="stretch")
    st.caption("Punjab Irrigation Department, Directorate of Land Reclamation (L.R.O., UCC Lahore) workbooks; "
               "redistribution terms unverified, so they are not included in the repository. Where files overlap, "
               "all 93 overlapping readings agree.")

    ui.weather_limits(len(wells), data["weather_meta"].get("distinct_weather_series"))
    ui.sensitivity_table(data["sensitivity"])

ui.footer()
