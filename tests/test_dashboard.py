from pathlib import Path

import pandas as pd
import pytest

from conftest import needs_processed, needs_raw
from lahore_gw.dashboard_data import (coordinate_flag_breakdown, headline_model_row, observation_period,
                                      to_units, well_forecast, well_history, well_summary)

ROOT = Path(__file__).resolve().parents[1]


def _toy():
    wells = pd.DataFrame({"well_id": ["A", "B"], "village": ["x", "y"], "tehsil": ["T", "T"],
                          "lat": [31.5, 31.4], "lon": [74.3, 74.2], "coord_quality": ["consistent"] * 2})
    long = pd.DataFrame({
        "well_id": ["A", "A", "A", "B", "B", "B"], "year": [2019, 2019, 2020] * 2,
        "season": ["pre", "post", "pre"] * 2, "season_index": [4038, 4039, 4040] * 2,
        "depth_ft": [10, 11, 12, 20, None, None],
        "status": ["reading", "reading", "reading", "reading", "dead", "dead"]})
    return wells, long


def test_summary_uses_last_real_reading_not_dead_status():
    wells, long = _toy()
    s = well_summary(wells, long).set_index("well_id")
    assert s.loc["A", "last_depth_ft"] == 12 and s.loc["A", "status_in_latest_season"] == "reading"
    assert s.loc["B", "last_depth_ft"] == 20 and s.loc["B", "last_reading_season"] == "pre"
    assert s.loc["B", "status_in_latest_season"] == "dead"


def test_history_excludes_unavailable_and_forecast_is_labelled():
    wells, long = _toy()
    assert len(well_history(long, "B")) == 1
    fwd = pd.DataFrame({"well_id": ["A"], "target_year": [2020], "target_season": ["post"],
                        "nn_forecast_depth_ft": [12.7]})
    f = well_forecast(fwd, "A")
    assert f.iloc[0]["kind"] == "Model forecast (unverified)"
    assert well_forecast(fwd, "B").empty


def test_summary_helpers():
    wells, long = _toy()
    assert observation_period(long) == ("Pre-monsoon 2019", "Pre-monsoon 2020")
    assert to_units(10.0, "m") == pytest.approx(3.048) and to_units(10.0, "ft") == 10.0
    met = pd.DataFrame({"model": ["Ridge regression (alpha=1)", "Ridge regression (alpha=1)", "Other"],
                        "split": ["validation", "test", "test"], "mae_ft": [1.0, 2.0, 3.0]})
    assert headline_model_row(met).mae_ft == 2.0 and headline_model_row(pd.DataFrame()) is None


def test_coordinate_flag_breakdown_explains_the_two_counts():
    wells = pd.DataFrame({"well_id": ["LHR-001", "LHR-002", "LHR-003", "LHR-004"],
                          "coord_quality": ["files disagree by up to 3.0 km (latest file used)", "consistent",
                                            "9.0 km from same-village well GIS 4", "consistent"]})
    disc = pd.DataFrame({"gis_no": [1, 2], "distance_km": [3.0, 0.4]})
    b = coordinate_flag_breakdown(wells, disc)
    assert b["files_disagree"] == ["LHR-001", "LHR-002"] and b["disagree_below_threshold"] == ["LHR-002"]
    assert b["same_village_only"] == ["LHR-003"] and b["flagged"] == ["LHR-001", "LHR-003"]


@needs_processed
def test_streamlit_app_runs_without_errors():
    testing = pytest.importorskip("streamlit.testing.v1")
    at = testing.AppTest.from_file(str(ROOT / "app" / "dashboard.py"), default_timeout=180).run()
    assert not at.exception, at.exception
    assert [t.label for t in at.tabs] == ["Overview", "Explore a Well", "Model Results", "Data Quality"]
    assert any("current groundwater levels unverified" in m.value for m in at.markdown)
    labels = {m.label for m in at.metric}
    assert {"Monitoring wells", "Valid readings", "Observation period", "Held-out Ridge MAE"} <= labels
    at.selectbox(key="well").set_value("LHR-155").run()         # High concern, has a 2030 scenario
    text = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)
    assert not at.exception and "not a validated forecast" in text and "Low" not in text
    at.selectbox(key="well").set_value("LHR-095").run()         # no reading in the latest survey
    assert not at.exception and any("No next-season model output" in i.value for i in at.info)
    at.radio(key="units").set_value("m").run()
    assert not at.exception and any(m.value.endswith(" m") for m in at.metric if "Last observed" in m.label)


@needs_raw
@needs_processed
def test_real_well_histories_match_source_rows():
    """Displayed history = raw workbook cells, read here with plain pandas (no values hard-coded)."""
    from lahore_gw.config import RAW_DIR
    from lahore_gw.dashboard_data import load_dashboard_data
    from lahore_gw.raw_loading import parse_season_column
    data = load_dashboard_data()
    long, wells = data["long"], data["wells"]
    assert len(well_summary(wells, long)) == 31
    raw = pd.read_excel(RAW_DIR / "Water Level PR-03 TO Pr-15 Lahore Zone.xlsx", header=1)
    raw = raw[raw["District"].astype(str).str.strip().str.lower() == "lahore"]
    seasons = {c: parse_season_column(c) for c in raw.columns if parse_season_column(c)}
    for wid in ["LHR-095", "LHR-108"]:
        row = raw[raw["GIS. No."] == int(wid[4:])].iloc[0]
        expected = {(y, s): float(row[c]) for c, (y, s) in seasons.items()
                    if isinstance(row[c], (int, float)) and not pd.isna(row[c])}
        hist = well_history(long, wid).set_index(["year", "season"])["depth_ft"]
        shown = {k: v for k, v in hist.items() if k in expected}
        assert expected and shown.keys() == expected.keys()
        assert all(shown[k] == pytest.approx(expected[k], abs=1e-6) for k in expected)
    h108 = well_history(long, "LHR-108")
    assert (h108.iloc[-1].season, h108.iloc[-1].year) == ("pre", 2020)
    if not data["forecast"].empty:
        assert data["forecast"]["nn_forecast_depth_ft"].notna().all()
        assert data["forecast"]["label"].str.contains("not a current measurement").all()
