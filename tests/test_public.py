"""The public view and public summary must never expose per-well data.

These tests run without the restricted data (they use only publishable files
and synthetic fixtures), so they also run in a public clone.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lahore_gw import public_summary, water_risk as wr
from lahore_gw.dashboard_data import public_mode_requested
from lahore_gw.synthetic import SYNTHETIC_ID, SYNTHETIC_LABEL, synthetic_well

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "reports" / "public_summary.json"
WELL_ID = re.compile(r"LHR-\d+")
COORD = re.compile(r"\b3[01]\.\d{3,}|\b7[34]\.\d{3,}")


def _fictional_inputs():
    long, wells = synthetic_well()
    wells = wells.assign(tehsil="T", coord_quality="consistent", elevation_status="consistent")
    weather = pd.DataFrame({"well_id": SYNTHETIC_ID, "year": range(2004, 2021), "season": "post",
                            "rain_mm": np.linspace(300, 500, 17)})
    long = long.assign(chosen_from="fictional.xlsx")
    return long, wells, weather


def test_public_mode_switch():
    assert public_mode_requested({"LGO_PUBLIC_MODE": "1"}) and public_mode_requested({"LGO_PUBLIC_MODE": "true"})
    assert not public_mode_requested({}) and not public_mode_requested({"LGO_PUBLIC_MODE": "0"})


def test_synthetic_well_is_labelled_and_shows_the_method():
    long, wells = synthetic_well()
    assert SYNTHETIC_ID.startswith("SYNTHETIC") and "not a real well" in SYNTHETIC_LABEL
    assert not WELL_ID.search(long.to_csv() + wells.to_csv())
    assert (long.status != "reading").sum() == 2 and long.loc[long.status != "reading", "depth_ft"].isna().all()
    r = wr.well_indicators(long, wells).iloc[0]
    assert r.trend_class == "Deepening" and np.isfinite(r.projected_depth_ft)


def test_summary_built_from_fictional_data_has_no_identifiers():
    long, wells, weather = _fictional_inputs()
    s = public_summary.build(long, wells, weather, pd.DataFrame(columns=["gis_no", "distance_km"]))
    text = json.dumps(s)
    assert not WELL_ID.search(text) and "SYNTHETIC" not in text and "Fictional" not in text
    assert s["readings"] == int((long.status == "reading").sum()) and s["wells"] == 1


@pytest.mark.skipif(not SUMMARY.exists(), reason="run scripts/05_water_level_risk.py first")
def test_published_summary_has_no_per_well_data():
    text = SUMMARY.read_text()
    assert not WELL_ID.search(text) and not COORD.search(text)
    s = json.loads(text)
    assert sum(s["concern"]["counts"].values()) == s["wells"]


@pytest.mark.skipif(not SUMMARY.exists(), reason="run scripts/05_water_level_risk.py first")
def test_public_dashboard_shows_no_per_well_data(monkeypatch):
    testing = pytest.importorskip("streamlit.testing.v1")
    monkeypatch.setenv("LGO_PUBLIC_MODE", "1")
    at = testing.AppTest.from_file(str(ROOT / "app" / "dashboard.py"), default_timeout=180).run()
    assert not at.exception, at.exception
    assert [t.label for t in at.tabs] == ["Overview", "Explore a Well", "Model Results", "Data Quality"]
    assert len(at.get("deck_gl_json_chart")) == 0                     # no well map
    parts = []
    for kind in ["markdown", "caption", "info", "warning", "subheader", "metric", "dataframe"]:
        for el in getattr(at, kind):
            v = el.value
            parts.append(v.to_csv() if hasattr(v, "to_csv") else v.data.to_csv() if hasattr(v, "data") else str(v))
    text = "\n".join(parts)
    assert not WELL_ID.search(text) and not COORD.search(text)
    assert "current groundwater levels unverified" in text and SYNTHETIC_LABEL in text
    assert "Low" not in text and "not a validated forecast" in text
