import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from conftest import needs_raw
from lahore_gw.cleaning import _elevation, clean, resolve_records
from lahore_gw.config import RAW_DIR
from lahore_gw.raw_loading import WorkbookInfo


def _fake_source(name, gis, values):
    info = WorkbookInfo(path=Path(name), sheet="s", header_row=1, elevation_header="SEA LEVEL (meter)",
                        season_columns={"Pre-10": (2010, "pre")})
    wide = pd.DataFrame({"gis_no": [gis], "dlr_no": ["1DN"], "district": ["Lahore"], "tehsil": ["T"],
                         "village": ["V"], "lat": [31.5], "lon": [74.3], "elevation_raw": [200.0]})
    long = pd.DataFrame([{**wide.iloc[0].to_dict(), "year": 2010, "season": "pre", "depth_ft": d,
                          "status": s, "raw_value": str(d), "source_file": name, "source_column": "Pre-10"}
                         for d, s in values])
    return wide, info, long


def test_conflicting_overlap_is_reported_not_hidden():
    a = _fake_source("old.xlsx", 7, [(20.0, "reading")])
    b = _fake_source("new.xlsx", 7, [(25.0, "reading")])
    wells = pd.DataFrame({"gis_no": [7], "well_id": ["LHR-007"]})
    long, conflicts = resolve_records([a, b], {"old.xlsx": 1, "new.xlsx": 2}, wells)
    assert len(conflicts) == 1
    assert conflicts.iloc[0]["conflict_type"] == "different numeric readings"
    assert long.iloc[0]["conflict"]
    assert long.iloc[0]["depth_ft"] == 25.0       # most recent file, flagged


def test_reading_beats_unavailable_status_but_is_flagged():
    a = _fake_source("old.xlsx", 7, [(np.nan, "closed")])
    b = _fake_source("new.xlsx", 7, [(25.0, "reading")])
    wells = pd.DataFrame({"gis_no": [7], "well_id": ["LHR-007"]})
    long, conflicts = resolve_records([a, b], {"old.xlsx": 1, "new.xlsx": 2}, wells)
    assert conflicts.iloc[0]["conflict_type"] == "reading vs unavailable status"


def test_conflicting_elevations_stay_unresolved_with_provenance():
    g = pd.DataFrame({"elevation_raw": [300.0, 250.0, 250.0], "rank": [1, 2, 3],        # fictional values
                      "source_file": ["old.xlsx", "mid.xlsx", "new.xlsx"]})
    e = _elevation(g)
    assert np.isnan(e["ground_elevation_m"]) and e["elevation_status"].startswith("unresolved")
    assert e["elevation_latest_file_m"] == 250.0
    assert e["elevation_provenance"] == "300 m: old.xlsx; 250 m: mid.xlsx, new.xlsx"
    same = _elevation(g.assign(elevation_raw=[205.0, 205.0, 205.0]))
    assert same["ground_elevation_m"] == 205.0 and same["elevation_status"] == "consistent"
    assert _elevation(g.assign(elevation_raw=np.nan))["elevation_status"] == "missing in all files"


@needs_raw
def test_clean_real_data_rules():
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in RAW_DIR.iterdir() if p.is_file()}
    res = clean()
    L, W = res.long, res.wells
    # Lahore District only, verified one-to-one identifiers
    assert res.identifier_checks["gis_dlr_consistent"].all()
    assert W["tehsil"].str.contains("Lahore").all()
    assert len(W) == W["gis_no"].nunique()
    # one record per well x year x season
    assert not L.duplicated(["well_id", "year", "season"]).any()
    # unavailable readings stay missing, never zero
    assert L.loc[L.status != "reading", "depth_ft"].isna().all()
    assert (L.loc[L.status == "reading", "depth_ft"] > 0).all()
    # depth and elevation are separate quantities
    assert "ground_elevation_m" in W.columns and "ground_elevation_m" not in L.columns
    assert np.allclose(L["depth_m"].dropna(), L["depth_ft"].dropna() * 0.3048)
    # files disagree on the elevation of two wells: left unresolved
    unresolved = W[W.elevation_status.str.startswith("unresolved")].set_index("well_id")
    assert set(unresolved.index) == {"LHR-144", "LHR-150"} and unresolved.ground_elevation_m.isna().all()
    # original files untouched
    assert hashes == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in RAW_DIR.iterdir() if p.is_file()}
