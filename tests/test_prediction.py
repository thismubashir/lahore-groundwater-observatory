import datetime as dt

import numpy as np
import pandas as pd
import pytest

from conftest import needs_processed
from lahore_gw.prediction import FORECAST_STATUS, forecast_next_season, forecast_rows_only


class PlusOne:
    """Stand-in model: latest depth + 1 ft."""

    def predict(self, df):
        return df["depth_t_ft"].to_numpy() + 1.0


def _toy():
    wells = pd.DataFrame({"well_id": ["A", "B"], "village": ["x", "y"], "tehsil": ["T", "T"],
                          "lat": [31.5, 31.4], "lon": [74.3, 74.2], "ground_elevation_m": [200.0, 205.0]})
    long = pd.DataFrame({
        "well_id": ["A"] * 3 + ["B"] * 3, "year": [2019, 2019, 2020] * 2,
        "season": ["pre", "post", "pre"] * 2, "season_index": [4038, 4039, 4040] * 2,
        "depth_ft": [10.0, 11.0, 12.0, 20.0, 21.0, np.nan],
        "status": ["reading"] * 3 + ["reading", "reading", "dead"]})
    return long, wells


def test_forecast_only_next_season_and_reasons_for_others():
    long, wells = _toy()
    fc = forecast_next_season(long, wells, None, {"nn": PlusOne()}, today=dt.date(2026, 10, 13),
                              version="v1", snapshot={"long_csv": "s"}, trained_snapshot={"long_csv": "s"})
    a, b = fc.set_index("well_id").loc["A"], fc.set_index("well_id").loc["B"]
    assert (a.target_season, a.target_year, a.nn_forecast_depth_ft) == ("post", 2020, 13.0)
    assert a.status == FORECAST_STATUS and a.stale and a.data_age_years == pytest.approx(6.0, abs=0.1)
    assert a.model_version == "v1" and a.snapshot_matches_training
    # B is dead in the latest season: no forecast, never a guessed number
    assert b.status.startswith("NO FORECAST") and "dead" in b.status and "post-2019" in b.status
    assert np.isnan(b.nn_forecast_depth_ft)
    assert forecast_rows_only(fc).well_id.tolist() == ["A"]


def test_snapshot_mismatch_is_flagged():
    long, wells = _toy()
    fc = forecast_next_season(long, wells, None, {"nn": PlusOne()},
                              snapshot={"long_csv": "new"}, trained_snapshot={"long_csv": "old"})
    assert not fc.snapshot_matches_training.any()


def test_bad_requests_raise():
    long, wells = _toy()
    with pytest.raises(ValueError, match="unknown well"):
        forecast_next_season(long, wells, None, {"nn": PlusOne()}, well_ids=["Z"])
    long.loc[2, "depth_ft"] = -5.0
    with pytest.raises(ValueError, match="implausible"):
        forecast_next_season(long, wells, None, {"nn": PlusOne()})


@needs_processed
def test_saved_models_reproduce_forward_forecast():
    pytest.importorskip("tensorflow")
    from lahore_gw.config import PROCESSED_DIR
    from lahore_gw.prediction import load_models

    models, _ = load_models()
    long = pd.read_csv(PROCESSED_DIR / "lahore_groundwater_long.csv")
    wells = pd.read_csv(PROCESSED_DIR / "lahore_wells.csv")
    weather = pd.read_csv(PROCESSED_DIR / "weather_seasonal_by_well.csv")
    fc = forecast_rows_only(forecast_next_season(long, wells, weather, models)).set_index("well_id")
    saved = pd.read_csv(PROCESSED_DIR / "forward_forecast.csv").set_index("well_id")
    assert list(fc.index) == list(saved.index)
    for col in ["nn_forecast_depth_ft", "ridge_forecast_depth_ft"]:
        assert np.allclose(fc[col], saved[col], atol=1e-3), col
