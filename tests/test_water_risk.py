import numpy as np
import pandas as pd
import pytest

from conftest import needs_processed
from lahore_gw import water_risk as wr


def _toy():
    """A: deepening 1.5 ft/yr; B: rising 1 ft/yr; C: too short; D: deepening but dead readings in between."""
    years = range(2008, 2016)
    rows = []
    for i, y in enumerate(years):
        for season, si in [("pre", y * 2), ("post", y * 2 + 1)]:
            post = season == "post"
            rows += [("A", y, season, si, 20 + 1.5 * i - (1.0 if post else 0), "reading"),
                     ("B", y, season, si, 40 - 1.0 * i - (2.0 if post else 0), "reading"),
                     ("C", y, season, si, 10.0 if i < 2 else np.nan, "reading" if i < 2 else "dead"),
                     ("D", y, season, si, (15 + 0.5 * i) if i % 3 != 1 else np.nan,
                      "reading" if i % 3 != 1 else "dead")]
    long = pd.DataFrame(rows, columns=["well_id", "year", "season", "season_index", "depth_ft", "status"])
    wells = pd.DataFrame({"well_id": list("ABCD"), "village": list("abcd"), "tehsil": ["T"] * 4,
                          "lat": [31.5] * 4, "lon": [74.3] * 4})
    return long, wells


def test_concern_levels_follow_the_rules():
    ind = wr.well_indicators(*_toy()).set_index("well_id")
    assert ind.loc["A", "trend_ft_per_year"] == pytest.approx(1.5)
    assert ind.loc["A", "concern"] == "High" and ind.loc["A", "trend_class"] == "Deepening"
    assert ind.loc["B", "concern"] == wr.NO_DECLINE and ind.loc["B", "trend_class"] == "Rising"
    assert ind.loc["C", "concern"] == "Unknown"
    assert ind.loc["D", "concern"] == "Medium"          # 0.5 ft/yr, readings around dead seasons
    assert list(ind["concern"].cat.categories) == wr.CONCERN_ORDER


def test_unavailable_readings_are_never_used_as_depths():
    ind = wr.well_indicators(*_toy()).set_index("well_id")
    assert ind.loc["C", "n_readings"] == 4 and ind.loc["C", "last_depth_ft"] == 10.0
    assert ind.loc["D", "n_readings"] == 10


def test_seasonal_balance_and_suction_limit():
    ind = wr.well_indicators(*_toy()).set_index("well_id")
    a = ind.loc["A"]
    assert a.monsoon_rise_ft == pytest.approx(1.0) and a.dry_season_fall_ft == pytest.approx(2.5)
    # A last read at post-2015: 20 + 1.5*7 - 1 = 29.5 ft > 25 ft limit
    assert a.deeper_than_25ft
    d = ind.loc["D"]                                   # last 18.0 ft (post-2014), 0.5 ft/yr -> 25 ft in 14 years
    assert d.last_depth_ft == 18.0 and d.last_year == 2014
    assert not d.deeper_than_25ft and d.suction_limit_year == 2014 + 14
    assert d.projected_depth_ft == pytest.approx(d.last_depth_ft + 0.5 * (wr.HORIZON_YEAR - d.last_year))
    assert np.isnan(ind.loc["B", "projected_depth_ft"])   # no projection for a rising well


def test_explanations_are_observations_not_claimed_causes():
    ind = wr.well_indicators(*_toy()).set_index("well_id")
    rain = {"rain_trend_mm_per_year": 5.0, "rain_trend_low": -3.0, "rain_trend_high": 12.0,
            "rain_first_year": 2008, "rain_last_year": 2015, "mean_monsoon_rain_mm": 450.0,
            "rise_ft_per_100mm": 0.4, "rise_rain_corr": 0.2, "n_monsoons": 30}
    why = wr.explain_why(ind.loc["A"], rain)
    assert any("ends each year deeper" in s for s in why)
    assert any(wr.RAINFALL_FINDING in s for s in why)
    assert wr.DRIVERS_NOT_MEASURED in why
    assert wr.rainfall_text(rain).startswith(wr.RAINFALL_FINDING)
    assert "association" in wr.rainfall_text(rain)
    text = " ".join(" ".join(wr.explain_why(r, rain) + wr.future_risks(r, "x")) for _, r in ind.iterrows())
    for banned in ["not the cause", "Less rain does not explain", "need submersible", "Low"]:
        assert banned not in text
    falling = {**rain, "rain_trend_low": -20.0, "rain_trend_high": -2.0}
    assert any("declined" in s for s in wr.explain_why(ind.loc["A"], falling))
    risks = wr.future_risks(ind.loc["A"], "Post-monsoon 2015")
    assert any("Straight-line historical trend scenario" in s and "not a validated forecast" in s for s in risks)
    assert any("Not enough readings" in s for s in wr.future_risks(ind.loc["C"], "Post-monsoon 2015"))
    assert any("Monitoring gap" in s for s in wr.future_risks(ind.loc["C"], "Post-monsoon 2015"))


@needs_processed
def test_real_data_risk_summary():
    from lahore_gw.dashboard_data import load_dashboard_data
    data = load_dashboard_data()
    ind = wr.well_indicators(data["long"], data["wells"])
    s = wr.network_summary(ind)
    assert s["wells"] == 31 and sum(s["counts"].values()) == 31
    top = ind.iloc[0]
    assert top.concern == "High" and top.trend_ft_per_year >= wr.FAST_FT_PER_YEAR
    assert (ind.loc[ind.concern == "High", "trend_low"] > 0).all()
    if data["weather"] is not None:
        rain = wr.rainfall_context(data["long"], data["weather"])
        assert rain["rain_first_year"] >= 2002 and rain["n_monsoons"] > 100
