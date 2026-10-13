import numpy as np
import pandas as pd

from lahore_gw.features import FEATURES, build_samples, chronological_split
from lahore_gw.modeling import make_preprocessor


def _toy_long():
    rows = []
    depths = {4020: 10.0, 4021: 9.0, 4022: 11.0, 4023: np.nan, 4024: 12.0, 4025: 11.5, 4026: 13.0}
    for si, d in depths.items():
        rows.append({"well_id": "W1", "season_index": si, "year": si // 2,
                     "season": "pre" if si % 2 == 0 else "post", "depth_ft": d,
                     "status": "reading" if not np.isnan(d) else "closed"})
    return pd.DataFrame(rows)


def _toy_wells():
    return pd.DataFrame({"well_id": ["W1"], "lat": [31.5], "lon": [74.3], "ground_elevation_m": [200.0],
                         "tehsil": ["T"], "village": ["V"]})


def test_samples_only_for_consecutive_observed_seasons():
    S = build_samples(_toy_long(), _toy_wells(), None)
    pairs = set(zip(S.origin_si, S.target_si))
    assert pairs == {(4020, 4021), (4021, 4022), (4024, 4025), (4025, 4026)}
    assert (S.target_si - S.origin_si == 1).all()


def test_features_use_only_past_values():
    S = build_samples(_toy_long(), _toy_wells(), None).set_index("origin_si")
    r = S.loc[4025]                     # origin post-2012, target pre-2013
    assert r.depth_t_ft == 11.5
    assert r.delta_prev_ft == 11.5 - 12.0
    assert np.isnan(r.delta_yoy_ft)      # 4023 was closed -> missing, not zero
    assert r.has_yoy == 0
    assert r.target_depth_ft == 13.0


def test_chronological_split_no_overlap_and_train_only_preprocessing():
    rng = np.random.default_rng(0)
    S = pd.DataFrame({f: rng.normal(size=60) for f in FEATURES})
    S["target_year"] = np.repeat(np.arange(2008, 2020), 5)
    tr, va, te = chronological_split(S, 2012, 2015)
    assert tr.target_year.max() < va.target_year.min() <= va.target_year.max() < te.target_year.min()
    prep = make_preprocessor().fit(tr[FEATURES])
    scaler = prep.named_transformers_["median"].named_steps["scale"]
    med_cols = [c for c in FEATURES if c not in ("delta_prev_ft", "delta_yoy_ft")]
    assert np.allclose(scaler.mean_, tr[med_cols].mean().to_numpy())   # fitted on train only
