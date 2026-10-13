import numpy as np
import pandas as pd
import pytest

from lahore_gw.features import FEATURES
from lahore_gw.modeling import (NNEnsemble, RidgeChange, SeasonalSwing, metrics, permutation_importance,
                                persistence)

tf = pytest.importorskip("tensorflow")


def _synthetic(n=200, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({f: rng.normal(size=n) for f in FEATURES})
    df["depth_t_ft"] = 30 + 5 * rng.normal(size=n)
    df["target_is_post"] = rng.integers(0, 2, n).astype(float)
    change = 2.0 * df["delta_prev_ft"] - 1.5 * df["target_is_post"] + 0.1 * rng.normal(size=n)
    df["target_change_ft"] = change
    df["target_depth_ft"] = df["depth_t_ft"] + change
    df["same_season_last_year_ft"] = df["depth_t_ft"]
    return df


def test_models_beat_persistence_on_learnable_signal(tmp_path):
    tr, te = _synthetic(300, 0), _synthetic(100, 1)
    base = metrics(te.target_depth_ft, persistence(te))["mae_ft"]
    ridge = RidgeChange(1.0).fit(tr)
    assert metrics(te.target_depth_ft, ridge.predict(te))["mae_ft"] < 0.5 * base
    nn = NNEnsemble(n_members=1, max_epochs=150, patience=20).fit(tr, te.iloc[:50])
    p = nn.predict(te)
    assert metrics(te.target_depth_ft, p)["mae_ft"] < 0.7 * base
    nn.save(tmp_path / "nn")
    again = NNEnsemble.load(tmp_path / "nn")
    assert np.allclose(again.predict(te), p, atol=1e-4)
    assert SeasonalSwing().fit(tr).predict(te).shape == (len(te),)


def test_permutation_importance_finds_true_driver():
    tr, te = _synthetic(300, 0), _synthetic(150, 2)
    ridge = RidgeChange(1.0).fit(tr)

    def fn(df, anchor=None):
        a = df.depth_t_ft.to_numpy() if anchor is None else anchor
        return a + ridge.pipe_.predict(df[FEATURES])

    imp = permutation_importance(fn, te, n_repeats=5)
    assert imp.iloc[0]["feature_group"] == "Last seasonal change"
