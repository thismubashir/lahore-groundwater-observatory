"""Baselines, a ridge regression and a small TensorFlow network.

Leakage controls
* chronological split by target year (train < validation < test);
* the preprocessor (imputation + scaling) is fitted on training rows only;
* hyper-parameters (ridge alpha, NN epochs) are chosen on validation rows;
* the test period is touched once, for the final numbers.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import FT_TO_M, MODELS_DIR, RANDOM_SEED
from .features import FEATURE_GROUPS, FEATURES

ZERO_FILL = ["delta_prev_ft", "delta_yoy_ft"]          # missing history -> 0 change + flag
MEDIAN_FILL = [f for f in FEATURES if f not in ZERO_FILL]


def make_preprocessor(features=FEATURES) -> ColumnTransformer:
    return ColumnTransformer([
        ("zero", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value=0.0)),
                           ("scale", StandardScaler())]), [f for f in ZERO_FILL if f in features]),
        ("median", Pipeline([("impute", SimpleImputer(strategy="median")),
                             ("scale", StandardScaler())]), [f for f in MEDIAN_FILL if f in features]),
    ])


# ----------------------------------------------------------------- metrics
def metrics(y_true, y_pred) -> dict:
    err = np.asarray(y_pred) - np.asarray(y_true)
    mae, rmse = float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err ** 2)))
    return {"n": int(len(err)), "mae_ft": mae, "rmse_ft": rmse,
            "mae_m": mae * FT_TO_M, "rmse_m": rmse * FT_TO_M, "bias_ft": float(np.mean(err))}


def bootstrap_mae_diff(y, p_model, p_ref, groups=None, n=2000, seed=RANDOM_SEED):
    """Paired bootstrap of MAE(model) - MAE(reference); negative = model better.

    If `groups` (e.g. well ids) is given, whole groups are resampled (cluster
    bootstrap), because seasons from the same well are not independent.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    d = np.abs(np.asarray(p_model) - y) - np.abs(np.asarray(p_ref) - y)
    if groups is None:
        idx = rng.integers(0, len(y), size=(n, len(y)))
        diffs = d[idx].mean(1)
    else:
        groups = np.asarray(groups)
        uniq = np.unique(groups)
        sums = np.array([d[groups == g].sum() for g in uniq])
        counts = np.array([(groups == g).sum() for g in uniq])
        pick = rng.integers(0, len(uniq), size=(n, len(uniq)))
        diffs = sums[pick].sum(1) / counts[pick].sum(1)
    return float(d.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


# --------------------------------------------------------------- baselines
def persistence(df):
    """Next reading = latest reading."""
    return df["depth_t_ft"].to_numpy()


class SeasonalSwing:
    """Latest reading + average change for that target season (from training)."""

    def fit(self, train):
        self.swing_ = train.groupby("target_is_post")["target_change_ft"].mean().to_dict()
        return self

    def predict(self, df):
        return (df["depth_t_ft"] + df["target_is_post"].map(self.swing_).fillna(0)).to_numpy()


def same_season_last_year(df):
    """Reading of the same season one year earlier; falls back to persistence."""
    return df["same_season_last_year_ft"].fillna(df["depth_t_ft"]).to_numpy()


class RidgeChange:
    """Linear regression (L2) predicting the seasonal change."""

    def __init__(self, alpha=1.0, features=None):
        self.alpha = alpha
        self.features = list(features) if features is not None else list(FEATURES)

    def fit(self, train):
        self.pipe_ = Pipeline([("prep", make_preprocessor(self.features)), ("ridge", Ridge(alpha=self.alpha))])
        self.pipe_.fit(train[self.features], train["target_change_ft"])
        return self

    def predict_change(self, df):
        return self.pipe_.predict(df[self.features])

    def predict(self, df):
        return df["depth_t_ft"].to_numpy() + self.predict_change(df)


# ---------------------------------------------------------- neural network
def _tf():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    return tf


def build_network(n_inputs: int, seed: int):
    tf = _tf()
    tf.keras.utils.set_random_seed(seed)
    reg = tf.keras.regularizers.l2(1e-3)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_inputs,)),
        tf.keras.layers.Dense(16, activation="relu", kernel_regularizer=reg),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(8, activation="relu", kernel_regularizer=reg),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=tf.keras.losses.Huber(delta=1.0))
    return model


@dataclass
class NNEnsemble:
    """Small MLP ensemble (several seeds) predicting the seasonal change."""
    n_members: int = 5
    max_epochs: int = 400
    patience: int = 30
    batch_size: int = 32
    features: list | None = None         # default: FEATURES

    def fit(self, train, val=None, epochs=None):
        tf = _tf()
        self.features = list(self.features or FEATURES)
        self.prep_ = make_preprocessor(self.features).fit(train[self.features])
        y = train["target_change_ft"].to_numpy()
        self.y_mean_, self.y_std_ = float(y.mean()), float(y.std() or 1.0)
        X = self.prep_.transform(train[self.features])
        ys = (y - self.y_mean_) / self.y_std_
        self.models_, self.histories_, self.best_epochs_ = [], [], []
        for k in range(self.n_members):
            m = build_network(X.shape[1], RANDOM_SEED + k)
            if val is not None:
                Xv = self.prep_.transform(val[self.features])
                yv = (val["target_change_ft"].to_numpy() - self.y_mean_) / self.y_std_
                es = tf.keras.callbacks.EarlyStopping(patience=self.patience, restore_best_weights=True)
                h = m.fit(X, ys, validation_data=(Xv, yv), epochs=self.max_epochs,
                          batch_size=self.batch_size, callbacks=[es], verbose=0)
                self.best_epochs_.append(int(np.argmin(h.history["val_loss"])) + 1)
            else:
                h = m.fit(X, ys, epochs=epochs, batch_size=self.batch_size, verbose=0)
            self.models_.append(m)
            self.histories_.append({k2: [float(v) for v in vals] for k2, vals in h.history.items()})
        return self

    def predict_change(self, df):
        X = np.asarray(self.prep_.transform(df[self.features]), dtype="float32")
        # direct call (not .predict) avoids tf.function retracing on many small batches
        preds = np.mean([np.asarray(m(X, training=False)).ravel() for m in self.models_], axis=0)
        return preds * self.y_std_ + self.y_mean_

    def predict(self, df):
        return df["depth_t_ft"].to_numpy() + self.predict_change(df)

    def save(self, folder):
        folder.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.prep_, folder / "preprocessor.joblib")
        for i, m in enumerate(self.models_):
            m.save(folder / f"nn_member_{i}.keras")
        (folder / "nn_config.json").write_text(json.dumps({
            "features": self.features, "y_mean_ft": self.y_mean_, "y_std_ft": self.y_std_,
            "n_members": len(self.models_), "target": "change in depth to water (ft) to next season",
            "prediction": "depth_t_ft + predicted change"}, indent=2))

    @classmethod
    def load(cls, folder):
        tf = _tf()
        cfg = json.loads((folder / "nn_config.json").read_text())
        obj = cls(n_members=cfg["n_members"], features=cfg["features"])
        obj.prep_ = joblib.load(folder / "preprocessor.joblib")
        obj.y_mean_, obj.y_std_ = cfg["y_mean_ft"], cfg["y_std_ft"]
        obj.models_ = [tf.keras.models.load_model(folder / f"nn_member_{i}.keras")
                       for i in range(cfg["n_members"])]
        return obj


# ------------------------------------------------------ feature importance
def permutation_importance(predict_fn, df, n_repeats=30, seed=RANDOM_SEED) -> pd.DataFrame:
    """Held-out permutation importance by feature group (MAE increase, ft).

    Shuffling a group breaks its link with the target; a large MAE increase
    means the fitted model *relies* on it. This is not evidence of causation.
    """
    rng = np.random.default_rng(seed)
    y = df["target_depth_ft"].to_numpy()
    base = np.mean(np.abs(predict_fn(df) - y))
    rows = []
    for group, cols in FEATURE_GROUPS.items():
        incs = []
        for _ in range(n_repeats):
            shuffled = df.copy()
            perm = rng.permutation(len(df))
            for c in cols:
                shuffled[c] = df[c].to_numpy()[perm]
            # depth_t_ft is also the anchor of the forecast; shuffle only its
            # use as an input, keep the anchor fixed.
            p = predict_fn(shuffled, anchor=df["depth_t_ft"].to_numpy())
            incs.append(np.mean(np.abs(p - y)) - base)
        rows.append({"feature_group": group, "features": ", ".join(cols),
                     "mae_increase_ft": float(np.mean(incs)), "std_ft": float(np.std(incs))})
    return pd.DataFrame(rows).sort_values("mae_increase_ft", ascending=False).reset_index(drop=True)


def save_ridge(model: RidgeChange, folder):
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, folder / "ridge_model.joblib")


MODELS_DIR = MODELS_DIR
