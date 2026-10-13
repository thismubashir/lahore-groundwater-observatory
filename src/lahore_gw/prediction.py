"""Next-season forecasts from saved, matched model artefacts.

Used by stage 3 (forward forecast right after training) and by
scripts/04_predict.py (re-forecast later from models/ without retraining).

Safeguards
* only the NEXT season after a well's reading in the latest surveyed season
  is supported; other wells get a row explaining why there is no forecast;
* unknown well ids, missing feature columns and implausible depths raise;
* the saved models must use the current feature list (matched artefacts);
* every row carries the model version, the data snapshot it came from,
  whether that snapshot is the one the models were trained on, and how old
  the target season is ("stale": it is a historical season, not today).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import MODELS_DIR, PROCESSED_DIR
from .features import FEATURES, build_samples

FORECAST_STATUS = "UNVERIFIED historical model forecast; not a current measurement"
MAX_PLAUSIBLE_DEPTH_FT = 500.0
ARTIFACT_FILES = ["training_info.json", "ridge_model.joblib", "nn/nn_config.json", "nn/preprocessor.joblib"]


def sha256(path: Path, n: int = 16) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:n]


def data_snapshot(processed_dir: Path = PROCESSED_DIR) -> dict:
    """Short hashes of the processed inputs a forecast depends on."""
    out = {"long_csv": sha256(processed_dir / "lahore_groundwater_long.csv"),
           "wells_csv": sha256(processed_dir / "lahore_wells.csv")}
    wx = processed_dir / "weather_seasonal_by_well.csv"
    out["weather_csv"] = sha256(wx) if wx.exists() else None
    return out


def model_version(models_dir: Path = MODELS_DIR) -> str:
    """Fingerprint of every saved artefact (changes whenever models are retrained)."""
    files = [models_dir / f for f in ARTIFACT_FILES] + sorted((models_dir / "nn").glob("nn_member_*.keras"))
    h = hashlib.sha256()
    for f in files:
        h.update(f.relative_to(models_dir).as_posix().encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:12]


def period_end(year: int, season: str) -> dt.date:
    return dt.date(int(year), 5, 31) if season == "pre" else dt.date(int(year), 10, 31)


def load_models(models_dir: Path = MODELS_DIR) -> tuple[dict, dict]:
    """Load the ridge model and NN ensemble and check they match the code's features."""
    from .modeling import NNEnsemble

    info = json.loads((models_dir / "training_info.json").read_text())
    nn_cfg = json.loads((models_dir / "nn" / "nn_config.json").read_text())
    ridge = joblib.load(models_dir / "ridge_model.joblib")
    for name, feats in [("training_info", info["features"]), ("nn_config", nn_cfg["features"]),
                        ("ridge", ridge.features)]:
        if list(feats) != FEATURES:
            raise ValueError(f"{name} was saved with features {feats}; the code expects {FEATURES}. "
                             "Retrain with scripts/03_train_evaluate.py.")
    nn = NNEnsemble.load(models_dir / "nn")
    return {"nn": nn, "ridge": ridge}, info


def _check_inputs(rows: pd.DataFrame):
    missing = [f for f in FEATURES if f not in rows.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    d = rows["depth_t_ft"]
    if not np.isfinite(d).all() or (d < 0).any() or (d > MAX_PLAUSIBLE_DEPTH_FT).any():
        raise ValueError(f"implausible latest depth(s): {d[~d.between(0, MAX_PLAUSIBLE_DEPTH_FT)].tolist()}")


def forecast_next_season(long: pd.DataFrame, wells: pd.DataFrame, weather: pd.DataFrame | None,
                         models: dict, well_ids=None, today: dt.date | None = None,
                         version: str | None = None, snapshot: dict | None = None,
                         trained_snapshot: dict | None = None) -> pd.DataFrame:
    """One row per requested well: a forecast, or the reason there is none.

    `models` maps a short name ("nn", "ridge") to an object with .predict(df).
    """
    today = today or dt.date.today()
    known = set(long["well_id"])
    wanted = sorted(known) if well_ids is None else list(well_ids)
    unknown = [w for w in wanted if w not in known]
    if unknown:
        raise ValueError(f"unknown well id(s): {unknown}")

    latest_si = int(long["season_index"].max())
    F = build_samples(long, wells, weather, include_unlabeled_last=True)
    rows = F[(F.origin_si == latest_si) & F.target_depth_ft.isna() & F.well_id.isin(wanted)].copy()
    if len(rows):
        _check_inputs(rows)
        for name, m in models.items():
            rows[f"{name}_forecast_depth_ft"] = m.predict(rows)
        rows["persistence_depth_ft"] = rows["depth_t_ft"]
        rows["status"] = FORECAST_STATUS

    # wells without a reading in the latest season: say why, never guess
    ok = long[long.status == "reading"].sort_values("season_index").groupby("well_id").tail(1)
    last = ok.set_index("well_id")
    latest_status = long[long.season_index == latest_si].set_index("well_id")["status"]
    info = wells.set_index("well_id")
    skipped = []
    for w in wanted:
        if w in set(rows.well_id):
            continue
        reason = f"no reading in the latest surveyed season ({latest_status.get(w, 'not_listed')})"
        if w in last.index:
            reason += f"; last reading {last.loc[w, 'season']}-{int(last.loc[w, 'year'])}"
        skipped.append({"well_id": w, "village": info.loc[w, "village"], "tehsil": info.loc[w, "tehsil"],
                        "lat": info.loc[w, "lat"], "lon": info.loc[w, "lon"],
                        "status": f"NO FORECAST: {reason}"})

    cols = ["well_id", "village", "tehsil", "lat", "lon", "origin_season", "origin_year", "depth_t_ft",
            "target_season", "target_year", *[f"{n}_forecast_depth_ft" for n in models],
            "persistence_depth_ft", "status"]
    out = pd.concat([rows.reindex(columns=cols), pd.DataFrame(skipped).reindex(columns=cols)],
                    ignore_index=True)
    out["label"] = out["status"]          # column name the dashboard reads
    has = out["target_year"].notna()
    out["target_period_end"] = None
    out["data_age_years"] = np.nan
    if has.any():
        ends = [period_end(y, s) for y, s in zip(out.loc[has, "target_year"], out.loc[has, "target_season"])]
        out.loc[has, "target_period_end"] = [e.isoformat() for e in ends]
        out.loc[has, "data_age_years"] = [round((today - e).days / 365.25, 1) for e in ends]
    out["stale"] = out["data_age_years"].gt(1.0)
    out["model_version"] = version
    out["data_snapshot"] = (snapshot or {}).get("long_csv")
    out["snapshot_matches_training"] = (None if not (snapshot and trained_snapshot)
                                        else snapshot == trained_snapshot)
    out["generated_on"] = today.isoformat()
    out["_none"] = out["status"].str.startswith("NO FORECAST")
    return out.sort_values(["_none", "well_id"]).drop(columns="_none").reset_index(drop=True)


def forecast_rows_only(fc: pd.DataFrame) -> pd.DataFrame:
    return fc[~fc["status"].str.startswith("NO FORECAST")].reset_index(drop=True)
