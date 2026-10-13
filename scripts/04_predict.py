"""Stage 4: next-season forecasts from the saved models (no retraining).

Run:  python scripts/04_predict.py                    (all wells)
      python scripts/04_predict.py --well <WELL_ID> [<WELL_ID> ...]
Output: data/processed/forecast_request.csv (one row per requested well: a
        forecast, or the reason there is none) and a printed summary.

Only the season after the latest surveyed season (post-monsoon 2020 for the
current files) is supported. These are historical-season model outputs, not
current groundwater levels.
"""

import argparse
import json

import _bootstrap  # noqa: F401
import pandas as pd

from lahore_gw.config import MODELS_DIR, PROCESSED_DIR
from lahore_gw.prediction import data_snapshot, forecast_next_season, load_models, model_version


def main(well_ids=None, out=PROCESSED_DIR / "forecast_request.csv"):
    long = pd.read_csv(PROCESSED_DIR / "lahore_groundwater_long.csv")
    wells = pd.read_csv(PROCESSED_DIR / "lahore_wells.csv")
    wpath = PROCESSED_DIR / "weather_seasonal_by_well.csv"
    weather = pd.read_csv(wpath) if wpath.exists() else None
    models, info = load_models()
    snap = data_snapshot()
    fc = forecast_next_season(long, wells, weather, models, well_ids=well_ids, version=model_version(),
                              snapshot=snap, trained_snapshot=info.get("data_snapshot"))
    fc.to_csv(out, index=False)

    print(f"model version {fc.model_version.iloc[0]}  (trained {info.get('created_utc', 'unknown')})")
    if info.get("data_snapshot") is None:
        print("WARNING: models predate snapshot recording; retrain to confirm they match the data.")
    elif snap != info["data_snapshot"]:
        print("WARNING: processed data differ from the snapshot the models were trained on:\n"
              f"  now {json.dumps(snap)}\n  trained {json.dumps(info['data_snapshot'])}")
    show = ["well_id", "village", "origin_season", "origin_year", "depth_t_ft", "target_season",
            "target_year", "nn_forecast_depth_ft", "ridge_forecast_depth_ft", "data_age_years", "status"]
    with pd.option_context("display.width", 200, "display.max_colwidth", 80):
        print(fc[show].round(2).to_string(index=False))
    print(f"\nSaved {len(fc)} rows to {out.relative_to(MODELS_DIR.parent)}")
    return fc


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--well", nargs="+", help="well ids in the form LHR-<GIS no.> (default: all)")
    main(ap.parse_args().well)
