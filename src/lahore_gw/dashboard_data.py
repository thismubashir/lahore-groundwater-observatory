"""Data helpers for the Streamlit dashboard (kept separate so they can be tested)."""

from __future__ import annotations

import json

import pandas as pd

from .config import FT_TO_M, PROCESSED_DIR, REPORTS_DIR

FORECAST_LABEL = ("Unverified model forecast for a HISTORICAL season - "
                  "not a measured or current groundwater level")


def required_files():
    return [PROCESSED_DIR / "lahore_wells.csv", PROCESSED_DIR / "lahore_groundwater_long.csv"]


def data_available() -> bool:
    return all(p.exists() for p in required_files())


def load_dashboard_data(processed_dir=PROCESSED_DIR, reports_dir=REPORTS_DIR) -> dict:
    wells = pd.read_csv(processed_dir / "lahore_wells.csv")
    long = pd.read_csv(processed_dir / "lahore_groundwater_long.csv")
    fwd_path = processed_dir / "forward_forecast.csv"
    fwd = pd.read_csv(fwd_path) if fwd_path.exists() else pd.DataFrame()
    if "nn_forecast_depth_ft" in fwd.columns:
        fwd = fwd[fwd["nn_forecast_depth_ft"].notna()].reset_index(drop=True)
    met_path = reports_dir / "metrics.csv"
    metrics = pd.read_csv(met_path) if met_path.exists() else pd.DataFrame()
    test_path = processed_dir / "test_predictions.csv"
    test = pd.read_csv(test_path) if test_path.exists() else pd.DataFrame()
    wx_path = processed_dir / "weather_seasonal_by_well.csv"
    weather = pd.read_csv(wx_path) if wx_path.exists() else None

    def opt_csv(path):
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    meta_path = processed_dir / "weather_metadata.json"
    return {"wells": wells, "long": long, "forecast": fwd, "metrics": metrics, "test": test, "weather": weather,
            "comparisons": opt_csv(reports_dir / "metrics_vs_persistence.csv"),
            "importance_nn": opt_csv(reports_dir / "feature_importance_nn.csv"),
            "importance_ridge": opt_csv(reports_dir / "feature_importance_ridge.csv"),
            "coord_discrepancies": opt_csv(reports_dir / "audit" / "coordinate_discrepancies.csv"),
            "sensitivity": opt_csv(reports_dir / "sensitivity.csv"),
            "weather_meta": json.loads(meta_path.read_text()) if meta_path.exists() else {}}


def season_label(year, season) -> str:
    return f"{'Pre' if season == 'pre' else 'Post'}-monsoon {int(year)}"


def observation_period(long: pd.DataFrame) -> tuple[str, str]:
    """First and last season with at least one real reading."""
    r = long[long["status"] == "reading"].sort_values("season_index")
    return season_label(r.iloc[0].year, r.iloc[0].season), season_label(r.iloc[-1].year, r.iloc[-1].season)


def headline_model_row(metrics: pd.DataFrame, prefix: str = "Ridge regression") -> pd.Series | None:
    """The held-out test row of a model (by name prefix), as reported by stage 3."""
    if metrics.empty:
        return None
    t = metrics[(metrics["split"] == "test") & metrics["model"].str.startswith(prefix)]
    return t.iloc[0] if len(t) else None


def to_units(feet, units: str):
    """Depths are stored in feet; convert for display."""
    return feet * FT_TO_M if units == "m" else feet


def coordinate_flag_breakdown(wells: pd.DataFrame, disc: pd.DataFrame, flag_km: float = 1.0) -> dict:
    """Why 'files disagree' and 'coordinate-flagged' give different well counts.

    disc lists every well whose coordinates differ between files (any distance);
    a well is flagged when files disagree by more than flag_km or it lies far
    from a same-village well.
    """
    ids = lambda gis: {f"LHR-{int(g):03d}" for g in gis}  # noqa: E731
    disagree = ids(disc["gis_no"]) if len(disc) else set()
    big = ids(disc.loc[disc["distance_km"] > flag_km, "gis_no"]) if len(disc) else set()
    flagged = set(wells.loc[wells["coord_quality"] != "consistent", "well_id"])
    village = set(wells.loc[wells["coord_quality"].str.contains("same-village", na=False), "well_id"])
    return {"files_disagree": sorted(disagree), "disagree_over_threshold": sorted(big),
            "disagree_below_threshold": sorted(disagree - big), "same_village_only": sorted(village - big),
            "flagged": sorted(flagged), "flag_km": flag_km}


def well_summary(wells: pd.DataFrame, long: pd.DataFrame) -> pd.DataFrame:
    """Last real reading per well (never a forecast)."""
    ok = long[long["status"] == "reading"].sort_values("season_index")
    last = ok.groupby("well_id").tail(1)[["well_id", "year", "season", "depth_ft"]]
    last = last.rename(columns={"year": "last_reading_year", "season": "last_reading_season",
                                "depth_ft": "last_depth_ft"})
    n = ok.groupby("well_id").size().rename("n_readings").reset_index()
    latest_si = long["season_index"].max()
    latest_status = (long[long["season_index"] == latest_si][["well_id", "status"]]
                     .rename(columns={"status": "status_in_latest_season"}))
    out = wells.merge(last, on="well_id", how="left").merge(n, on="well_id", how="left")
    out = out.merge(latest_status, on="well_id", how="left")
    out["n_readings"] = out["n_readings"].fillna(0).astype(int)
    return out


def well_history(long: pd.DataFrame, well_id: str) -> pd.DataFrame:
    g = long[(long["well_id"] == well_id) & (long["status"] == "reading")].copy()
    g["time"] = g["year"] + g["season"].map({"pre": 0.4, "post": 0.8})
    g["season_label"] = g["season"].map({"pre": "Pre-monsoon", "post": "Post-monsoon"}) + " " + g["year"].astype(str)
    g["kind"] = "Observed reading"
    return g.sort_values("season_index")


def well_forecast(fwd: pd.DataFrame, well_id: str) -> pd.DataFrame:
    if fwd.empty:
        return fwd
    f = fwd[fwd["well_id"] == well_id].copy()
    if f.empty:
        return f
    f["time"] = f["target_year"] + f["target_season"].map({"pre": 0.4, "post": 0.8})
    f["depth_ft"] = f["nn_forecast_depth_ft"]
    f["season_label"] = f["target_season"].map({"pre": "Pre-monsoon", "post": "Post-monsoon"}) + " " + \
        f["target_year"].astype(str)
    f["kind"] = "Model forecast (unverified)"
    return f


def public_mode_requested(env=None) -> bool:
    """LGO_PUBLIC_MODE=1 (or true/yes) forces the aggregate-only public view."""
    import os
    env = os.environ if env is None else env
    return str(env.get("LGO_PUBLIC_MODE", "")).strip().lower() in {"1", "true", "yes"}


def load_public_data(reports_dir=REPORTS_DIR) -> dict:
    """Only files that can be published: aggregate summary, metrics, comparisons, importance, sensitivity."""
    def opt_csv(path):
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    p = reports_dir / "public_summary.json"
    return {"summary": json.loads(p.read_text()) if p.exists() else None,
            "metrics": opt_csv(reports_dir / "metrics.csv"),
            "comparisons": opt_csv(reports_dir / "metrics_vs_persistence.csv"),
            "importance_ridge": opt_csv(reports_dir / "feature_importance_ridge.csv"),
            "importance_nn": opt_csv(reports_dir / "feature_importance_nn.csv"),
            "sensitivity": opt_csv(reports_dir / "sensitivity.csv")}
