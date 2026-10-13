"""Aggregate-only summary for the public dashboard and README (no per-well data).

Written by scripts/05_water_level_risk.py to reports/public_summary.json. It
holds counts, periods and network-level statistics only: no well identifiers,
coordinates, village names or individual readings, so it can be published
while the source workbooks' redistribution terms are unverified.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import water_risk as wr
from .dashboard_data import coordinate_flag_breakdown, observation_period, season_label

WELL_ID = re.compile(r"LHR-\d+")


def _plain(v):
    """JSON-safe Python values."""
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else round(float(v), 4)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def build(long: pd.DataFrame, wells: pd.DataFrame, weather: pd.DataFrame | None,
          disc: pd.DataFrame, raw_dir: Path | None = None) -> dict:
    risk = wr.well_indicators(long, wells)
    summ = wr.network_summary(risk)
    first, last = observation_period(long)
    cov = (long.groupby("season_index")["status"].value_counts().unstack(fill_value=0)
           .reindex(columns=["reading", "closed", "dead", "blank", "not_listed"], fill_value=0).reset_index())
    cov.insert(0, "season", [season_label(si // 2, "pre" if si % 2 == 0 else "post") for si in cov.season_index])
    br = coordinate_flag_breakdown(wells, disc)
    reads = long[long.status == "reading"]
    sources = []
    for f, g in reads.groupby("chosen_from"):
        a, b = g.season_index.min(), g.season_index.max()
        p = (raw_dir / f) if raw_dir else None
        sources.append({"file": f, "readings": len(g),
                        "seasons": f"{season_label(a // 2, 'pre' if a % 2 == 0 else 'post')} – "
                                   f"{season_label(b // 2, 'pre' if b % 2 == 0 else 'post')}",
                        "sha256_16": hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p and p.exists() else None,
                        "first_index": int(a)})
    sources = sorted(sources, key=lambda s: s.pop("first_index"))
    out = {
        "generated_on": dt.date.today().isoformat(),
        "note": "Aggregate counts only; no per-well identifiers, coordinates or readings.",
        "wells": len(wells), "tehsils": int(wells.tehsil.nunique()),
        "readings": len(reads), "well_seasons": len(long),
        "status_counts": long.status.value_counts().to_dict(),
        "observation_period": [first, last],
        "latest_survey": season_label(*long.loc[long.season_index.idxmax(), ["year", "season"]]),
        "coverage_by_season": cov.drop(columns="season_index").to_dict("records"),
        "concern": {"title": wr.CONCERN_TITLE, "order": wr.CONCERN_ORDER, "rules": wr.CONCERN_RULES,
                    **{k: summ[k] for k in ["counts", "deepening", "rising", "with_trend",
                                            "median_trend_ft_per_year", "deeper_than_25ft"]}},
        "rainfall": wr.rainfall_context(long, weather),
        "coordinates": {"files_disagree": len(br["files_disagree"]),
                        "disagree_over_threshold": len(br["disagree_over_threshold"]),
                        "disagree_below_threshold": len(br["disagree_below_threshold"]),
                        "same_village_only": len(br["same_village_only"]), "flagged": len(br["flagged"]),
                        "flag_km": br["flag_km"],
                        "max_difference_km": float(disc.distance_km.max()) if len(disc) else 0.0},
        "elevation": {"unresolved": int(wells.elevation_status.str.startswith("unresolved").sum()),
                      "missing": int(wells.elevation_status.str.startswith("missing").sum())},
        "weather_series": int(weather.groupby("well_id")["rain_mm"].apply(
            lambda s: tuple(np.round(s.fillna(-1), 3))).nunique()) if weather is not None else None,
        "sources": sources,
    }
    out = _plain(out)
    assert not WELL_ID.search(json.dumps(out)), "public summary must not contain well identifiers"
    return out


def write(summary: dict, path: Path):
    path.write_text(json.dumps(summary, indent=1, ensure_ascii=False))


def load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None
