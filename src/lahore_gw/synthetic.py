"""A fictional well for the public dashboard: generated numbers, not observations.

It lets the public view demonstrate the per-well charts and the decline-concern
wording without exposing any restricted reading. Every label says SYNTHETIC.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SYNTHETIC_ID = "SYNTHETIC-1"
SYNTHETIC_LABEL = "Synthetic example (fictional numbers, not a real well)"


def synthetic_well(seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Long table and wells table for one fictional well, 2005 to pre-monsoon 2020."""
    rng = np.random.default_rng(seed)
    rows = []
    for i, year in enumerate(range(2005, 2021)):
        for season in ("pre", "post"):
            if year == 2020 and season == "post":
                continue
            si = year * 2 + (season == "post")
            depth = 18.0 + 0.7 * i - (1.2 if season == "post" else 0.0) + rng.normal(0, 0.4)
            gap = (year, season) in {(2012, "post"), (2013, "pre")}       # two missing seasons, on purpose
            rows.append({"well_id": SYNTHETIC_ID, "year": year, "season": season, "season_index": si,
                         "depth_ft": np.nan if gap else round(depth, 2), "status": "dead" if gap else "reading"})
    long = pd.DataFrame(rows)
    long["depth_m"] = long["depth_ft"] * 0.3048
    wells = pd.DataFrame({"well_id": [SYNTHETIC_ID], "village": ["Fictional example"], "tehsil": ["Synthetic"],
                          "lat": [np.nan], "lon": [np.nan]})
    return long, wells
