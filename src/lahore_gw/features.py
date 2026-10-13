"""Turn the long dataset into forecasting samples without look-ahead.

One sample = (well, origin season t) -> target = depth at the NEXT season t+1.
A sample exists only when both t and t+1 have real readings and are
consecutive seasons (e.g. pre-2010 -> post-2010, post-2010 -> pre-2011).

Every feature is known by the end of season t's observation window:
  depth_t_ft           latest observed depth (ft below ground)
  delta_prev_ft        depth_t - depth_(t-1)       (+ has_prev flag)
  delta_yoy_ft         depth_t - depth_(t-2)       (+ has_yoy flag; same season last year)
  target_is_post       1 if the target is a post-monsoon reading (calendar fact)
  rain_t_mm, t2m_t_c   NASA POWER weather in the window that ENDS at season t
  rain_prev_mm         NASA POWER rainfall in the window before that
  lat, lon, ground_elevation_m   static location descriptors
The model predicts the change (depth_(t+1) - depth_t); the depth forecast is
depth_t + predicted change. Weather of the target window is NOT used because
it would not be known on the forecast date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "depth_t_ft", "delta_prev_ft", "has_prev", "delta_yoy_ft", "has_yoy",
    "target_is_post", "rain_t_mm", "rain_prev_mm", "t2m_t_c",
    "lat", "lon", "ground_elevation_m",
]

WEATHER_FEATURES = ["rain_t_mm", "rain_prev_mm", "t2m_t_c"]

# Features that are moved together in permutation importance.
FEATURE_GROUPS = {
    "Latest depth": ["depth_t_ft"],
    "Last seasonal change": ["delta_prev_ft", "has_prev"],
    "Change vs same season last year": ["delta_yoy_ft", "has_yoy"],
    "Target season (pre/post)": ["target_is_post"],
    "Rainfall, window ending at origin": ["rain_t_mm"],
    "Rainfall, previous window": ["rain_prev_mm"],
    "Temperature, window ending at origin": ["t2m_t_c"],
    "Location (lat/lon)": ["lat", "lon"],
    "Ground elevation": ["ground_elevation_m"],
}


def build_samples(long: pd.DataFrame, wells: pd.DataFrame, weather: pd.DataFrame | None,
                  include_unlabeled_last: bool = False) -> pd.DataFrame:
    """Create one row per (well, origin season) with features and target.

    include_unlabeled_last=True also returns rows whose target is not yet
    observed (used only for the clearly-labelled forward forecast).
    """
    L = long.sort_values(["well_id", "season_index"])
    piv = L.pivot(index="well_id", columns="season_index", values="depth_ft")
    all_si = sorted(L["season_index"].unique())
    si_set = set(all_si)
    rows = []
    for wid, series in piv.iterrows():
        for si in all_si:
            d_t = series.get(si, np.nan)
            if np.isnan(d_t):
                continue
            target_si = si + 1
            d_next = series.get(target_si, np.nan)
            has_target = not np.isnan(d_next)
            if not has_target and not include_unlabeled_last:
                continue
            if not has_target and target_si in si_set:
                # the next season was surveyed but this well had no reading
                continue
            d_prev = series.get(si - 1, np.nan)
            d_yoy = series.get(si - 2, np.nan)
            rows.append({
                "well_id": wid,
                "origin_si": si, "target_si": target_si,
                "origin_year": si // 2, "origin_season": "pre" if si % 2 == 0 else "post",
                "target_year": target_si // 2, "target_season": "pre" if target_si % 2 == 0 else "post",
                "depth_t_ft": d_t,
                "delta_prev_ft": d_t - d_prev if not np.isnan(d_prev) else np.nan,
                "has_prev": float(not np.isnan(d_prev)),
                "delta_yoy_ft": d_t - d_yoy if not np.isnan(d_yoy) else np.nan,
                "has_yoy": float(not np.isnan(d_yoy)),
                "target_is_post": float(target_si % 2 == 1),
                "same_season_last_year_ft": series.get(si - 1, np.nan),
                "target_depth_ft": d_next,
                "target_change_ft": d_next - d_t if has_target else np.nan,
            })
    S = pd.DataFrame(rows)
    S = S.merge(wells[["well_id", "lat", "lon", "ground_elevation_m", "tehsil", "village"]],
                on="well_id", how="left")
    if weather is not None and len(weather):
        w = weather.copy()
        w["si"] = w["year"] * 2 + (w["season"] == "post").astype(int)
        cur = w[["well_id", "si", "rain_mm", "t2m_mean_c"]].rename(
            columns={"si": "origin_si", "rain_mm": "rain_t_mm", "t2m_mean_c": "t2m_t_c"})
        prev = w[["well_id", "si", "rain_mm"]].rename(columns={"rain_mm": "rain_prev_mm"})
        prev["origin_si"] = prev.pop("si") + 1
        S = S.merge(cur, on=["well_id", "origin_si"], how="left")
        S = S.merge(prev, on=["well_id", "origin_si"], how="left")
    else:
        S["rain_t_mm"] = np.nan
        S["rain_prev_mm"] = np.nan
        S["t2m_t_c"] = np.nan
    return S.sort_values(["target_si", "well_id"]).reset_index(drop=True)


def chronological_split(S: pd.DataFrame, train_last_year: int, val_last_year: int):
    """Split by TARGET year so no test-period readings reach training."""
    train = S[S["target_year"] <= train_last_year]
    val = S[(S["target_year"] > train_last_year) & (S["target_year"] <= val_last_year)]
    test = S[S["target_year"] > val_last_year]
    return train, val, test
