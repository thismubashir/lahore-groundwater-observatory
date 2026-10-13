"""NASA POWER daily rainfall/temperature: download, cache, seasonal aggregation.

Source: NASA Langley Research Center POWER Project, Daily API
https://power.larc.nasa.gov/docs/services/api/temporal/daily/
Parameters:
  PRECTOTCORR  bias-corrected precipitation, mm/day
  T2M          mean air temperature at 2 m, deg C
  T2M_MAX      daily maximum air temperature at 2 m, deg C
Spatial limitation: POWER meteorology is gridded at roughly 0.5 x 0.625 deg
(about 50-60 km), so all Lahore wells fall in only a few grid cells. It
cannot describe neighbourhood-scale differences in rainfall.
"""

from __future__ import annotations

import json
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests   # bundles CA certificates (python.org macOS builds lack them)

from .config import POWER_CACHE_DIR, PROCESSED_DIR, SEASON_WINDOWS

API = "https://power.larc.nasa.gov/api/temporal/daily/point"
PARAMS = ["PRECTOTCORR", "T2M", "T2M_MAX"]
START, END = "20020601", "20201231"   # covers every window used, incl. post-monsoon 2002


def cache_path(lat: float, lon: float):
    return POWER_CACHE_DIR / f"power_daily_{lat:.4f}_{lon:.4f}_{START}_{END}.json"


def fetch_point(lat: float, lon: float, retries: int = 3, timeout: int = 120) -> dict:
    """Download (or load from cache) one point's daily series."""
    path = cache_path(lat, lon)
    if path.exists():
        return json.loads(path.read_text())
    POWER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    query = {
        "parameters": ",".join(PARAMS), "community": "AG",
        "latitude": f"{lat:.4f}", "longitude": f"{lon:.4f}",
        "start": START, "end": END, "format": "JSON",
    }
    last_err = None
    for attempt in range(retries):
        try:
            resp = requests.get(API, params=query, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            path.write_text(json.dumps(data))
            return data
        except Exception as err:          # network errors, rate limits
            last_err = err
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"NASA POWER download failed for {lat},{lon}: {last_err}")


def to_daily_frame(data: dict) -> pd.DataFrame:
    fill = data.get("header", {}).get("fill_value", -999.0)
    param = data["properties"]["parameter"]
    df = pd.DataFrame({k: pd.Series(v) for k, v in param.items()})
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    return df.replace(fill, np.nan).sort_index()


def window_bounds(year: int, season: str, shift_days: int = 0) -> tuple[date, date]:
    """Weather window for a reading; shift_days > 0 moves the whole window earlier (sensitivity checks)."""
    w = SEASON_WINDOWS[season]
    (sy, sm, sd), (ey, em, ed) = w["start"], w["end"]
    shift = timedelta(days=shift_days)
    return date(year + sy, sm, sd) - shift, date(year + ey, em, ed) - shift


def seasonal_aggregates(daily: pd.DataFrame, years, seasons=("pre", "post"), shift_days: int = 0) -> pd.DataFrame:
    rows = []
    for y in years:
        for s in seasons:
            a, b = window_bounds(y, s, shift_days)
            w = daily.loc[str(a):str(b)]
            n_expected = (b - a).days + 1
            complete = len(w) == n_expected and w["PRECTOTCORR"].notna().all()
            rows.append({"year": y, "season": s,
                         "window_start": a.isoformat(), "window_end": b.isoformat(),
                         "rain_mm": w["PRECTOTCORR"].sum() if complete else np.nan,
                         "t2m_mean_c": w["T2M"].mean() if complete else np.nan,
                         "t2m_max_mean_c": w["T2M_MAX"].mean() if complete else np.nan,
                         "days": len(w)})
    return pd.DataFrame(rows)


def build_weather_table(wells: pd.DataFrame, years, shift_days: int = 0) -> tuple[pd.DataFrame, dict]:
    """Seasonal weather per well (from that well's POWER grid cell)."""
    out, meta = [], {}
    for _, w in wells.dropna(subset=["lat", "lon"]).iterrows():
        data = fetch_point(w["lat"], w["lon"])
        agg = seasonal_aggregates(to_daily_frame(data), years, shift_days=shift_days)
        agg.insert(0, "well_id", w["well_id"])
        out.append(agg)
        if not meta:
            meta = {"parameters": data.get("parameters", {}), "header": data.get("header", {})}
    table = pd.concat(out, ignore_index=True)
    # How many genuinely different weather series do we have?
    sig = table.groupby("well_id")["rain_mm"].apply(lambda s: tuple(np.round(s.fillna(-1), 3)))
    meta["distinct_weather_series"] = int(sig.nunique())
    return table, meta


def save_weather(table: pd.DataFrame, meta: dict):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(PROCESSED_DIR / "weather_seasonal_by_well.csv", index=False)
    (PROCESSED_DIR / "weather_metadata.json").write_text(json.dumps(meta, indent=2))
