"""Stage 2: download (or reuse cached) NASA POWER daily weather, aggregate by season.

Run:  python scripts/02_fetch_weather.py
Cache: data/external/nasa_power/*.json   (one file per location; git-ignored)
Output: data/processed/weather_seasonal_by_well.csv, weather_metadata.json
"""

import json

import _bootstrap  # noqa: F401
import pandas as pd

from lahore_gw.config import PROCESSED_DIR, REPORTS_DIR
from lahore_gw.weather import build_weather_table, save_weather


def main():
    wells = pd.read_csv(PROCESSED_DIR / "lahore_wells.csv")
    long = pd.read_csv(PROCESSED_DIR / "lahore_groundwater_long.csv")
    years = range(int(long.year.min()) - 1, int(long.year.max()) + 1)   # -1: previous window
    table, meta = build_weather_table(wells, years)
    save_weather(table, meta)
    summary = (table.groupby("season")[["rain_mm", "t2m_mean_c"]]
               .describe().round(1).T)
    (REPORTS_DIR / "weather_summary.md").write_text(
        "# NASA POWER seasonal weather summary\n\n"
        f"Distinct weather series across {table.well_id.nunique()} wells: "
        f"**{meta['distinct_weather_series']}** (coarse grid; see README).\n\n"
        "Windows: pre-monsoon = 1 Nov (previous year) – 31 May; post-monsoon = 1 Jun – 31 Oct.\n\n"
        "```\n" + summary.to_string() + "\n```\n\n"
        "Parameter metadata returned by the API:\n\n```json\n"
        + json.dumps(meta.get("parameters", {}), indent=2) + "\n```\n")
    print(f"weather rows={len(table)} distinct series={meta['distinct_weather_series']}")


if __name__ == "__main__":
    main()
