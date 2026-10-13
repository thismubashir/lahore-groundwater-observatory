"""Historical decline concern at monitored wells: where, what the data show, what could follow.

Everything here is computed from the observed readings (and NASA POWER rainfall);
no forecast model is involved.

Per well:
  trend_ft_per_year   Theil-Sen slope of PRE-monsoon depths against year (one
                      reading per year, so the monsoon cycle does not bias it),
                      with its approximate 95% range. Positive = getting deeper.
  monsoon_rise_ft     mean rise of the water table from pre- to post-monsoon
                      of the same year (recharge season)
  dry_season_fall_ft  mean fall from post-monsoon to the next pre-monsoon
  concern             High / Medium / No clear decline detected / Unknown (CONCERN_RULES)

Limits: the workbooks have no pumping, canal, river or land-use records, so
the data can show HOW a well is changing but cannot prove WHY. "No clear
decline detected" does not mean the water is safe or the supply sufficient.
2030 values are straight-line historical trend scenarios, not forecasts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import theilslopes

from .config import FT_TO_M

MIN_PRE_READINGS = 5          # pre-monsoon readings needed for a trend
MIN_SPAN_YEARS = 5
FAST_FT_PER_YEAR = 1.0        # deepening at least this fast -> High concern
# Indicator only: surface-mounted SUCTION pumps typically cannot lift water from
# deeper than about 7-8 m (23-26 ft). It does not apply to submersible or other
# deep-well pumps, and real limits vary by pump and site.
SUCTION_LIMIT_FT = 25.0
HORIZON_YEAR = 2030
STALE_YEARS = 3               # no reading for this long before the latest survey -> monitoring gap

NO_DECLINE = "No clear decline detected"
CONCERN_TITLE = "Historical decline concern at monitored wells"
CONCERN_RULES = {
    "High": f"pre-monsoon depth increased at {FAST_FT_PER_YEAR:.0f} ft/year or faster, "
            "and the whole 95% range of the trend is above zero",
    "Medium": f"pre-monsoon depth increased more slowly than {FAST_FT_PER_YEAR:.0f} ft/year, "
              "and the whole 95% range of the trend is above zero",
    NO_DECLINE: "the 95% range of the trend includes zero, or the water level rose. "
                "This does not mean the water is safe or the supply sufficient",
    "Unknown": f"fewer than {MIN_PRE_READINGS} pre-monsoon readings or under {MIN_SPAN_YEARS} years of record",
}
CONCERN_ORDER = ["High", "Medium", NO_DECLINE, "Unknown"]
CONCERN_COLORS = {"High": "#c62828", "Medium": "#ef8a17", NO_DECLINE: "#5f7d95", "Unknown": "#b8c0cc"}

RAINFALL_FINDING = "No clear rainfall decline was detected in the available data; causes cannot be confirmed."
DRIVERS_NOT_MEASURED = (
    "Drivers commonly cited for Lahore, such as pumping by municipal and private tubewells, less recharge "
    "from the Ravi River and canals, and paved surfaces that stop rain soaking in, are not recorded in "
    "these data. They cannot be confirmed or ranked for any well here.")


def _label(year, season) -> str:
    return f"{'Pre' if season == 'pre' else 'Post'}-monsoon {int(year)}"


def well_indicators(long: pd.DataFrame, wells: pd.DataFrame) -> pd.DataFrame:
    """One row per well with trend, seasonal balance, concern level and scenario."""
    latest_si = int(long["season_index"].max())
    ok = long[long["status"] == "reading"].sort_values("season_index")
    info = wells.set_index("well_id")
    rows = []
    for wid in sorted(long["well_id"].unique()):
        g = ok[ok["well_id"] == wid]
        r = {"well_id": wid, "village": info.loc[wid, "village"], "tehsil": info.loc[wid, "tehsil"],
             "lat": info.loc[wid, "lat"], "lon": info.loc[wid, "lon"], "n_readings": len(g)}
        if g.empty:
            rows.append(r)
            continue
        last = g.iloc[-1]
        r.update(first_reading=_label(g.iloc[0].year, g.iloc[0].season),
                 last_reading=_label(last.year, last.season), last_year=int(last.year),
                 last_depth_ft=float(last.depth_ft), last_depth_m=float(last.depth_ft) * FT_TO_M,
                 years_without_reading=(latest_si - int(last.season_index)) / 2)

        pre = g[g["season"] == "pre"]
        if len(pre) >= MIN_PRE_READINGS and pre.year.max() - pre.year.min() >= MIN_SPAN_YEARS:
            s, _, lo, hi = theilslopes(pre.depth_ft.to_numpy(), pre.year.to_numpy())
            r.update(trend_ft_per_year=s, trend_low=lo, trend_high=hi,
                     trend_from=int(pre.year.min()), trend_to=int(pre.year.max()))

        d = g.set_index("season_index")["depth_ft"]
        rise = [d[si] - d[si + 1] for si in d.index if si % 2 == 0 and si + 1 in d.index]
        fall = [d[si + 1] - d[si] for si in d.index if si % 2 == 1 and si + 1 in d.index]
        r.update(monsoon_rise_ft=np.mean(rise) if rise else np.nan, n_monsoons=len(rise),
                 dry_season_fall_ft=np.mean(fall) if fall else np.nan, n_dry_seasons=len(fall))
        rows.append(r)

    out = pd.DataFrame(rows)
    for c in ["trend_ft_per_year", "trend_low", "trend_high", "monsoon_rise_ft", "dry_season_fall_ft"]:
        if c not in out:
            out[c] = np.nan
    out["trend_class"] = np.select(
        [out.trend_ft_per_year.isna(), out.trend_low > 0, out.trend_high < 0],
        ["Insufficient data", "Deepening", "Rising"], "No clear trend")
    out["concern"] = np.select(
        [out.trend_class == "Insufficient data",
         (out.trend_class == "Deepening") & (out.trend_ft_per_year >= FAST_FT_PER_YEAR),
         out.trend_class == "Deepening"],
        ["Unknown", "High", "Medium"], NO_DECLINE)
    out["deeper_than_25ft"] = out["last_depth_ft"] > SUCTION_LIMIT_FT

    # straight-line historical trend scenario (deepening wells only): not a forecast
    deep = out.trend_class == "Deepening"
    yrs = HORIZON_YEAR - out["last_year"]
    out["projected_depth_ft"] = np.where(deep, out.last_depth_ft + out.trend_ft_per_year * yrs, np.nan)
    out["projected_low_ft"] = np.where(deep, out.last_depth_ft + out.trend_low * yrs, np.nan)
    out["projected_high_ft"] = np.where(deep, out.last_depth_ft + out.trend_high * yrs, np.nan)
    crosses = deep & (out.last_depth_ft <= SUCTION_LIMIT_FT)
    out["suction_limit_year"] = np.where(
        crosses, out.last_year + np.ceil((SUCTION_LIMIT_FT - out.last_depth_ft) / out.trend_ft_per_year), np.nan)
    out["concern"] = pd.Categorical(out["concern"], CONCERN_ORDER, ordered=True)
    return out.sort_values(["concern", "trend_ft_per_year"], ascending=[True, False]).reset_index(drop=True)


def rainfall_context(long: pd.DataFrame, weather: pd.DataFrame | None) -> dict | None:
    """Network-wide check: did monsoon rainfall decline, and is it associated with monsoon recovery?"""
    if weather is None or weather.empty:
        return None
    post = weather[(weather.season == "post")].dropna(subset=["rain_mm"])
    years = sorted(set(long.loc[long.status == "reading", "year"]))
    annual = post[post.year.between(min(years), max(years))].groupby("year")["rain_mm"].mean()
    s, _, lo, hi = theilslopes(annual.to_numpy(), annual.index.to_numpy())

    ok = long[long.status == "reading"]
    d = ok.pivot(index="well_id", columns="season_index", values="depth_ft")
    pairs = []
    for wid, ser in d.iterrows():
        for si in ser.index:
            if si % 2 == 0 and si + 1 in ser.index and np.isfinite(ser[si]) and np.isfinite(ser[si + 1]):
                pairs.append((wid, si // 2, ser[si] - ser[si + 1]))
    p = pd.DataFrame(pairs, columns=["well_id", "year", "rise"]).merge(
        post[["well_id", "year", "rain_mm"]], on=["well_id", "year"])
    p["rise_within"] = p.rise - p.groupby("well_id").rise.transform("mean")   # compare years within a well
    slope = np.polyfit(p.rain_mm, p.rise_within, 1)[0] * 100
    return {"rain_trend_mm_per_year": s, "rain_trend_low": lo, "rain_trend_high": hi,
            "rain_first_year": int(annual.index.min()), "rain_last_year": int(annual.index.max()),
            "mean_monsoon_rain_mm": float(annual.mean()),
            "rise_ft_per_100mm": float(slope), "rise_rain_corr": float(np.corrcoef(p.rain_mm, p.rise_within)[0, 1]),
            "n_monsoons": len(p)}


def rainfall_text(rain: dict | None) -> str:
    """Plain-English network-wide rainfall finding."""
    if rain is None:
        return "Weather data not available, so rainfall could not be checked."
    finding = ("Monsoon rainfall declined in the available data; causes cannot be confirmed."
               if rain["rain_trend_high"] < 0 else RAINFALL_FINDING)
    return (f"{finding} Monsoon rainfall (NASA POWER, 1 Jun-31 Oct, coarse grid) averaged "
            f"{rain['mean_monsoon_rain_mm']:.0f} mm over {rain['rain_first_year']}-{rain['rain_last_year']} "
            f"(trend {rain['rain_trend_mm_per_year']:+.0f} mm/year, 95% range {rain['rain_trend_low']:+.0f} to "
            f"{rain['rain_trend_high']:+.0f}). Wetter monsoons were associated with a larger monsoon rise in the "
            f"same well (about {rain['rise_ft_per_100mm']:.2f} ft per extra 100 mm, correlation "
            f"{rain['rise_rain_corr']:.2f}, {rain['n_monsoons']} well-monsoons). This is a predictive association, "
            "not a measured cause.")


def explain_why(r: pd.Series, rain: dict | None = None) -> list[str]:
    """What the readings show about how this well is changing (observations, not proven causes)."""
    out = []
    if r.trend_class == "Insufficient data":
        out.append(f"Too few readings ({r.n_readings}) to measure a long-term trend.")
    else:
        a, b = int(r.trend_from), int(r.trend_to)
        if r.trend_class == "No clear trend":
            out.append(f"From {a} to {b} the pre-monsoon water level showed no clear long-term change "
                       f"(trend {r.trend_ft_per_year:+.2f} ft/year, 95% range {r.trend_low:+.2f} to {r.trend_high:+.2f}).")
        else:
            what = "dropped (water got deeper)" if r.trend_class == "Deepening" else "rose (water got shallower)"
            out.append(f"From {a} to {b} the pre-monsoon water level {what} by about "
                       f"{abs(r.trend_ft_per_year):.2f} ft per year (95% range {abs(r.trend_low):.2f} to "
                       f"{abs(r.trend_high):.2f}), about {abs(r.trend_ft_per_year) * (b - a):.0f} ft in {b - a} years.")

    rise, fall = r.monsoon_rise_ft, r.dry_season_fall_ft
    if r.get("n_monsoons", 0) >= 3 and r.get("n_dry_seasons", 0) >= 3:
        net = fall - rise          # > 0: the dry season takes more than the monsoon gives back
        if net > 0.3 and rise <= 0.5:
            moved = f"dropped a further {-rise:.1f} ft" if rise < 0 else f"rose only {rise:.1f} ft"
            out.append(f"The monsoon brings almost no recovery: between pre- and post-monsoon readings the water "
                       f"level {moved} on average, while it dropped {fall:.1f} ft in the dry season. The monsoon does "
                       "not replace what the water table loses in the dry season.")
        elif net > 0.3:
            out.append(f"The monsoon recovery ({rise:.1f} ft on average) is smaller than the dry-season drop "
                       f"({fall:.1f} ft), so the well ends each year deeper than it started. This is consistent with "
                       "more water leaving the aquifer near this well than is recharged.")
        elif net < -0.3:
            out.append(f"The monsoon recovery ({rise:.1f} ft on average) is larger than the dry-season drop "
                       f"({fall:.1f} ft), so the water level did not fall from year to year.")
        else:
            out.append(f"Monsoon recovery ({rise:.1f} ft) and dry-season drop ({fall:.1f} ft) roughly balance.")

    if rain and r.trend_class == "Deepening":
        trend = (f"over {rain['rain_first_year']}-{rain['rain_last_year']} (trend {rain['rain_trend_mm_per_year']:+.0f} "
                 f"mm/year, 95% range {rain['rain_trend_low']:+.0f} to {rain['rain_trend_high']:+.0f})")
        if rain["rain_trend_high"] < 0:
            out.append(f"Monsoon rainfall (NASA POWER) declined {trend}; causes cannot be confirmed.")
        else:
            out.append(f"{RAINFALL_FINDING} Monsoon rainfall (NASA POWER) {trend}.")
    if r.trend_class == "Deepening":
        out.append(DRIVERS_NOT_MEASURED)
    return out


def future_risks(r: pd.Series, latest_label: str) -> list[str]:
    """Problems that could follow if the observed pattern continued (scenario, not a forecast)."""
    out = []
    if r.get("years_without_reading", 0) >= STALE_YEARS:
        out.append(f"Monitoring gap: no reading since {r.last_reading} ({r.years_without_reading:.0f} years before "
                   f"the latest survey, {latest_label}). The present condition here is unknown.")
    if r.deeper_than_25ft:
        out.append(f"At the last reading ({r.last_reading}) the water was {r.last_depth_ft:.0f} ft "
                   f"({r.last_depth_m:.1f} m) deep. That is beyond the typical ~{SUCTION_LIMIT_FT:.0f} ft (7-8 m) lift "
                   "of surface-mounted suction pumps, which may struggle or fail at such depths. Submersible and "
                   "other deep-well pumps are not limited this way; actual limits vary by pump and site.")
    if r.trend_class == "Deepening":
        out.append(f"Straight-line historical trend scenario: if the {int(r.trend_from)}-{int(r.trend_to)} trend "
                   f"continued, the depth would be about {r.projected_depth_ft:.0f} ft by {HORIZON_YEAR} (range "
                   f"{r.projected_low_ft:.0f}-{r.projected_high_ft:.0f} ft), against {r.last_depth_ft:.0f} ft at the "
                   "last reading. This is not a validated forecast.")
        if np.isfinite(r.suction_limit_year):
            out.append(f"In that scenario the depth would pass ~{SUCTION_LIMIT_FT:.0f} ft around "
                       f"{int(r.suction_limit_year)}, where surface-mounted suction pumps typically stop being able "
                       "to lift water.")
        extra = r.projected_depth_ft - r.last_depth_ft
        out.append(f"In that scenario, pumping lift would grow by about {extra:.0f} ft ({extra / r.last_depth_ft:.0%}), raising "
                   "pumping energy and cost roughly in proportion. Shallow bores may dry up and need deepening.")
        out.append("Water drawn from deeper layers can differ in quality. This project has no water-quality "
                   "data, so that risk cannot be assessed here.")
    elif r.trend_class in ("No clear trend", "Rising"):
        out.append("No clear decline was detected at this well, so no trend scenario is shown. This does not "
                   "mean the water is safe or the supply sufficient.")
    else:
        out.append("Not enough readings to project a risk.")
    return out


def network_summary(ind: pd.DataFrame) -> dict:
    t = ind.dropna(subset=["trend_ft_per_year"])
    return {"wells": len(ind), "with_trend": len(t),
            "counts": ind["concern"].value_counts().reindex(CONCERN_ORDER, fill_value=0).to_dict(),
            "deepening": int((ind.trend_class == "Deepening").sum()),
            "rising": int((ind.trend_class == "Rising").sum()),
            "median_trend_ft_per_year": float(t.trend_ft_per_year.median()) if len(t) else np.nan,
            "deeper_than_25ft": int(ind.deeper_than_25ft.sum())}
