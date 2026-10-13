"""OPTIONAL and SEPARATE: compare GRACE regional storage anomalies with well trends.

Why separate? GRACE/GRACE-FO satellites measure changes in *total terrestrial water
storage* (soil moisture + groundwater + surface water) averaged over roughly
300 km or more (mascon cells). They do not measure the depth to water in any
individual well, and one mascon covers Lahore together with large parts of
Punjab, India and surrounding districts. GRACE therefore cannot be a feature for
per-well, per-season forecasts without strong extra assumptions, so the models
do not use it.

What this script does (only if you provide a file): it plots the regional GRACE
anomaly next to the median Lahore well depth anomaly, as two separate panels
(no shared axis), for qualitative context.

Input: data/optional/grace_tws.csv with columns `date` (YYYY-MM-DD) and
`tws_anomaly_cm`, e.g. exported for a point near Lahore (31.5 N, 74.3 E) from
NASA's GRACE Tellus / PO.DAAC mascon products (requires a free NASA Earthdata login).
"""

import sys

import _bootstrap  # noqa: F401
import pandas as pd

from lahore_gw import plots  # noqa: F401  (sets plot style)
from lahore_gw.config import FIGURES_DIR, OPTIONAL_DIR, PROCESSED_DIR


def main():
    path = OPTIONAL_DIR / "grace_tws.csv"
    if not path.exists():
        print(f"No GRACE file at {path}. This analysis is optional; see the docstring.")
        return 0
    import matplotlib.pyplot as plt
    g = pd.read_csv(path, parse_dates=["date"])
    long = pd.read_csv(PROCESSED_DIR / "lahore_groundwater_long.csv")
    ok = long[long.status == "reading"].copy()
    ok["anom"] = ok.depth_ft - ok.groupby("well_id").depth_ft.transform("mean")
    med = ok.groupby("assumed_period_end").anom.median().reset_index()
    med["date"] = pd.to_datetime(med.assumed_period_end)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    a1.plot(g.date, g.tws_anomaly_cm, color=plots.SERIES[0])
    a1.set_ylabel("GRACE TWS anomaly (cm)")
    a1.set_title("Regional storage (GRACE, ~300 km footprint)")
    a2.plot(med.date, med.anom, marker="o", color=plots.SERIES[1])
    a2.invert_yaxis()
    a2.set_ylabel("Median well depth anomaly (ft)\n(down = deeper)")
    a2.set_title("Lahore piezometers (point measurements)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "optional_grace_context.png", dpi=150)
    print("Saved optional_grace_context.png (qualitative context only)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
