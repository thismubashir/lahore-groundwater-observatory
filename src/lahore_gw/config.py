"""Central paths and constants used by every pipeline stage."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "WL-LHR"          # original files, never modified
PROCESSED_DIR = DATA_DIR / "processed"           # generated, git-ignored
EXTERNAL_DIR = DATA_DIR / "external"             # downloaded caches, git-ignored
POWER_CACHE_DIR = EXTERNAL_DIR / "nasa_power"
OPTIONAL_DIR = DATA_DIR / "optional"             # survey template, GRACE input

REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = ROOT / "models"

# District filter: the workbooks cover the whole Lahore *Irrigation Zone*
# (Sialkot, Kasur, Okara, ...). We keep only rows whose District is Lahore.
TARGET_DISTRICT = "lahore"

FT_TO_M = 0.3048

# Text markers in reading cells that mean "no reading was available".
# They are NOT zero depth and NOT evidence of aquifer depletion.
UNAVAILABLE_MARKERS = {"closed": "closed", "dead": "dead"}

# Seasonal timing assumption (the workbooks contain no measurement dates).
# Punjab piezometer surveys are conventionally done once before the monsoon
# (around May-June) and once after it (around October-November). We
# therefore assume the reading reflects conditions at the END of these
# windows and only use weather inside the window that ends at the reading.
#   pre-monsoon  reading of year Y  <- weather 1 Nov (Y-1) .. 31 May Y
#   post-monsoon reading of year Y  <- weather 1 Jun Y     .. 31 Oct Y
SEASON_WINDOWS = {
    "pre": {"start": (-1, 11, 1), "end": (0, 5, 31)},
    "post": {"start": (0, 6, 1), "end": (0, 10, 31)},
}

RANDOM_SEED = 42
