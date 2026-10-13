import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lahore_gw.config import PROCESSED_DIR, RAW_DIR  # noqa: E402

needs_raw = pytest.mark.skipif(not any(RAW_DIR.glob("*.xlsx")) if RAW_DIR.exists() else True,
                               reason="original workbooks not present (see README: Data sources)")
needs_processed = pytest.mark.skipif(not (PROCESSED_DIR / "lahore_groundwater_long.csv").exists(),
                                     reason="run scripts/01_clean_data.py first")
