"""Read the original Punjab piezometer workbooks without modifying them.

Every workbook has a slightly different layout, so instead of hard-coding
column names we *discover* them:

* the header row is the first row whose second cell is "GIS. No."
  (row index 1 for the 2003-2015 file, as expected);
* season columns are recognised from names such as ``Pre-03``, ``Po.09``,
  ``Pr.2010``, ``Post-2011``, ``Premonsoon\\n2014``;
* the six coordinate cells (D, M, S, D, M, S) sit right after "GIS. No.";
  latitude vs longitude is decided from the degree value because one file
  labels latitude as "X-coordinate".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RAW_DIR, UNAVAILABLE_MARKERS

SEASON_RE = re.compile(
    r"^(?P<season>premonsoon|postmonsoon|pre|post|pr|po)[\s.\-]*(?P<year>\d{4}|\d{2})$"
)


@dataclass
class WorkbookInfo:
    path: Path
    sheet: str
    header_row: int
    elevation_header: str
    season_columns: dict          # original column name -> (year, season)


def parse_season_column(name) -> tuple[int, str] | None:
    """Return (year, 'pre'|'post') for a reading column name, else None."""
    if not isinstance(name, str):
        return None
    text = re.sub(r"\s+", "", name.strip().lower())
    m = SEASON_RE.match(text)
    if not m:
        return None
    season = "pre" if m["season"].startswith("pr") else "post"
    year = int(m["year"])
    if year < 100:
        year += 2000
    return year, season


def find_header_row(path: Path, sheet=0, max_rows: int = 8) -> int:
    probe = pd.read_excel(path, sheet_name=sheet, header=None, nrows=max_rows, dtype=object)
    for i, row in probe.iterrows():
        if isinstance(row.iloc[1], str) and row.iloc[1].strip().lower().startswith("gis"):
            return int(i)
    raise ValueError(f"Could not find a 'GIS. No.' header row in {path.name}")


def discover_workbooks(raw_dir: Path = RAW_DIR) -> list[Path]:
    """List the actual Excel files present (nothing is assumed)."""
    return sorted(p for p in raw_dir.glob("*.xlsx") if not p.name.startswith("~$"))


def to_float(value) -> float:
    """Convert numbers and numeric strings such as '074' to float."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    try:
        return float(str(value).strip())
    except ValueError:
        return np.nan


def dms_to_decimal(d, m, s) -> float:
    """Degrees/minutes/seconds -> decimal degrees, NaN if invalid."""
    d, m, s = to_float(d), to_float(m), to_float(s)
    if any(np.isnan(v) for v in (d, m, s)):
        return np.nan
    if not (0 <= m < 60 and 0 <= s < 60):
        return np.nan
    return d + m / 60.0 + s / 3600.0


def classify_reading(value) -> tuple[float, str]:
    """Return (depth_ft or NaN, status).

    status is one of: 'reading', 'closed', 'dead', 'blank', 'unparsed'.
    Unavailable readings stay NaN; they are never converted to zero.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan, "blank"
    if isinstance(value, str):
        text = value.strip().lower()
        if text == "":
            return np.nan, "blank"
        if text in UNAVAILABLE_MARKERS:
            return np.nan, UNAVAILABLE_MARKERS[text]
    number = to_float(value)
    if np.isnan(number):
        return np.nan, "unparsed"
    return number, "reading"


def read_workbook(path: Path) -> tuple[pd.DataFrame, WorkbookInfo]:
    """Read one workbook into a tidy *wide* frame with standard column names."""
    xls = pd.ExcelFile(path)
    if len(xls.sheet_names) != 1:
        raise ValueError(f"{path.name}: expected one sheet, found {xls.sheet_names}")
    sheet = xls.sheet_names[0]
    header_row = find_header_row(path, sheet)
    df = pd.read_excel(path, sheet_name=sheet, header=header_row, dtype=object)
    cols = list(df.columns)

    gis_idx = next(i for i, c in enumerate(cols) if str(c).strip().lower().startswith("gis"))
    elev_col = next(c for c in cols if str(c).strip().upper().startswith("SEA LEVEL"))

    def col(prefix):
        return next(c for c in cols if str(c).replace("\n", " ").strip().lower().startswith(prefix))

    season_cols = {c: parse_season_column(c) for c in cols}
    season_cols = {c: v for c, v in season_cols.items() if v is not None}

    # Some sheets have a units row ("D", "M", "S") directly under the header.
    df = df[pd.to_numeric(df[cols[gis_idx]], errors="coerce").notna()].copy()

    coord = df[cols[gis_idx + 1: gis_idx + 7]].to_numpy()
    first = np.array([dms_to_decimal(*r[0:3]) for r in coord])
    second = np.array([dms_to_decimal(*r[3:6]) for r in coord])
    # Pakistan: latitude ~24-37 N, longitude ~60-78 E. Decide per file.
    if np.nanmedian(first) < np.nanmedian(second):
        lat, lon = first, second
    else:
        lat, lon = second, first

    out = pd.DataFrame({
        "gis_no": pd.to_numeric(df[cols[gis_idx]]).astype(int).to_numpy(),
        "dlr_no": df[col("dlr")].astype(str).str.strip().str.upper().to_numpy(),
        "district": df[col("district")].astype(str).str.strip().to_numpy(),
        "tehsil": df[col("tehsil")].astype(str).str.strip().to_numpy(),
        "village": df[col("village")].astype(str).str.strip().to_numpy(),
        "lat": lat,
        "lon": lon,
        "elevation_raw": [to_float(v) for v in df[elev_col]],
    })
    for c in season_cols:
        out[c] = df[c].to_numpy()

    info = WorkbookInfo(path=path, sheet=sheet, header_row=header_row,
                        elevation_header=str(elev_col), season_columns=season_cols)
    return out, info


def wide_to_long(wide: pd.DataFrame, info: WorkbookInfo) -> pd.DataFrame:
    """Melt season columns into one row per well x year x season."""
    id_cols = [c for c in wide.columns if c not in info.season_columns]
    rows = []
    for _, r in wide.iterrows():
        for col_name, (year, season) in info.season_columns.items():
            depth, status = classify_reading(r[col_name])
            rec = {c: r[c] for c in id_cols}
            rec.update(year=year, season=season, depth_ft=depth, status=status,
                       raw_value=None if pd.isna(r[col_name]) else str(r[col_name]).strip(),
                       source_file=info.path.name, source_column=str(col_name).replace("\n", " ").strip())
            rows.append(rec)
    return pd.DataFrame(rows)
