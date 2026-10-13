"""Read the 2019 readings from 'WL Lahore Zone 2019.pdf' (text-based PDF).

pdfplumber's table extractor returns one 22-cell row per piezometer with the
same column order as the Excel workbooks:
  0 Sr | 1 GIS | 2-4 North D M S | 5-7 East D M S | 8 SEA LEVEL (m) | 9 DLR |
  10-15 canal/admin | 16 Tehsil | 17 District | 18 Village | 19 Address |
  20 Pre-2019 | 21 Post-2019
The header text is checked on every page so a layout change fails loudly.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .config import RAW_DIR, TARGET_DISTRICT
from .raw_loading import WorkbookInfo, dms_to_decimal, parse_season_column, to_float, wide_to_long

PDF_NAME_PATTERN = "*.pdf"


def _clean(cell) -> str:
    return re.sub(r"\s+", " ", (cell or "").replace("\n", " ")).strip()


def read_pdf(path) -> tuple[pd.DataFrame, WorkbookInfo]:
    import pdfplumber

    rows, season_names = [], None
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                for raw in table:
                    cells = [_clean(c) for c in raw]
                    if len(cells) != 22:
                        continue
                    seasons = [parse_season_column(c) for c in cells[20:22]]
                    if all(seasons):
                        season_names = cells[20:22]   # header row, e.g. 'Pre- 2019'
                        continue
                    if not cells[1].isdigit():
                        continue
                    rows.append(cells)
    if season_names is None:
        raise ValueError(f"{path.name}: could not find Pre/Post season headers")
    s1, s2 = season_names
    wide = pd.DataFrame({
        "gis_no": [int(r[1]) for r in rows],
        "dlr_no": [r[9].upper() for r in rows],
        "district": [r[17] for r in rows],
        "tehsil": [r[16] for r in rows],
        "village": [r[18] for r in rows],
        "lat": [dms_to_decimal(*r[2:5]) for r in rows],
        "lon": [dms_to_decimal(*r[5:8]) for r in rows],
        "elevation_raw": [to_float(r[8]) if r[8] else np.nan for r in rows],
        s1: [r[20] or None for r in rows],
        s2: [r[21] or None for r in rows],
    })
    info = WorkbookInfo(path=path, sheet="pdf", header_row=-1, elevation_header="SEA LEVEL (meter)",
                        season_columns={s1: parse_season_column(s1), s2: parse_season_column(s2)})
    return wide, info


def load_pdf_sources():
    out = []
    for path in sorted(RAW_DIR.glob(PDF_NAME_PATTERN)):
        wide, info = read_pdf(path)
        lah = wide[wide["district"].str.lower() == TARGET_DISTRICT].reset_index(drop=True)
        out.append((lah, info, wide_to_long(lah, info)))
    return out
