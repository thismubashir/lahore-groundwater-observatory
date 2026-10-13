import math

import numpy as np
import pytest

from conftest import needs_raw
from lahore_gw.raw_loading import (classify_reading, discover_workbooks, dms_to_decimal,
                                   find_header_row, parse_season_column, read_workbook)


@pytest.mark.parametrize("name,expected", [
    ("Pre-03", (2003, "pre")), ("Po-03", (2003, "post")), ("Po.09", (2009, "post")),
    ("Pr.2010", (2010, "pre")), ("Post-2011 ", (2011, "post")), ("Pre -14", (2014, "pre")),
    ("Premonsoon\n2014", (2014, "pre")), ("Postmonsoon\n2016", (2016, "post")),
    ("Pre- 2019", (2019, "pre")), ("Address", None), ("SEA LEVEL ft", None), (None, None),
])
def test_parse_season_column(name, expected):
    assert parse_season_column(name) == expected


def test_dms_to_decimal():
    assert math.isclose(dms_to_decimal("31", "34", 33), 31 + 34 / 60 + 33 / 3600)
    assert math.isclose(dms_to_decimal("074", "07", "39"), 74 + 7 / 60 + 39 / 3600)
    assert np.isnan(dms_to_decimal(31, 61, 0))      # invalid minutes
    assert np.isnan(dms_to_decimal(None, 1, 1))


@pytest.mark.parametrize("value,status", [
    (None, "blank"), (float("nan"), "blank"), ("  ", "blank"), ("Closed", "closed"),
    ("Closed ", "closed"), ("Dead", "dead"), ("DEAD", "dead"), ("n/a?", "unparsed"),
])
def test_unavailable_readings_are_never_numbers(value, status):
    depth, st = classify_reading(value)
    assert st == status
    assert np.isnan(depth)          # never zero


def test_numeric_readings():
    assert classify_reading(12.5) == (12.5, "reading")
    assert classify_reading("24") == (24.0, "reading")


@needs_raw
def test_2003_2015_header_row_is_1():
    path = next(p for p in discover_workbooks() if "PR-03" in p.name)
    assert find_header_row(path) == 1


@needs_raw
def test_all_workbooks_read_with_valid_lahore_coordinates():
    files = discover_workbooks()
    assert len(files) >= 1
    for p in files:
        wide, info = read_workbook(p)
        assert info.season_columns, p.name
        lah = wide[wide["district"].str.lower() == "lahore"].dropna(subset=["lat"])
        assert lah["lat"].between(31.0, 31.8).all(), p.name
        assert lah["lon"].between(73.9, 74.8).all(), p.name
