"""Combine the workbooks into one clean, long-format Lahore dataset + audit.

Rules (see README "Data cleaning"):
* keep District == Lahore only (not the whole Lahore Irrigation Zone);
* a monitoring location is identified by its GIS number, and we *verify*
  that every GIS number maps to exactly one DLR number in every file;
* blank / Closed / Dead => no reading (NaN with a status), never zero;
* overlapping records (same well, year, season in two files) are compared;
  disagreements are written to an audit table;
* depth to water (ft below ground) is kept separate from ground elevation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import FT_TO_M, PROCESSED_DIR, REPORTS_DIR, TARGET_DISTRICT
from .raw_loading import discover_workbooks, read_workbook, wide_to_long

STATUS_UNAVAILABLE = {"closed", "dead", "blank", "unparsed"}
TOLERANCE_FT = 0.005
ELEV_TOLERANCE_M = 1.0          # files differing by more than this leave elevation unresolved


@dataclass
class CleanResult:
    wells: pd.DataFrame
    long: pd.DataFrame
    conflicts: pd.DataFrame
    coord_discrepancies: pd.DataFrame
    identifier_checks: pd.DataFrame
    excluded_rows: pd.DataFrame
    notes: list = field(default_factory=list)


def season_index(year: int, season: str) -> int:
    return int(year) * 2 + (0 if season == "pre" else 1)


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def _norm_village(v: str) -> str:
    v = re.sub(r"\(.*?\)", "", str(v).lower())
    return re.sub(r"[^a-z]", "", v)


def load_all_sources(include_pdf: bool = True):
    """Return (list of (wide_lahore, info, long_lahore), excluded rows)."""
    sources, excluded = [], []
    for path in discover_workbooks():
        wide, info = read_workbook(path)
        district = wide["district"].str.lower()
        is_lahore = district == TARGET_DISTRICT
        # Rows with a missing district but Lahore-looking tehsil are reported,
        # not silently included.
        odd = (~is_lahore) & (district.isin(["nan", "none", ""])) & wide["gis_no"].notna()
        if odd.any():
            ex = wide.loc[odd, ["gis_no", "dlr_no", "tehsil", "village"]].copy()
            ex["source_file"] = path.name
            ex["reason"] = "District cell empty"
            excluded.append(ex)
        lah = wide[is_lahore].reset_index(drop=True)
        sources.append((lah, info, wide_to_long(lah, info)))
    if include_pdf:
        from .pdf_loading import load_pdf_sources
        sources.extend(load_pdf_sources())
    excluded = pd.concat(excluded, ignore_index=True) if excluded else pd.DataFrame()
    return sources, excluded


def _source_order(sources) -> dict:
    """Rank files by the latest season they contain (later = more recent)."""
    order = {}
    for wide, info, long in sources:
        latest = max(season_index(y, s) for y, s in info.season_columns.values())
        order[info.path.name] = latest
    return order


def verify_identifiers(sources) -> pd.DataFrame:
    meta = pd.concat([w.assign(source_file=i.path.name) for w, i, _ in sources], ignore_index=True)
    checks = []
    for gis, g in meta.groupby("gis_no"):
        checks.append({
            "gis_no": gis,
            "dlr_values": ";".join(sorted(g["dlr_no"].unique())),
            "n_files": g["source_file"].nunique(),
            "village_variants": ";".join(sorted(g["village"].unique())),
            "tehsil_variants": ";".join(sorted(g["tehsil"].unique())),
            "gis_dlr_consistent": g["dlr_no"].nunique() == 1,
            "village_consistent": g["village"].map(_norm_village).nunique() == 1,
        })
    checks = pd.DataFrame(checks)
    # The reverse direction: one DLR number must not belong to two GIS numbers.
    dlr_to_gis = meta.groupby("dlr_no")["gis_no"].nunique()
    bad_dlr = dlr_to_gis[dlr_to_gis > 1]
    if (~checks["gis_dlr_consistent"]).any() or len(bad_dlr):
        raise ValueError(
            "Identifier verification failed: GIS<->DLR mapping is not one-to-one.\n"
            f"{checks[~checks['gis_dlr_consistent']]}\n{bad_dlr}")
    return checks


def _elevation(g: pd.DataFrame) -> dict:
    """Ground elevation with provenance; files that disagree leave it UNRESOLVED.

    A newer file is not assumed to be right: when files give different values
    (beyond ELEV_TOLERANCE_M) the elevation used for modelling is missing and
    every value is kept with the files that report it.
    """
    e = g.dropna(subset=["elevation_raw"])
    by_value = (e.sort_values("rank").groupby("elevation_raw", sort=False)["source_file"]
                .apply(lambda s: ", ".join(dict.fromkeys(s))))
    provenance = "; ".join(f"{v:g} m: {files}" for v, files in by_value.items())
    if e.empty:
        return {"ground_elevation_m": np.nan, "elevation_status": "missing in all files",
                "elevation_latest_file_m": np.nan, "elevation_provenance": ""}
    latest = float(e.sort_values("rank", ascending=False)["elevation_raw"].iloc[0])
    spread = e["elevation_raw"].max() - e["elevation_raw"].min()
    if spread > ELEV_TOLERANCE_M:
        status = f"unresolved: files disagree by {spread:g} m"
        value = np.nan
    else:
        status, value = "consistent", latest
    return {"ground_elevation_m": value, "elevation_status": status,
            "elevation_latest_file_m": latest, "elevation_provenance": provenance}


def build_wells(sources, order) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per monitoring location with canonical metadata.

    Coordinates come from the most recent file that has them and are flagged
    when files disagree (they are not averaged). Elevation is left unresolved
    when files disagree (see _elevation).
    """
    meta = pd.concat([w.assign(source_file=i.path.name, rank=order[i.path.name])
                      for w, i, _ in sources], ignore_index=True)
    wells, disc = [], []
    for gis, g in meta.groupby("gis_no"):
        g = g.sort_values("rank", ascending=False)
        with_xy = g.dropna(subset=["lat", "lon"])
        best = with_xy.iloc[0] if len(with_xy) else g.iloc[0]
        elev = g["elevation_raw"].dropna()
        for _, r in with_xy.iterrows():
            d = haversine_km(best["lat"], best["lon"], r["lat"], r["lon"])
            if d > 0.05:
                disc.append({"gis_no": gis, "village": best["village"],
                             "canonical_file": best["source_file"], "other_file": r["source_file"],
                             "canonical_lat": round(best["lat"], 5), "canonical_lon": round(best["lon"], 5),
                             "other_lat": round(r["lat"], 5), "other_lon": round(r["lon"], 5),
                             "distance_km": round(float(d), 2)})
        wells.append({
            "well_id": f"LHR-{gis:03d}",
            "gis_no": gis,
            "dlr_no": g["dlr_no"].iloc[0],
            "tehsil": best["tehsil"],
            "village": g.iloc[0]["village"],
            "lat": best["lat"], "lon": best["lon"],
            "coord_source": best["source_file"] if len(with_xy) else None,
            **_elevation(g),
            "elevation_values_seen": ";".join(str(int(v)) if float(v).is_integer() else str(v)
                                             for v in sorted(elev.unique())),
            "n_source_files": g["source_file"].nunique(),
        })
    wells = pd.DataFrame(wells)
    disc = pd.DataFrame(disc)
    if len(disc):
        max_d = disc.groupby("gis_no")["distance_km"].max()
        wells["max_coord_discrepancy_km"] = wells["gis_no"].map(max_d).fillna(0.0)
    else:
        wells["max_coord_discrepancy_km"] = 0.0
    wells["coord_quality"] = [_coord_quality(r, wells) for _, r in wells.iterrows()]
    return wells, disc


def _coord_quality(row, wells) -> str:
    """Plain-language flag; coordinates are never silently edited."""
    flags = []
    if pd.isna(row["lat"]):
        return "missing"
    if row["max_coord_discrepancy_km"] > 1.0:
        flags.append(f"files disagree by up to {row['max_coord_discrepancy_km']:.1f} km (latest file used)")
    same = wells[(wells["village"].map(_norm_village) == _norm_village(row["village"]))
                 & (wells["gis_no"] != row["gis_no"])].dropna(subset=["lat"])
    for _, o in same.iterrows():
        d = haversine_km(row["lat"], row["lon"], o["lat"], o["lon"])
        if d > 5:
            flags.append(f"{d:.1f} km from same-village well GIS {o['gis_no']}")
    if row["n_source_files"] == 1 or str(row["coord_source"]) == "nan":
        flags.append("single source")
    return "; ".join(flags) if flags else "consistent"



def resolve_records(sources, order, wells) -> tuple[pd.DataFrame, pd.DataFrame]:
    long = pd.concat([l for _, _, l in sources], ignore_index=True)
    long["rank"] = long["source_file"].map(order)
    covered_seasons = sorted({(int(y), s) for y, s in zip(long["year"], long["season"])},
                             key=lambda t: season_index(*t))
    records, conflicts = [], []
    for (gis, year, season), g in long.groupby(["gis_no", "year", "season"]):
        g = g.sort_values("rank", ascending=False)
        nums = g[g["status"] == "reading"]
        distinct = []
        for v in nums["depth_ft"]:
            if not any(abs(v - d) <= TOLERANCE_FT for d in distinct):
                distinct.append(v)
        rec = {"gis_no": gis, "year": int(year), "season": season,
               "sources": ";".join(g["source_file"]), "n_sources": len(g),
               "raw_values": " | ".join(f"{f}: {v}" for f, v in zip(g["source_file"], g["raw_value"])),
               "conflict": False, "conflict_type": ""}
        if len(distinct) == 0:
            rec.update(depth_ft=np.nan, status=g.iloc[0]["status"])
        else:
            chosen = nums.iloc[0]   # most recent file wins, but we flag it
            rec.update(depth_ft=float(chosen["depth_ft"]), status="reading",
                       chosen_from=chosen["source_file"])
        if len(g) > 1:
            if len(distinct) > 1:
                rec.update(conflict=True, conflict_type="different numeric readings")
            elif len(distinct) == 1 and (g["status"] != "reading").any():
                rec.update(conflict=True, conflict_type="reading vs unavailable status")
            elif len(distinct) == 0 and g["status"].nunique() > 1:
                rec.update(conflict=True, conflict_type="different unavailable statuses")
        if rec["conflict"]:
            conflicts.append({**rec, "spread_ft": (max(distinct) - min(distinct)) if distinct else np.nan})
        records.append(rec)
    rec = pd.DataFrame(records)
    conflict_cols = ["gis_no", "year", "season", "conflict_type", "spread_ft", "sources",
                     "raw_values", "depth_ft", "chosen_from", "status"]
    conflicts = pd.DataFrame(conflicts).reindex(columns=conflict_cols)

    # Complete grid: every well x every season covered by at least one file.
    grid = pd.MultiIndex.from_product(
        [wells["gis_no"], [c for c in covered_seasons]], names=["gis_no", "ys"]).to_frame(index=False)
    grid["year"] = grid["ys"].map(lambda t: t[0])
    grid["season"] = grid["ys"].map(lambda t: t[1])
    grid = grid.drop(columns="ys")
    out = grid.merge(rec, on=["gis_no", "year", "season"], how="left")
    out["status"] = out["status"].fillna("not_listed")   # well absent from that file
    out["n_sources"] = out["n_sources"].fillna(0).astype(int)
    out["conflict"] = out["conflict"].fillna(False).astype(bool)
    out = out.merge(wells[["gis_no", "well_id"]], on="gis_no")
    out["season_index"] = [season_index(y, s) for y, s in zip(out["year"], out["season"])]
    out["depth_m"] = out["depth_ft"] * FT_TO_M
    out["assumed_period_end"] = [f"{y}-05-31" if s == "pre" else f"{y}-10-31"
                                 for y, s in zip(out["year"], out["season"])]
    cols = ["well_id", "gis_no", "year", "season", "season_index", "assumed_period_end",
            "depth_ft", "depth_m", "status", "n_sources", "sources", "conflict",
            "conflict_type", "chosen_from", "raw_values"]
    out = out[cols].sort_values(["well_id", "season_index"]).reset_index(drop=True)
    return out, conflicts


def elevation_unit_note(sources) -> str:
    """Check whether the 'SEA LEVEL ft' column is really in feet."""
    ft = [(w, i) for w, i, _ in sources if "ft" in i.elevation_header.lower()]
    m = [(w, i) for w, i, _ in sources if "meter" in i.elevation_header.lower()]
    if not ft or not m:
        return "Elevation units: single labelling, no cross-check needed."
    a = ft[0][0][["gis_no", "elevation_raw"]].dropna()
    b = pd.concat([w[["gis_no", "elevation_raw"]] for w, _ in m]).dropna().drop_duplicates("gis_no")
    j = a.merge(b, on="gis_no", suffixes=("_ft_label", "_m_label"))
    same = int((j["elevation_raw_ft_label"] == j["elevation_raw_m_label"]).sum())
    return (f"Elevation units: '{ft[0][1].elevation_header}' in {ft[0][1].path.name} has the same value as the "
            f"'meter' columns for {same} of {len(j)} shared wells (range {a.elevation_raw.min():.0f}-"
            f"{a.elevation_raw.max():.0f}). Lahore's ground surface is roughly 200-215 m above sea level, "
            "and ~200 ft (61 m) would be impossible, so all elevation values are treated as metres. "
            "Elevation (ground surface above sea level) is kept separate from depth to water (below ground).")


def clean(include_pdf: bool = True) -> CleanResult:
    sources, excluded = load_all_sources(include_pdf=include_pdf)
    order = _source_order(sources)
    checks = verify_identifiers(sources)
    wells, disc = build_wells(sources, order)
    long, conflicts = resolve_records(sources, order, wells)
    overlaps = long[long["n_sources"] > 1]
    notes = [f"Overlapping well-season records (present in >1 file): {len(overlaps)}; "
             f"with any disagreement: {len(conflicts)}; "
             f"with different numeric readings: {int((conflicts['conflict_type'] == 'different numeric readings').sum())}"]
    elev_headers = {i.path.name: i.elevation_header for _, i, _ in sources if i.elevation_header}
    notes.append(f"Elevation headers seen: {elev_headers}")
    notes.append(elevation_unit_note(sources))
    multi = wells[wells["elevation_status"].str.startswith("unresolved")]
    if len(multi):
        notes.append("Ground elevation differs between files for: " + ", ".join(
            f"{r.well_id} ({r.elevation_provenance}; left unresolved, not used by the models)"
            for r in multi.itertuples()))
    return CleanResult(wells, long, conflicts, disc, checks, excluded, notes)


def coverage_tables(res: CleanResult):
    L = res.long
    by_season = (L.assign(available=L["status"] == "reading")
                   .groupby(["year", "season", "season_index"])
                   .agg(wells_listed=("status", lambda s: (s != "not_listed").sum()),
                        readings=("available", "sum"),
                        closed=("status", lambda s: (s == "closed").sum()),
                        dead=("status", lambda s: (s == "dead").sum()),
                        blank=("status", lambda s: (s == "blank").sum()),
                        not_listed=("status", lambda s: (s == "not_listed").sum()))
                   .reset_index().sort_values("season_index"))
    rows = []
    for wid, g in L.groupby("well_id"):
        g = g.sort_values("season_index")
        ok = g["status"].eq("reading").to_numpy()
        si = g["season_index"].to_numpy()
        pairs = int(sum(ok[i] and ok[i + 1] and si[i + 1] - si[i] == 1 for i in range(len(g) - 1)))
        avail = g[ok]
        rows.append({"well_id": wid, "readings": int(ok.sum()), "seasons_in_grid": len(g),
                     "closed": int((g.status == "closed").sum()), "dead": int((g.status == "dead").sum()),
                     "blank": int((g.status == "blank").sum()),
                     "not_listed": int((g.status == "not_listed").sum()),
                     "first_reading": f"{avail.iloc[0].season}-{avail.iloc[0].year}" if len(avail) else "",
                     "last_reading": f"{avail.iloc[-1].season}-{avail.iloc[-1].year}" if len(avail) else "",
                     "consecutive_pairs": pairs})
    by_well = pd.DataFrame(rows)
    by_well["usable_for_modeling"] = (by_well["readings"] >= 4) & (by_well["consecutive_pairs"] >= 2)
    by_well = res.wells[["well_id", "tehsil", "village", "max_coord_discrepancy_km"]].merge(by_well, on="well_id")
    return by_season, by_well


def save(res: CleanResult):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "audit").mkdir(parents=True, exist_ok=True)
    res.wells.to_csv(PROCESSED_DIR / "lahore_wells.csv", index=False)
    res.long.to_csv(PROCESSED_DIR / "lahore_groundwater_long.csv", index=False)
    by_season, by_well = coverage_tables(res)
    a = REPORTS_DIR / "audit"
    by_season.to_csv(a / "coverage_by_season.csv", index=False)
    by_well.to_csv(a / "coverage_by_well.csv", index=False)
    res.conflicts.to_csv(a / "overlap_conflicts.csv", index=False)
    res.coord_discrepancies.to_csv(a / "coordinate_discrepancies.csv", index=False)
    res.identifier_checks.to_csv(a / "identifier_checks.csv", index=False)
    res.excluded_rows.to_csv(a / "excluded_rows.csv", index=False)
    return by_season, by_well
