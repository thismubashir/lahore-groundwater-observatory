# Publishing checklist

Prepared on 13 October 2026 by Mubashir Ali for the initial release of
<https://github.com/thismubashir/lahore-groundwater-observatory>. This repository was created from a
fresh history containing only reviewed files. A box is ticked only where a check was run and its
output reviewed; the evidence follows each item.

## Identity and license

- [x] Author/copyright holder is correct: Mubashir Ali, in `LICENSE` and `CITATION.cff`.
- [x] Existing code license and third-party notices reviewed: the project had no earlier license; no
  third-party code is vendored (dependencies are installed from `requirements.txt`).
- [x] Root `LICENSE` contains the MIT terms. They cover the code and documentation only, not the
  groundwater data (see [DATA_POLICY.md](DATA_POLICY.md)).
- [x] WAFER is credited as inspiration; no WAFER code is reused.
- [x] `CITATION.cff` has the repository URL. Add release metadata when a release is made.

## Scientific and execution checks

- [x] Source code matches the README (methods, splits, features and outputs re-read).
- [x] Conflicting ground elevations reconciled: where files disagree (36 m and 9 m for two wells), the
  value is left unresolved, kept with its files in the local audit, and not used by the models.
  Earlier runs used the newest file's value; models were retrained after the change.
- [x] Eight inconsistent versus nine flagged coordinate records explained: 8 differ between files, 2 of
  them by under 1 km (not flagged) → 6; plus 3 flagged only by the same-village distance check → 9.
- [x] Test counts from actual output: 53 passed, 0 skipped, 0 failed with the restricted local data;
  in a clone of this repository without data, see the README's Verification section.
- [x] Timing assumptions, metadata sensitivity and split/preprocessing checks are produced by
  `scripts/06_sensitivity.py` and `scripts/verify_project.py` (13 of 13 checks passed locally).
- [x] Forecast reload and representative dashboard flows were checked on the local data; the public
  view was rendered without per-well data.
- [x] Historical observations, next-season model outputs and 2030 trend scenarios use distinct labels
  and chart styles.
- [x] README metrics and verification statements match the current `reports/` outputs.

## Contents

- [x] Code, scripts, app and tests reviewed for embedded data and secrets: no source readings,
  coordinates, addresses, keys or e-mail addresses. Real-data tests compare against the raw files at
  run time and skip when the data is absent.
- [x] Only fictional fixtures and the blank survey template are included. The public dashboard's
  example well is generated and labelled "SYNTHETIC".
- [x] Raw/processed observations, caches, trained models and per-well exports are not in this
  repository.
- [x] Notebook reviewed: outputs cleared; no coordinates or readings in its cells.
- [x] Screenshots: only `docs/screenshots/public/` (public view: aggregates plus the synthetic example),
  viewed and free of well IDs, coordinates and readings.
- [x] `.gitignore` covers the local export paths, so running the pipeline in a clone does not stage
  restricted outputs.
- [x] History: the initial commit has no parent, and the earlier local commits (which contained
  per-well files) are not part of this repository.
- [x] No credentials, local environment or backup archives are included.

## Reuse and contribution

- [x] Setup and source-data access instructions are in the README.
- [x] `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md` and issue/PR templates are included.
- [x] README screenshot links point to `docs/screenshots/public/`, which exists.
- [x] The restricted originals remain only on the maintainer's computer.

## Kept out of the repository (git-ignored if generated locally)

| Path | Why |
|---|---|
| `data/raw/`, `data/processed/`, `data/external/` | source workbooks/PDF, cleaned readings, NASA cache |
| `models/` | trained on the restricted data; excluded from this initial release by choice |
| `reports/audit/` (incl. `data_audit_full.md`) | per-well coordinates, coverage and identifiers |
| `reports/water_level_risk.*`, `reports/verification.md`, `reports/test_error_by_well.csv` | per-well trends, readings, errors |
| per-well figures, `**/forward_forecast.csv`, `**/test_predictions.csv`, `**/forecast_request.csv` | per-well readings and model outputs |
| `notebooks/*.executed.ipynb`, `docs/screenshots/*` (except `public/`) | per-well tables and images |
| `.venv/`, caches, `*.log`, archives, `.env*`, `.streamlit/secrets.toml`, keys | environment and secrets |

## Open items

- [ ] **Data permission.** Ask the Punjab Irrigation Department whether derived per-well values
  (depths, trends, coordinates, model outputs) may be published. Until then only aggregates are public.
- [ ] **Well identifiers in code and tests.** Documentation names no wells, but some scripts, the
  dashboard default and real-data tests refer to wells by ID (form `LHR-<GIS no.>`) without values.
  Confirm that identifiers may be public, or replace them with data-driven choices.
- [ ] **Conduct contact.** `CODE_OF_CONDUCT.md` refers to a maintainer contact; none is listed yet.

## Git review

Before each commit:

```bash
git status --short
git diff --check
git diff --cached --name-status
```

Stage only reviewed paths; do not use `git add -A` or `git add .`.
