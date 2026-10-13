# Contributing

Thank you for helping improve Lahore Groundwater Observatory. Contributions to parsing, data-quality checks, reproducibility, tests, documentation, accessible charts and forecasting methods are welcome.

## Getting started

1. Fork the repository and clone your fork.
2. From the repository root, create a virtual environment and install the existing dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

The project README documents its tested Python/platform versions. Tests requiring restricted source data may skip when those files are absent. Report passed, skipped and failed counts; a skip is not a verified result.

For the complete local workflow, obtain the original data through the instructions in the project README and put it in the excluded raw-data directory. Never attach it to a public issue or pull request.

```bash
python scripts/run_all.py          # stages 1, 2, 3 and 5 (water-level concern)
python scripts/06_sensitivity.py   # timing/metadata sensitivity
python scripts/verify_project.py   # output checks -> reports/verification.md (local only)
streamlit run app/dashboard.py
```

Without the restricted data, the dashboard starts in its public mode, which shows only aggregate
results and a clearly labelled synthetic example. Force that mode with local data present by
setting `LGO_PUBLIC_MODE=1`, and use it for any screenshot meant for a public page.

## Making a change

Create a descriptive branch, keep changes focused and open a pull request explaining the concrete problem, resulting behavior and verification. Documentation or low-impact presentation edits do not need new tests unless behavior changes. Add meaningful regression checks for parser, calculation or model changes.

Use synthetic fixtures with obviously fictional identifiers. Do not copy private source rows into tests, screenshots, notebooks or issues. Keep source files unchanged.

## Research requirements

- Preserve source provenance, units, unavailable statuses and metadata flags.
- Pair only supported consecutive seasons.
- Fit transformations and tune models using permitted training/validation rows.
- Disclose timing assumptions, sensitivity variants and repeated test-set analyses.
- Report errors in depth units, identical-target comparisons and uncertainty where supported.
- Distinguish predictive association, observed trends and causal claims.
- Label old observations and forecasts with their periods; do not imply current full-city coverage.
- Do not interpret absence of a declining trend as safe drinking water or sufficient supply.

Changing a scientific assumption requires updating the methodology and affected outputs. Never fabricate readings, successful checks or model performance.

## Before submitting

Run the checks relevant to your change and inspect your diff, including notebook outputs. Follow [DATA_POLICY.md](docs/DATA_POLICY.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). Contributions you have authority to submit are accepted under this repository's MIT license; third-party code retains its own notices and terms.
