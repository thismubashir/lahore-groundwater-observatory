# Optional inputs

## `survey_template.csv` (blank on purpose)
A template for *future* community or field surveys of private wells/bores. The pipeline
does **not** read it and runs fully without it. Do not fill it with invented responses.
If you collect real data:
* record only approximate locations (e.g. 3 decimals) and **no names, phone numbers or
  house addresses**;
* record `consent_given` = yes/no and keep consent forms offline;
* keep `measurement_method` honest (e.g. "tape", "owner estimate").

## GRACE (optional, separate)
Place a GRACE/GRACE-FO terrestrial-water-storage CSV here (see
`scripts/optional_grace_comparison.py`). GRACE is not used by the forecasting models.
