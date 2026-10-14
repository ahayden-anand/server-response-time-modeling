# Server Response Time Modeling — Member 1: Data Management

This module prepares response-time datasets before statistical modelling. It does **not** calculate descriptive statistics, fit distributions, calculate probabilities, or render charts.

## Setup and run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Choose **Upload CSV** to preview data and select its response-time column, or **Generate sample** to make reproducible exponential response times with a sample count, rate, and seed. The interface displays total, valid, and removed rows, a validation table, the cleaned data, and a cleaned-CSV download.

## Data rules

Any non-empty CSV may be uploaded. Select the column that contains response times in **milliseconds (ms)**. The app reports missing, non-numeric, `NaN`, infinite, zero, and negative values, then retains only finite values strictly greater than zero. Other columns are preserved for valid rows.

Expected example format (additional columns are allowed):

```csv
timestamp,server,response_time_ms
2026-10-01T09:00:00,api-east,82.4
```

The sample generator's rate is in **inverse milliseconds (1/ms)**. Its exponential scale is `1 / rate_per_ms` ms.

## Python API

- `load_csv(source)` reads CSV data and raises `CSVDataError` for empty, malformed, unreadable, or header-only files.
- `generate_sample_response_times(count, rate_per_ms, seed)` creates reproducible positive response times. Counts must be positive integers; rates must be finite and positive.
- `validate_and_clean_response_times(frame, response_column)` returns a cleaned DataFrame and a structured `ValidationReport`.

`ValidationReport.to_frame()` produces a table of issue counts and original row indexes.

## Test

```powershell
pytest -q
```
