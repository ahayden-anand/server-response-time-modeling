# Server Response Time Modeling Using Exponential Distribution

This Streamlit application prepares response-time datasets and summarizes them with descriptive statistics and an estimated exponential model. Response-time observations are measured in **milliseconds (ms)** and exponential rates in **inverse milliseconds (1/ms)**.

## Setup and run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Choose **Upload CSV** to preview data and select its response-time column, or **Generate sample** to make reproducible exponential response times with a sample count, rate, and seed. The interface displays total, valid, and removed rows, a validation table, the cleaned data, a cleaned-CSV download, descriptive statistics, and an estimated exponential rate based on the cleaned observations.

## Data rules

Any non-empty CSV may be uploaded. Select the column that contains response times in **milliseconds (ms)**. The app reports missing, non-numeric, `NaN`, infinite, zero, and negative values, then retains only finite values strictly greater than zero. Other columns are preserved for valid rows.

Expected example format (additional columns are allowed):

```csv
timestamp,server,response_time_ms
2026-10-01T09:00:00,api-east,82.4
```

The sample generator's rate is in **inverse milliseconds (1/ms)**. Its exponential scale is `1 / rate_per_ms` ms.

## Data-management Python API

- `load_csv(source)` reads CSV data and raises `CSVDataError` for empty, malformed, unreadable, or header-only files.
- `generate_sample_response_times(count, rate_per_ms, seed)` creates reproducible positive response times. Counts must be positive integers; rates must be finite and positive.
- `validate_and_clean_response_times(frame, response_column)` returns a cleaned DataFrame and a structured `ValidationReport`.

`ValidationReport.to_frame()` produces a table of issue counts and original row indexes.

## Statistical analysis API

The reusable `statistical_analysis.py` module accepts a non-empty, one-dimensional collection of finite, strictly positive numeric response times. It rejects invalid input instead of silently cleaning it; use `validate_and_clean_response_times` first. Its summaries report population variance and standard deviation with denominator `n`, plus sample variance and standard deviation with denominator `n - 1`. Sample statistics are undefined (`None`) for fewer than two observations.

For cleaned observations with arithmetic mean `x̄`, the exponential maximum-likelihood rate estimate is `λ̂ = 1 / x̄` (1/ms), and the estimated mean is `μ̂ = 1 / λ̂` (ms). The module provides:

- `descriptive_statistics(observations)` for count, mean, median, minimum, maximum, and population/sample variances and standard deviations.
- `estimate_exponential_rate(observations)` and `estimated_mean_response_time(rate)` for parameter estimation.
- `exponential_pdf(rate, times)`, `exponential_cdf(rate, times)`, and `survival_probability(rate, times)` for scalar or array-like non-negative time thresholds in ms. These return a float for scalar input and a NumPy array otherwise.

Example for Member 3:

```python
from statistical_analysis import (
    estimate_exponential_rate,
    exponential_pdf,
    exponential_cdf,
    survival_probability,
)

rate_per_ms = estimate_exponential_rate(cleaned_response_times)
pdf_at_100_ms = exponential_pdf(rate_per_ms, 100)
cdf_at_100_ms = exponential_cdf(rate_per_ms, 100)
survival_after_100_ms = survival_probability(rate_per_ms, 100)
```

Pass the cleaned response-time column (in ms), not the sample generator's configured rate, when estimating `λ̂`.

## Tests

```powershell
pytest -q
```

The suite includes both the data-management tests and statistical-analysis tests.
