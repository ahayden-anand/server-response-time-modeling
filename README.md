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

## Distribution modeling API (Member 3)

The `distribution_modeling.py` module provides distribution comparison and interval analysis:

- `interval_probability(rate, lower, upper)` calculates $P(a \le X \le b)$ for an exponential distribution.
- `generate_distribution_curves(rate, max_time, points)` produces points for theoretical PDF and CDF curves.
- `compute_histogram_data(observations, bins)` generates binned histogram frequencies and densities.
- `compare_observed_and_theoretical(observations, rate, bins)` computes observed vs. theoretical counts per bin.
- `calculate_distribution_comparison_metrics(observations, rate)` computes observed vs theoretical means, medians, standard deviations, and the exact two-sided Kolmogorov-Smirnov (KS) statistic.
- `compute_ecdf_comparison(observations, rate)` calculates empirical and theoretical CDF values.

## Performance analysis and SLA API (Member 4)

The `performance_analysis.py` module evaluates server response-time latency percentiles, Service Level Agreement (SLA) threshold compliance, and categorical group comparisons:

- `theoretical_exponential_percentile(rate, percentile)` calculates the exact theoretical exponential percentile $Q(p) = -\ln(1-p)/\lambda$ in ms.
- `calculate_percentiles(observations, rate, percentiles)` computes empirical percentiles (P50, P75, P90, P95, P99 using standard linear interpolation) and compares them against theoretical exponential percentiles with absolute and relative differences.
- `evaluate_sla_compliance(observations, sla_threshold_ms, target_compliance_percentage)` evaluates response times against an SLA threshold $T_{\text{sla}}$, returning total, compliant, and breached counts, compliance rate (%), and a PASS/FAIL status.
- `compare_group_performance(frame, response_column, group_column, sla_threshold_ms)` computes request count, mean, median, P95, min, max, and SLA compliance grouped by columns such as server or endpoint.

## Reporting and export API (Member 4)

The `reporting.py` module generates comprehensive analysis artifacts for download:

- `generate_json_report(...)` produces a complete, JSON-serializable structured report containing validation metrics, descriptive statistics, exponential model parameters, latency percentiles, SLA evaluation, and goodness-of-fit indicators.
- `generate_markdown_report(...)` formats a clean, human-readable executive Markdown report summarizing data quality, tail latency, SLA status, and model fit.
- `generate_summary_csv(...)` creates a consolidated tabular CSV comparing observed and theoretical performance metrics.

In the Streamlit interface, users can download the cleaned dataset (CSV), full analysis report (JSON), executive report (Markdown), and metrics summary (CSV).

## Tests

Run the full automated test suite using:

```powershell
.\.venv\Scripts\python -m pytest -v
```

The suite includes 136 tests across:
- `tests/test_data_management.py`: CSV loading, validation, and sample generation tests.
- `tests/test_statistical_analysis.py`: Descriptive statistics, rate estimation, and distribution function tests.
- `tests/test_distribution_modeling.py`: Interval probabilities, curves, histograms, and KS metrics tests.
- `tests/test_performance_analysis.py`: Percentiles, SLA compliance, and group comparison tests.
- `tests/test_reporting.py`: JSON schema validation, Markdown formatting, and CSV summary tests.
- `tests/test_integration.py`: End-to-end integration workflows with real and generated data.
