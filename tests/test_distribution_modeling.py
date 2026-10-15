from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from distribution_modeling import (
    DistributionComparisonMetrics,
    calculate_distribution_comparison_metrics,
    compare_observed_and_theoretical,
    compute_ecdf_comparison,
    compute_histogram_data,
    format_interval_label,
    generate_distribution_curves,
    interval_probability,
)
from statistical_analysis import (
    estimate_exponential_rate,
    exponential_cdf,
    exponential_pdf,
    survival_probability,
)


# ---------------------------------------------------------------------------
# 1. Probability Calculator & Interval Tests
# ---------------------------------------------------------------------------


def test_interval_probability_matches_theoretical_difference() -> None:
    rate = 0.05
    lower = 10.0
    upper = 30.0

    expected = np.exp(-rate * lower) - np.exp(-rate * upper)
    result = interval_probability(rate, lower, upper)
    assert result == pytest.approx(expected)


def test_interval_probability_edge_cases() -> None:
    rate = 0.02
    # P(a <= X <= a) = 0
    assert interval_probability(rate, 15.0, 15.0) == 0.0

    # P(0 <= X <= b) == CDF(b)
    assert interval_probability(rate, 0.0, 50.0) == pytest.approx(
        float(exponential_cdf(rate, 50.0))
    )

    # For large upper bound, P(a <= X <= b) approaches survival_probability(a)
    assert interval_probability(rate, 20.0, 10000.0) == pytest.approx(
        float(survival_probability(rate, 20.0)), rel=1e-5
    )


@pytest.mark.parametrize(
    "lower, upper",
    [
        (50.0, 20.0),  # lower > upper
        (-5.0, 10.0),  # negative lower
        (10.0, -5.0),  # negative upper
        (np.nan, 20.0),
        (10.0, np.nan),
        (np.inf, 20.0),
        (10.0, np.inf),
        (True, 20.0),
        (10.0, False),
    ],
)
def test_interval_probability_rejects_invalid_bounds(
    lower: float, upper: float
) -> None:
    with pytest.raises(ValueError):
        interval_probability(0.02, lower, upper)


@pytest.mark.parametrize("rate", [0, -0.01, np.nan, np.inf, True, "fast"])
def test_interval_probability_rejects_invalid_rates(rate: object) -> None:
    with pytest.raises(ValueError):
        interval_probability(rate, 10.0, 20.0)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 2. Distribution Curves Tests (PDF & CDF)
# ---------------------------------------------------------------------------


def test_generate_distribution_curves_structure_and_values() -> None:
    rate = 0.04
    max_time = 100.0
    points = 50

    df = generate_distribution_curves(rate, max_time=max_time, points=points)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == points
    assert set(["time_ms", "pdf", "cdf"]).issubset(df.columns)

    # Monotonicity & boundary conditions
    assert df["time_ms"].iloc[0] == pytest.approx(0.0)
    assert df["time_ms"].iloc[-1] == pytest.approx(max_time)
    assert df["pdf"].iloc[0] == pytest.approx(rate)
    assert df["cdf"].iloc[0] == pytest.approx(0.0)
    assert df["cdf"].iloc[-1] == pytest.approx(1.0 - np.exp(-rate * max_time))

    # PDF strictly decreasing, CDF strictly increasing
    assert (np.diff(df["pdf"]) < 0).all()
    assert (np.diff(df["cdf"]) > 0).all()


@pytest.mark.parametrize(
    "rate, max_time, points",
    [
        (0.04, 0.0, 50),
        (0.04, -10.0, 50),
        (0.04, np.nan, 50),
        (0.04, 100.0, 1),
        (0.04, 100.0, 0),
        (0.04, 100.0, True),
        (-0.01, 100.0, 50),
    ],
)
def test_generate_distribution_curves_rejects_invalid_inputs(
    rate: float, max_time: float, points: int
) -> None:
    with pytest.raises(ValueError):
        generate_distribution_curves(rate, max_time=max_time, points=points)


# ---------------------------------------------------------------------------
# 3. Histogram Data Tests
# ---------------------------------------------------------------------------


def test_compute_histogram_data_preserves_counts() -> None:
    observations = [10.0, 25.5, 30.0, 42.0, 55.0, 60.5, 75.0, 90.0]
    bins = 4

    hist = compute_histogram_data(observations, bins=bins)

    assert isinstance(hist, pd.DataFrame)
    assert len(hist) == bins
    assert hist["count"].sum() == len(observations)
    assert hist["relative_frequency"].sum() == pytest.approx(1.0)
    assert (hist["density"] >= 0).all()
    assert (hist["bin_start"] < hist["bin_end"]).all()


def test_compute_histogram_data_handles_single_observation() -> None:
    hist = compute_histogram_data([42.0], bins=5)
    assert hist["count"].sum() == 1
    assert hist["relative_frequency"].sum() == pytest.approx(1.0)


@pytest.mark.parametrize("bins", [0, -1, 3.5, True, "many"])
def test_compute_histogram_data_rejects_invalid_bins(bins: object) -> None:
    with pytest.raises(ValueError):
        compute_histogram_data([10.0, 20.0], bins=bins)  # type: ignore[arg-type]


def test_compute_histogram_data_rejects_empty_and_invalid_data() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        compute_histogram_data([])

    with pytest.raises(ValueError):
        compute_histogram_data([10.0, -5.0])

    with pytest.raises(ValueError):
        compute_histogram_data([10.0, 0.0])


# ---------------------------------------------------------------------------
# 4. Observed vs Theoretical Comparison Table Tests
# ---------------------------------------------------------------------------


def test_compare_observed_and_theoretical_table() -> None:
    observations = [12.0, 24.0, 36.0, 48.0, 60.0]
    rate = 0.03
    bins = 3

    comp = compare_observed_and_theoretical(observations, rate=rate, bins=bins)

    assert isinstance(comp, pd.DataFrame)
    assert len(comp) == bins
    assert comp["observed_count"].sum() == len(observations)
    assert comp["observed_probability"].sum() == pytest.approx(1.0)

    # Theoretical probabilities must be in [0, 1]
    assert (comp["theoretical_probability"] >= 0).all()
    assert (comp["theoretical_probability"] <= 1).all()

    # Expected count must match n * theoretical_probability
    np.testing.assert_allclose(
        comp["expected_count"],
        len(observations) * comp["theoretical_probability"],
    )

    # Difference column check
    np.testing.assert_allclose(
        comp["difference"],
        comp["observed_count"] - comp["expected_count"],
    )


def test_compare_observed_and_theoretical_single_observation() -> None:
    comp = compare_observed_and_theoretical([25.0], rate=0.04, bins=2)
    assert len(comp) == 2
    assert comp["observed_count"].sum() == 1


# ---------------------------------------------------------------------------
# 5. Distribution Comparison Metrics Tests
# ---------------------------------------------------------------------------


def test_calculate_distribution_comparison_metrics_known_sample() -> None:
    observations = [10.0, 20.0, 30.0, 40.0]
    rate = 0.04

    metrics = calculate_distribution_comparison_metrics(observations, rate)

    assert isinstance(metrics, DistributionComparisonMetrics)
    assert metrics.observed_mean == pytest.approx(25.0)
    assert metrics.theoretical_mean == pytest.approx(1.0 / rate)  # 25.0
    assert metrics.observed_median == pytest.approx(25.0)
    assert metrics.theoretical_median == pytest.approx(np.log(2.0) / rate)
    assert metrics.theoretical_std == pytest.approx(1.0 / rate)

    # Sample standard deviation (ddof=1)
    assert metrics.observed_std == pytest.approx(float(np.std(observations, ddof=1)))

    # KS statistic is bounded in [0, 1]
    assert 0.0 <= metrics.ks_statistic <= 1.0
    assert metrics.ks_statistic == metrics.cdf_distance

    # Dict access & DataFrame conversion
    assert metrics["observed_mean"] == pytest.approx(25.0)
    frame = metrics.to_frame()
    assert isinstance(frame, pd.DataFrame)
    assert "Metric" in frame.columns
    assert len(frame) == 4


def test_calculate_distribution_comparison_metrics_single_observation() -> None:
    # Single observation should not fail on sample standard deviation calculation
    metrics = calculate_distribution_comparison_metrics([30.0], rate=0.05)
    assert metrics.observed_mean == pytest.approx(30.0)
    assert metrics.observed_median == pytest.approx(30.0)
    assert metrics.observed_std == pytest.approx(0.0)
    assert 0.0 <= metrics.ks_statistic <= 1.0


# ---------------------------------------------------------------------------
# 6. Empirical CDF Comparison Tests
# ---------------------------------------------------------------------------


def test_compute_ecdf_comparison() -> None:
    observations = [10.0, 50.0, 20.0, 40.0, 30.0]
    rate = 0.02

    ecdf_df = compute_ecdf_comparison(observations, rate)

    assert isinstance(ecdf_df, pd.DataFrame)
    assert len(ecdf_df) == 5
    # Should be sorted
    assert (np.diff(ecdf_df["time_ms"]) >= 0).all()
    # Empirical CDF steps from 1/n to 1.0
    np.testing.assert_allclose(
        ecdf_df["Empirical CDF"], [0.2, 0.4, 0.6, 0.8, 1.0]
    )
    # Theoretical CDF must be strictly increasing
    assert (np.diff(ecdf_df["Theoretical CDF"]) > 0).all()
    # Differences must be non-negative
    assert (ecdf_df["difference"] >= 0).all()


# ---------------------------------------------------------------------------
# 7. Focused Regression Tests for Interval Label & Sorted CDF Data
# ---------------------------------------------------------------------------


def test_format_interval_label_exact_bounds() -> None:
    # Exact integer bounds formatting
    assert format_interval_label(10, 20) == "P(10 ≤ X ≤ 20 ms)"
    assert format_interval_label(10.0, 20.0) == "P(10 ≤ X ≤ 20 ms)"

    # Exact decimal bounds formatting
    assert format_interval_label(10.5, 25.75) == "P(10.5 ≤ X ≤ 25.75 ms)"
    assert format_interval_label(59.6, 129.3) == "P(59.6 ≤ X ≤ 129.3 ms)"

    # Equal bounds
    assert format_interval_label(15.0, 15.0) == "P(15 ≤ X ≤ 15 ms)"


@pytest.mark.parametrize(
    "lower, upper",
    [
        (25.0, 10.0),  # lower > upper
        (-1.0, 10.0),  # negative lower
        (10.0, -1.0),  # negative upper
        (np.nan, 20.0),
        (10.0, np.inf),
    ],
)
def test_format_interval_label_rejects_invalid_bounds(
    lower: float, upper: float
) -> None:
    with pytest.raises(ValueError):
        format_interval_label(lower, upper)


def test_ecdf_comparison_sorted_and_non_decreasing_with_duplicates() -> None:
    # Unsorted input observations with duplicate response times
    raw_observations = [60.0, 20.0, 40.0, 20.0, 50.0]
    rate = 0.025

    df = compute_ecdf_comparison(raw_observations, rate)

    # 1. x-axis increases from smallest (20.0) to largest (60.0)
    assert df["time_ms"].iloc[0] == pytest.approx(20.0)
    assert df["time_ms"].iloc[-1] == pytest.approx(60.0)
    assert (np.diff(df["time_ms"]) >= 0).all()

    # 2. Empirical CDF is non-decreasing across all points
    assert (np.diff(df["Empirical CDF"]) >= 0).all()
    assert (df["Empirical CDF"] <= 1.0).all()
    assert (df["Empirical CDF"] >= 0.0).all()

    # 3. Theoretical CDF is evaluated at the corresponding sorted time values
    expected_theoretical = exponential_cdf(rate, df["time_ms"])
    np.testing.assert_allclose(df["Theoretical CDF"], expected_theoretical)

    # 4. Tie handling: for two 20.0 ms observations in a 5-element sample, ecdf is 2/5 = 0.4
    assert df["Empirical CDF"].iloc[0] == pytest.approx(0.4)
    assert df["Empirical CDF"].iloc[1] == pytest.approx(0.4)


def test_distribution_comparison_metrics_table_four_decimal_places() -> None:
    # Verify values and differences display to four decimal places
    metrics = DistributionComparisonMetrics(
        observed_mean=25.0,
        theoretical_mean=25.0 + 7e-15,  # 0.000000000000007 difference
        observed_median=17.32867,
        theoretical_median=17.32861,
        observed_std=12.90994,
        theoretical_std=25.0,
        ks_statistic=0.123456,
        cdf_distance=0.123456,
        mean_difference=-7e-15,
        median_difference=0.00006,
        std_difference=-12.09006,
    )
    frame = metrics.to_frame()
    # Check that 0.000000000000007 displays as 0.0000
    assert frame["Difference (Obs - Theo)"].iloc[0] == "0.0000"
    assert frame["Observed"].iloc[0] == "25.0000"
    assert frame["Theoretical"].iloc[0] == "25.0000"
    # Check four decimal places on other rows
    assert frame["Observed"].iloc[1] == "17.3287"
    assert frame["Theoretical"].iloc[1] == "17.3286"
    assert frame["Observed"].iloc[3] == "0.1235"
    assert frame["Theoretical"].iloc[3] == "0.0000"
    assert frame["Difference (Obs - Theo)"].iloc[3] == "0.1235"

    # Underlying metrics data is unrounded
    assert metrics.observed_mean == 25.0
    assert metrics.theoretical_mean == 25.0 + 7e-15
