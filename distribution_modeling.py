"""Member 3 distribution modeling and visualization helpers.

Calculates interval probabilities, prepares curve and histogram datasets,
and compares observed response-time distributions against theoretical
exponential models. All response times are in milliseconds (ms) and
rates in inverse milliseconds (1/ms).
"""

from __future__ import annotations

from numbers import Real

import numpy as np
import pandas as pd

from statistical_analysis import (
    estimate_exponential_rate,
    exponential_cdf,
    exponential_pdf,
    survival_probability,
)


def _validate_rate(rate: object) -> float:
    """Validate and convert an exponential rate expressed in 1/ms."""
    if isinstance(rate, (bool, np.bool_)) or not isinstance(rate, Real):
        raise ValueError("Rate parameter must be a finite positive number in 1/ms.")
    value = float(rate)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("Rate parameter must be a finite positive number in 1/ms.")
    return value


def _validate_observations(observations: object) -> np.ndarray:
    """Validate observations as a non-empty 1D array of finite positive numbers."""
    try:
        source = np.asarray(observations)
    except (TypeError, ValueError) as error:
        raise ValueError("Observations must be a non-empty one-dimensional numeric dataset.") from error

    if source.ndim != 1:
        raise ValueError("Observations must be a one-dimensional dataset.")
    if source.size == 0:
        raise ValueError("Observations dataset must not be empty.")
    if source.dtype.kind == "b" or (
        source.dtype.kind == "O"
        and any(isinstance(value, (bool, np.bool_)) for value in source)
    ):
        raise ValueError("Observations must be numeric values greater than zero.")

    try:
        values = np.asarray(source, dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Observations must contain only numeric values.") from error

    if not np.isfinite(values).all():
        raise ValueError("Observations must contain only finite values.")
    if (values <= 0).any():
        raise ValueError("Observations must be greater than zero.")
    return values


def _validate_threshold(threshold: object, name: str = "Time threshold") -> float:
    """Validate a non-negative finite numeric threshold in milliseconds."""
    if isinstance(threshold, (bool, np.bool_)) or not isinstance(threshold, Real):
        raise ValueError(f"{name} must be a finite non-negative number in ms.")
    value = float(threshold)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative number in ms.")
    return value


def _validate_bins(bins: object) -> int:
    """Validate a positive integer bin count."""
    if isinstance(bins, (bool, np.bool_)) or not isinstance(bins, (int, np.integer)):
        raise ValueError("Bin count must be a positive integer.")
    count = int(bins)
    if count <= 0:
        raise ValueError("Bin count must be a positive integer.")
    return count


class DistributionComparisonMetrics(dict):
    """Dictionary holding observed versus theoretical distribution metrics with attribute access."""

    def __init__(self, **kwargs: float) -> None:
        super().__init__(**kwargs)
        self.__dict__.update(kwargs)

    @property
    def observed_mean(self) -> float:
        return self["observed_mean"]

    @property
    def theoretical_mean(self) -> float:
        return self["theoretical_mean"]

    @property
    def observed_median(self) -> float:
        return self["observed_median"]

    @property
    def theoretical_median(self) -> float:
        return self["theoretical_median"]

    @property
    def observed_std(self) -> float:
        return self["observed_std"]

    @property
    def theoretical_std(self) -> float:
        return self["theoretical_std"]

    @property
    def ks_statistic(self) -> float:
        return self["ks_statistic"]

    @property
    def cdf_distance(self) -> float:
        return self["cdf_distance"]

    def to_frame(self, formatted: bool = True) -> pd.DataFrame:
        """Return a DataFrame comparing observed and theoretical metrics.

        When formatted is True (default), numerical values and differences
        are displayed formatted to four decimal places (e.g. 0.000000000000007 -> 0.0000).
        """

        def _fmt(val: Real) -> str:
            rounded = round(float(val), 4)
            if rounded == 0.0:
                return "0.0000"
            return f"{rounded:.4f}"

        mean_diff = self["observed_mean"] - self["theoretical_mean"]
        median_diff = self["observed_median"] - self["theoretical_median"]
        std_diff = self["observed_std"] - self["theoretical_std"]
        ks_val = float(self["ks_statistic"])

        if formatted:
            return pd.DataFrame(
                [
                    {
                        "Metric": "Mean response time (ms)",
                        "Observed": _fmt(self["observed_mean"]),
                        "Theoretical": _fmt(self["theoretical_mean"]),
                        "Difference (Obs - Theo)": _fmt(mean_diff),
                    },
                    {
                        "Metric": "Median response time (ms)",
                        "Observed": _fmt(self["observed_median"]),
                        "Theoretical": _fmt(self["theoretical_median"]),
                        "Difference (Obs - Theo)": _fmt(median_diff),
                    },
                    {
                        "Metric": "Standard deviation (ms)",
                        "Observed": _fmt(self["observed_std"]),
                        "Theoretical": _fmt(self["theoretical_std"]),
                        "Difference (Obs - Theo)": _fmt(std_diff),
                    },
                    {
                        "Metric": "Max CDF distance (KS statistic)",
                        "Observed": _fmt(ks_val),
                        "Theoretical": "0.0000",
                        "Difference (Obs - Theo)": _fmt(ks_val),
                    },
                ]
            )

        return pd.DataFrame(
            [
                {
                    "Metric": "Mean response time (ms)",
                    "Observed": self["observed_mean"],
                    "Theoretical": self["theoretical_mean"],
                    "Difference (Obs - Theo)": mean_diff,
                },
                {
                    "Metric": "Median response time (ms)",
                    "Observed": self["observed_median"],
                    "Theoretical": self["theoretical_median"],
                    "Difference (Obs - Theo)": median_diff,
                },
                {
                    "Metric": "Standard deviation (ms)",
                    "Observed": self["observed_std"],
                    "Theoretical": self["theoretical_std"],
                    "Difference (Obs - Theo)": std_diff,
                },
                {
                    "Metric": "Max CDF distance (KS statistic)",
                    "Observed": ks_val,
                    "Theoretical": 0.0,
                    "Difference (Obs - Theo)": ks_val,
                },
            ]
        )


def interval_probability(rate: Real, lower: Real, upper: Real) -> float:
    """Calculate P(a <= X <= b) for an exponential distribution with rate in 1/ms.

    Both lower and upper bounds must be finite, non-negative numbers in ms
    satisfying lower <= upper.
    """
    valid_rate = _validate_rate(rate)
    valid_lower = _validate_threshold(lower, "Lower bound")
    valid_upper = _validate_threshold(upper, "Upper bound")

    if valid_lower > valid_upper:
        raise ValueError(
            f"Lower bound ({valid_lower}) must be less than or equal to upper bound ({valid_upper})."
        )
    if valid_lower == valid_upper:
        return 0.0

    prob = float(
        survival_probability(valid_rate, valid_lower)
        - survival_probability(valid_rate, valid_upper)
    )
    return float(np.clip(prob, 0.0, 1.0))


def format_interval_label(lower: Real, upper: Real) -> str:
    """Format interval probability label P(a <= X <= b ms) using exact input values."""
    val_lower = _validate_threshold(lower, "Lower bound")
    val_upper = _validate_threshold(upper, "Upper bound")

    if val_lower > val_upper:
        raise ValueError(
            f"Lower bound ({val_lower}) must be less than or equal to upper bound ({val_upper})."
        )

    str_lower = str(int(val_lower)) if val_lower.is_integer() else str(val_lower)
    str_upper = str(int(val_upper)) if val_upper.is_integer() else str(val_upper)
    return f"P({str_lower} ≤ X ≤ {str_upper} ms)"


def generate_distribution_curves(
    rate: Real, max_time: Real, points: int = 200
) -> pd.DataFrame:
    """Generate time, PDF, and CDF curves for an exponential distribution.

    Returns a DataFrame with columns: 'time_ms', 'time', 'pdf', 'cdf'.
    """
    valid_rate = _validate_rate(rate)
    if isinstance(max_time, (bool, np.bool_)) or not isinstance(max_time, Real):
        raise ValueError("max_time must be a finite positive number in ms.")
    t_max = float(max_time)
    if not np.isfinite(t_max) or t_max <= 0:
        raise ValueError("max_time must be a finite positive number in ms.")

    if isinstance(points, (bool, np.bool_)) or not isinstance(points, (int, np.integer)):
        raise ValueError("points must be an integer greater than or equal to 2.")
    pts = int(points)
    if pts < 2:
        raise ValueError("points must be an integer greater than or equal to 2.")

    time_values = np.linspace(0.0, t_max, pts)
    pdf_values = np.asarray(exponential_pdf(valid_rate, time_values), dtype=float)
    cdf_values = np.asarray(exponential_cdf(valid_rate, time_values), dtype=float)

    return pd.DataFrame(
        {
            "time_ms": time_values,
            "time": time_values,
            "pdf": pdf_values,
            "cdf": cdf_values,
        }
    )


def compute_histogram_data(observations: object, bins: int = 20) -> pd.DataFrame:
    """Calculate histogram bins, counts, relative frequencies, and densities.

    Preserves total counts and handles single observations safely.
    """
    values = _validate_observations(observations)
    bin_count = _validate_bins(bins)

    counts, edges = np.histogram(values, bins=bin_count)
    n = len(values)
    bin_widths = np.diff(edges)
    bin_centers = (edges[:-1] + edges[1:]) / 2.0

    with np.errstate(divide="ignore", invalid="ignore"):
        densities = np.where(bin_widths > 0, counts / (n * bin_widths), 0.0)

    relative_frequencies = counts / float(n)
    bin_labels = [
        f"[{edges[i]:.2f}, {edges[i+1]:.2f})"
        if i < len(counts) - 1
        else f"[{edges[i]:.2f}, {edges[i+1]:.2f}]"
        for i in range(len(counts))
    ]

    return pd.DataFrame(
        {
            "bin_start": edges[:-1],
            "bin_end": edges[1:],
            "bin_center": bin_centers,
            "bin_label": bin_labels,
            "count": counts.astype(int),
            "frequency": counts.astype(int),
            "relative_frequency": relative_frequencies,
            "density": densities,
        }
    )


def compare_observed_and_theoretical(
    observations: object, rate: Real, bins: int = 10
) -> pd.DataFrame:
    """Calculate observed versus theoretical counts and probabilities across bins."""
    values = _validate_observations(observations)
    valid_rate = _validate_rate(rate)
    bin_count = _validate_bins(bins)

    counts, edges = np.histogram(values, bins=bin_count)
    n = len(values)

    rows: list[dict[str, object]] = []
    for i in range(len(counts)):
        lower = float(edges[i])
        upper = float(edges[i + 1])
        obs_count = int(counts[i])
        obs_prob = float(obs_count / n)
        theo_prob = interval_probability(valid_rate, lower, upper)
        theo_count = float(n * theo_prob)

        label = (
            f"[{lower:.2f}, {upper:.2f})"
            if i < len(counts) - 1
            else f"[{lower:.2f}, {upper:.2f}]"
        )
        rows.append(
            {
                "bin_lower": lower,
                "bin_upper": upper,
                "bin_range": label,
                "observed_count": obs_count,
                "theoretical_count": theo_count,
                "expected_count": theo_count,
                "observed_probability": obs_prob,
                "theoretical_probability": theo_prob,
                "difference": float(obs_count - theo_count),
            }
        )

    return pd.DataFrame(rows)


def calculate_distribution_comparison_metrics(
    observations: object, rate: Real
) -> DistributionComparisonMetrics:
    """Calculate summary metrics and Kolmogorov-Smirnov CDF distance.

    Safely handles single observations (using population standard deviation when n=1).
    """
    values = _validate_observations(observations)
    valid_rate = _validate_rate(rate)
    n = len(values)

    obs_mean = float(np.mean(values))
    obs_median = float(np.median(values))
    obs_std = float(np.std(values, ddof=1)) if n > 1 else float(np.std(values, ddof=0))

    theo_mean = 1.0 / valid_rate
    theo_median = float(np.log(2.0) / valid_rate)
    theo_std = 1.0 / valid_rate

    # Exact Kolmogorov-Smirnov distance between empirical CDF and theoretical CDF
    sorted_vals = np.sort(values)
    theo_cdf = np.asarray(exponential_cdf(valid_rate, sorted_vals), dtype=float)
    empirical_cdf_upper = np.arange(1, n + 1) / n
    empirical_cdf_lower = np.arange(0, n) / n
    d_plus = float(np.max(empirical_cdf_upper - theo_cdf))
    d_minus = float(np.max(theo_cdf - empirical_cdf_lower))
    ks_distance = float(max(d_plus, d_minus))

    return DistributionComparisonMetrics(
        observed_mean=obs_mean,
        theoretical_mean=theo_mean,
        observed_median=obs_median,
        theoretical_median=theo_median,
        observed_std=obs_std,
        theoretical_std=theo_std,
        ks_statistic=ks_distance,
        cdf_distance=ks_distance,
        mean_difference=obs_mean - theo_mean,
        median_difference=obs_median - theo_median,
        std_difference=obs_std - theo_std,
    )


def compute_ecdf_comparison(observations: object, rate: Real) -> pd.DataFrame:
    """Compute empirical CDF and theoretical CDF values for comparison charting.

    Sorts observations in ascending order, calculates non-decreasing empirical CDF
    probabilities, and evaluates theoretical CDF at corresponding sorted times.
    """
    values = _validate_observations(observations)
    valid_rate = _validate_rate(rate)
    n = len(values)

    sorted_times = np.sort(values)
    empirical_cdf = np.searchsorted(sorted_times, sorted_times, side="right") / float(n)
    theoretical_cdf = np.asarray(exponential_cdf(valid_rate, sorted_times), dtype=float)

    return pd.DataFrame(
        {
            "time_ms": sorted_times,
            "Empirical CDF": empirical_cdf,
            "Theoretical CDF": theoretical_cdf,
            "difference": np.abs(empirical_cdf - theoretical_cdf),
        }
    )
