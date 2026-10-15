"""Descriptive statistics and exponential-distribution calculations.

Response-time observations are measured in milliseconds (ms), so estimated
rates are measured in inverse milliseconds (1/ms).
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np


@dataclass(frozen=True)
class DescriptiveStatistics:
    """Summary of finite, strictly positive response-time observations.

    Population statistics use denominator ``n``. Sample variance and sample
    standard deviation use denominator ``n - 1`` and are ``None`` when fewer
    than two observations are provided.
    """

    count: int
    mean: float
    median: float
    minimum: float
    maximum: float
    population_variance: float
    population_standard_deviation: float
    sample_variance: float | None
    sample_standard_deviation: float | None


def _positive_observations(observations: object) -> np.ndarray:
    """Convert validated response times to a one-dimensional float array."""
    try:
        source = np.asarray(observations)
    except (TypeError, ValueError) as error:
        raise ValueError("Response times must be a non-empty one-dimensional numeric dataset.") from error

    if source.ndim != 1:
        raise ValueError("Response times must be a one-dimensional dataset.")
    if source.size == 0:
        raise ValueError("Response-time dataset must not be empty.")
    if source.dtype.kind == "b" or (
        source.dtype.kind == "O"
        and any(isinstance(value, (bool, np.bool_)) for value in source)
    ):
        raise ValueError("Response times must be numeric values greater than zero.")

    try:
        values = np.asarray(source, dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Response times must contain only numeric values.") from error

    if not np.isfinite(values).all():
        raise ValueError("Response times must contain only finite values.")
    if (values <= 0).any():
        raise ValueError("Response times must be greater than zero.")
    return values


def _positive_rate(rate: object) -> float:
    """Validate and convert an exponential rate expressed in 1/ms."""
    if isinstance(rate, (bool, np.bool_)) or not isinstance(rate, Real):
        raise ValueError("Rate parameter must be a finite positive number in 1/ms.")
    value = float(rate)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("Rate parameter must be a finite positive number in 1/ms.")
    return value


def _non_negative_times(times: object) -> tuple[np.ndarray, bool]:
    """Validate times and return their array representation and scalar flag."""
    try:
        source = np.asarray(times)
    except (TypeError, ValueError) as error:
        raise ValueError("Time threshold must be a finite non-negative number or array.") from error

    if source.size == 0:
        raise ValueError("Time threshold array must not be empty.")
    if source.dtype.kind == "b" or (
        source.dtype.kind == "O"
        and any(isinstance(value, (bool, np.bool_)) for value in source.flat)
    ):
        raise ValueError("Time thresholds must be finite non-negative numbers.")
    try:
        values = np.asarray(source, dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Time thresholds must contain only numeric values.") from error

    if not np.isfinite(values).all():
        raise ValueError("Time thresholds must contain only finite values.")
    if (values < 0).any():
        raise ValueError("Time thresholds must be non-negative.")
    return values, values.ndim == 0


def _return_scalar_or_array(values: np.ndarray, scalar: bool) -> float | np.ndarray:
    """Return a Python float for scalar inputs and an array otherwise."""
    if scalar:
        return float(values)
    return values


def descriptive_statistics(observations: object) -> DescriptiveStatistics:
    """Calculate population and sample summaries for response times in ms.

    The input is expected to come from the cleaned response-time column. This
    function validates it and raises ``ValueError`` rather than dropping or
    repairing invalid observations.
    """
    values = _positive_observations(observations)
    count = int(values.size)
    scale = float(np.max(values))
    mean = float(np.mean(values / scale) * scale)
    middle = count // 2
    if count % 2:
        median = float(np.partition(values, middle)[middle])
    else:
        lower, upper = np.partition(values, (middle - 1, middle))[[middle - 1, middle]]
        median = float(lower + (upper - lower) / 2)
    with np.errstate(over="ignore", invalid="ignore"):
        population_variance = float(np.var(values, ddof=0))
        sample_variance = float(np.var(values, ddof=1)) if count > 1 else None

    computed = [mean, median, population_variance]
    if sample_variance is not None:
        computed.append(sample_variance)
    if not np.isfinite(computed).all():
        raise ValueError("The dataset's statistics exceed the supported numeric range.")

    return DescriptiveStatistics(
        count=count,
        mean=mean,
        median=median,
        minimum=float(np.min(values)),
        maximum=float(np.max(values)),
        population_variance=population_variance,
        population_standard_deviation=float(np.sqrt(population_variance)),
        sample_variance=sample_variance,
        sample_standard_deviation=(
            float(np.sqrt(sample_variance)) if sample_variance is not None else None
        ),
    )


def estimate_exponential_rate(observations: object) -> float:
    """Estimate lambda as the reciprocal of the observed mean (rate in 1/ms)."""
    mean = descriptive_statistics(observations).mean
    rate = 1.0 / mean
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("Estimated rate is outside the supported numeric range.")
    return rate


def estimated_mean_response_time(rate: Real) -> float:
    """Return the exponential mean ``1 / rate`` in milliseconds."""
    value = _positive_rate(rate)
    mean = 1.0 / value
    if not np.isfinite(mean) or mean <= 0:
        raise ValueError("Estimated mean is outside the supported numeric range.")
    return mean


def exponential_pdf(rate: Real, times: object) -> float | np.ndarray:
    """Evaluate the exponential PDF at time(s) in milliseconds."""
    rate_value = _positive_rate(rate)
    time_values, scalar = _non_negative_times(times)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result = rate_value * np.exp(-rate_value * time_values)
    return _return_scalar_or_array(result, scalar)


def exponential_cdf(rate: Real, times: object) -> float | np.ndarray:
    """Evaluate the exponential CDF at time(s) in milliseconds."""
    rate_value = _positive_rate(rate)
    time_values, scalar = _non_negative_times(times)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result = -np.expm1(-rate_value * time_values)
    return _return_scalar_or_array(result, scalar)


def survival_probability(rate: Real, times: object) -> float | np.ndarray:
    """Evaluate ``P(T > t)`` at time(s) in milliseconds."""
    rate_value = _positive_rate(rate)
    time_values, scalar = _non_negative_times(times)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result = np.exp(-rate_value * time_values)
    return _return_scalar_or_array(result, scalar)
