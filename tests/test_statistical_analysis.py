import numpy as np
import pytest

from statistical_analysis import (
    descriptive_statistics,
    estimate_exponential_rate,
    estimated_mean_response_time,
    exponential_cdf,
    exponential_pdf,
    survival_probability,
)


def test_descriptive_statistics_for_known_dataset() -> None:
    statistics = descriptive_statistics([1, 2, 3, 4])
    assert statistics.count == 4
    assert statistics.mean == pytest.approx(2.5)
    assert statistics.median == pytest.approx(2.5)
    assert statistics.minimum == pytest.approx(1)
    assert statistics.maximum == pytest.approx(4)


def test_population_and_sample_variance_are_distinguished() -> None:
    statistics = descriptive_statistics([1, 2, 3, 4])
    assert statistics.population_variance == pytest.approx(1.25)
    assert statistics.sample_variance == pytest.approx(5 / 3)


def test_population_and_sample_standard_deviation_are_distinguished() -> None:
    statistics = descriptive_statistics([1, 2, 3, 4])
    assert statistics.population_standard_deviation == pytest.approx(np.sqrt(1.25))
    assert statistics.sample_standard_deviation == pytest.approx(np.sqrt(5 / 3))


def test_single_observation_has_undefined_sample_statistics() -> None:
    statistics = descriptive_statistics([4])
    assert statistics.population_variance == 0
    assert statistics.population_standard_deviation == 0
    assert statistics.sample_variance is None
    assert statistics.sample_standard_deviation is None


def test_exponential_rate_uses_observed_mean() -> None:
    assert estimate_exponential_rate([1, 2, 3, 4]) == pytest.approx(0.4)


def test_estimated_mean_is_reciprocal_of_rate() -> None:
    assert estimated_mean_response_time(0.4) == pytest.approx(2.5)


def test_pdf_supports_scalar_and_array_inputs() -> None:
    assert exponential_pdf(0.5, 0) == pytest.approx(0.5)
    np.testing.assert_allclose(exponential_pdf(0.5, [0, 2]), [0.5, 0.5 * np.exp(-1)])


def test_cdf_supports_scalar_and_array_inputs() -> None:
    assert exponential_cdf(0.5, 0) == pytest.approx(0)
    np.testing.assert_allclose(exponential_cdf(0.5, [0, 2]), [0, 1 - np.exp(-1)])


def test_survival_probability_supports_scalar_and_array_inputs() -> None:
    assert survival_probability(0.5, 0) == pytest.approx(1)
    np.testing.assert_allclose(survival_probability(0.5, [0, 2]), [1, np.exp(-1)])


def test_empty_dataset_is_rejected() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        descriptive_statistics([])


@pytest.mark.parametrize(
    "observations",
    [
        [1, None],
        [1, "not-a-number"],
        [1, np.nan],
        [1, np.inf],
        [1, -np.inf],
        [1, 0],
        [1, -2],
    ],
)
def test_invalid_observations_are_rejected(observations: list[object]) -> None:
    with pytest.raises(ValueError):
        descriptive_statistics(observations)


@pytest.mark.parametrize("rate", [0, -1, np.nan, np.inf, -np.inf])
def test_invalid_rate_parameters_are_rejected(rate: float) -> None:
    with pytest.raises(ValueError, match="Rate parameter"):
        exponential_pdf(rate, 1)


@pytest.mark.parametrize("time", [-1, np.nan, np.inf, -np.inf])
def test_invalid_time_thresholds_are_rejected(time: float) -> None:
    with pytest.raises(ValueError):
        exponential_cdf(0.5, time)


def test_distribution_functions_reject_invalid_array_thresholds() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        survival_probability(0.5, [0, -1])
