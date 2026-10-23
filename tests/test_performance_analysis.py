from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from performance_analysis import (
    SLAComplianceResult,
    calculate_percentiles,
    compare_group_performance,
    evaluate_sla_compliance,
    theoretical_exponential_percentile,
)


# ---------------------------------------------------------------------------
# 1. Theoretical Percentile Tests
# ---------------------------------------------------------------------------


def test_theoretical_exponential_percentile_known_values() -> None:
    rate = 0.02
    scale = 1.0 / rate  # 50 ms

    p50 = theoretical_exponential_percentile(rate, 50.0)
    assert p50 == pytest.approx(scale * np.log(2.0))

    p75 = theoretical_exponential_percentile(rate, 75.0)
    assert p75 == pytest.approx(scale * np.log(4.0))

    p90 = theoretical_exponential_percentile(rate, 90.0)
    assert p90 == pytest.approx(scale * np.log(10.0))

    p95 = theoretical_exponential_percentile(rate, 95.0)
    assert p95 == pytest.approx(scale * np.log(20.0))

    p99 = theoretical_exponential_percentile(rate, 99.0)
    assert p99 == pytest.approx(scale * np.log(100.0))


@pytest.mark.parametrize("percentile", [0.0, 100.0, -5.0, 105.0, np.nan, np.inf, True, "50"])
def test_theoretical_percentile_rejects_invalid_percentiles(percentile: object) -> None:
    with pytest.raises(ValueError):
        theoretical_exponential_percentile(0.02, percentile)  # type: ignore[arg-type]


@pytest.mark.parametrize("rate", [0, -0.01, np.nan, np.inf, True, "slow"])
def test_theoretical_percentile_rejects_invalid_rates(rate: object) -> None:
    with pytest.raises(ValueError):
        theoretical_exponential_percentile(rate, 50.0)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 2. Percentile Calculation & Comparison Table Tests
# ---------------------------------------------------------------------------


def test_calculate_percentiles_known_dataset() -> None:
    # 10 observations: 10, 20, 30, 40, 50, 60, 70, 80, 90, 100
    observations = np.linspace(10, 100, 10)
    rate = 0.02  # theoretical mean = 50 ms

    df = calculate_percentiles(observations, rate)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5
    expected_cols = {
        "Percentile",
        "Percentile_Value",
        "Observed_ms",
        "Theoretical_ms",
        "Difference_ms",
        "Relative_Difference_pct",
    }
    assert expected_cols.issubset(df.columns)
    assert list(df["Percentile"]) == ["P50", "P75", "P90", "P95", "P99"]

    # Check P50 observed
    p50_row = df.loc[df["Percentile"] == "P50"].iloc[0]
    assert p50_row["Observed_ms"] == pytest.approx(55.0)
    assert p50_row["Theoretical_ms"] == pytest.approx(50.0 * np.log(2.0))
    assert p50_row["Difference_ms"] == pytest.approx(55.0 - (50.0 * np.log(2.0)))
    expected_rel = ((55.0 - (50.0 * np.log(2.0))) / (50.0 * np.log(2.0))) * 100.0
    assert p50_row["Relative_Difference_pct"] == pytest.approx(expected_rel)


def test_calculate_percentiles_single_observation() -> None:
    df = calculate_percentiles([42.0], rate=0.01)
    assert len(df) == 5
    # For a single observation, all observed percentiles equal 42.0
    assert (df["Observed_ms"] == 42.0).all()


def test_calculate_percentiles_custom_percentiles() -> None:
    df = calculate_percentiles([10, 20, 30], rate=0.05, percentiles=[25.0, 50.0, 85.5])
    assert len(df) == 3
    assert list(df["Percentile"]) == ["P25", "P50", "P85.5"]


@pytest.mark.parametrize("bad_obs", [[], [0], [-5], [np.nan], ["abc"]])
def test_calculate_percentiles_rejects_invalid_observations(bad_obs: list[object]) -> None:
    with pytest.raises(ValueError):
        calculate_percentiles(bad_obs, rate=0.02)


def test_calculate_percentiles_rejects_empty_percentiles() -> None:
    with pytest.raises(ValueError, match="empty"):
        calculate_percentiles([10, 20], rate=0.02, percentiles=[])


# ---------------------------------------------------------------------------
# 3. SLA Compliance Evaluation Tests
# ---------------------------------------------------------------------------


def test_evaluate_sla_compliance_pass_and_fail() -> None:
    observations = [10.0, 20.0, 30.0, 40.0, 50.0]

    # Target 60% with threshold 30ms -> 3/5 = 60% -> PASS
    res_pass = evaluate_sla_compliance(observations, sla_threshold_ms=30.0, target_compliance_percentage=60.0)
    assert isinstance(res_pass, SLAComplianceResult)
    assert res_pass.total_requests == 5
    assert res_pass.compliant_requests == 3
    assert res_pass.breached_requests == 2
    assert res_pass.actual_compliance_percentage == pytest.approx(60.0)
    assert res_pass.target_compliance_percentage == pytest.approx(60.0)
    assert res_pass.is_compliant is True
    assert res_pass.status == "PASS"

    # Target 80% with threshold 30ms -> 3/5 = 60% -> FAIL
    res_fail = evaluate_sla_compliance(observations, sla_threshold_ms=30.0, target_compliance_percentage=80.0)
    assert res_fail.is_compliant is False
    assert res_fail.status == "FAIL"


def test_evaluate_sla_compliance_all_compliant_or_all_breached() -> None:
    observations = [15.0, 25.0, 35.0]

    # 100% compliant
    res_all_pass = evaluate_sla_compliance(observations, sla_threshold_ms=40.0, target_compliance_percentage=99.0)
    assert res_all_pass.compliant_requests == 3
    assert res_all_pass.breached_requests == 0
    assert res_all_pass.actual_compliance_percentage == 100.0
    assert res_all_pass.is_compliant is True

    # 100% breached
    res_all_fail = evaluate_sla_compliance(observations, sla_threshold_ms=10.0, target_compliance_percentage=50.0)
    assert res_all_fail.compliant_requests == 0
    assert res_all_fail.breached_requests == 3
    assert res_all_fail.actual_compliance_percentage == 0.0
    assert res_all_fail.is_compliant is False


def test_sla_result_to_dict_and_to_frame() -> None:
    res = evaluate_sla_compliance([10.0, 20.0], sla_threshold_ms=15.0, target_compliance_percentage=90.0)
    d = res.to_dict()
    assert d["total_requests"] == 2
    assert d["compliant_requests"] == 1
    assert d["breached_requests"] == 1
    assert d["actual_compliance_percentage"] == 50.0
    assert d["status"] == "FAIL"

    frame = res.to_frame()
    assert isinstance(frame, pd.DataFrame)
    assert len(frame) == 1
    assert "SLA Threshold (ms)" in frame.columns
    assert "Status" in frame.columns


@pytest.mark.parametrize("thresh", [0, -10, np.nan, np.inf, True, "50"])
def test_evaluate_sla_rejects_invalid_thresholds(thresh: object) -> None:
    with pytest.raises(ValueError):
        evaluate_sla_compliance([10, 20], sla_threshold_ms=thresh)  # type: ignore[arg-type]


@pytest.mark.parametrize("target", [0, -10, 105, np.nan, np.inf, True, "95"])
def test_evaluate_sla_rejects_invalid_targets(target: object) -> None:
    with pytest.raises(ValueError):
        evaluate_sla_compliance([10, 20], sla_threshold_ms=25.0, target_compliance_percentage=target)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 4. Group Performance Comparison Tests
# ---------------------------------------------------------------------------


def test_compare_group_performance_basic() -> None:
    df = pd.DataFrame(
        {
            "server": ["api-east", "api-east", "api-west", "api-west", "api-west"],
            "latency_ms": [10.0, 20.0, 100.0, 200.0, 300.0],
        }
    )
    result = compare_group_performance(df, "latency_ms", "server", sla_threshold_ms=150.0)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert "server" in result.columns
    assert "Request_Count" in result.columns
    assert "Mean_ms" in result.columns
    assert "P95_ms" in result.columns
    assert "SLA_Compliance_pct" in result.columns

    east = result.loc[result["server"] == "api-east"].iloc[0]
    assert east["Request_Count"] == 2
    assert east["Mean_ms"] == pytest.approx(15.0)
    assert east["Min_ms"] == pytest.approx(10.0)
    assert east["Max_ms"] == pytest.approx(20.0)
    assert east["SLA_Compliance_pct"] == pytest.approx(100.0)

    west = result.loc[result["server"] == "api-west"].iloc[0]
    assert west["Request_Count"] == 3
    assert west["Mean_ms"] == pytest.approx(200.0)
    assert west["SLA_Compliance_pct"] == pytest.approx((1 / 3) * 100.0)


def test_compare_group_performance_handles_missing_group_labels() -> None:
    df = pd.DataFrame(
        {
            "endpoint": ["/orders", None, "   ", "/health"],
            "response_time_ms": [100.0, 80.0, 70.0, 50.0],
        }
    )
    result = compare_group_performance(df, "response_time_ms", "endpoint")
    groups = set(result["endpoint"])
    assert "(missing)" in groups
    assert "(empty)" in groups
    assert "/orders" in groups
    assert "/health" in groups


def test_compare_group_performance_single_row_per_group() -> None:
    df = pd.DataFrame(
        {
            "server": ["s1", "s2"],
            "rt": [55.0, 95.0],
        }
    )
    result = compare_group_performance(df, "rt", "server")
    assert len(result) == 2
    assert result.loc[result["server"] == "s1", "Mean_ms"].iloc[0] == 55.0
    assert result.loc[result["server"] == "s1", "P95_ms"].iloc[0] == 55.0


def test_compare_group_performance_validates_inputs() -> None:
    df = pd.DataFrame({"rt": [10.0, 20.0], "grp": ["a", "b"]})

    with pytest.raises(TypeError):
        compare_group_performance("not a df", "rt", "grp")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="no rows"):
        compare_group_performance(pd.DataFrame(), "rt", "grp")

    with pytest.raises(ValueError, match="Response-time column"):
        compare_group_performance(df, "bad_col", "grp")

    with pytest.raises(ValueError, match="Grouping column"):
        compare_group_performance(df, "rt", "bad_grp")

    with pytest.raises(ValueError, match="SLA threshold"):
        compare_group_performance(df, "rt", "grp", sla_threshold_ms=-5.0)
