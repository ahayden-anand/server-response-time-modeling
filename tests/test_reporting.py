from __future__ import annotations

import io
import json

import pandas as pd
import pytest

from data_management import ValidationReport
from distribution_modeling import DistributionComparisonMetrics
from performance_analysis import SLAComplianceResult, calculate_percentiles
from reporting import (
    generate_json_report,
    generate_markdown_report,
    generate_summary_csv,
)
from statistical_analysis import DescriptiveStatistics


@pytest.fixture
def sample_analysis_objects() -> dict[str, object]:
    val_report = ValidationReport(
        response_column="latency_ms",
        total_rows=10,
        valid_rows=8,
        issue_rows={"missing": [0], "negative": [1]},
    )
    desc_stats = DescriptiveStatistics(
        count=8,
        mean=50.0,
        median=45.0,
        minimum=10.0,
        maximum=120.0,
        population_variance=600.0,
        population_standard_deviation=24.494897,
        sample_variance=685.714285,
        sample_standard_deviation=26.186146,
    )
    rate = 0.02
    mean = 50.0
    percentiles_df = calculate_percentiles(
        [10.0, 20.0, 30.0, 45.0, 55.0, 70.0, 90.0, 120.0],
        rate=rate,
    )
    sla_res = SLAComplianceResult(
        total_requests=8,
        compliant_requests=7,
        breached_requests=1,
        actual_compliance_percentage=87.5,
        target_compliance_percentage=95.0,
        sla_threshold_ms=100.0,
        is_compliant=False,
        status="FAIL",
    )
    comp_metrics = DistributionComparisonMetrics(
        observed_mean=50.0,
        theoretical_mean=50.0,
        observed_median=45.0,
        theoretical_median=34.657359,
        observed_std=26.186146,
        theoretical_std=50.0,
        ks_statistic=0.125,
        cdf_distance=0.125,
        mean_difference=0.0,
        median_difference=10.342641,
        std_difference=-23.813854,
    )
    group_df = pd.DataFrame(
        [
            {"server": "api-east", "Request_Count": 4, "Mean_ms": 40.0, "P95_ms": 70.0, "SLA_Compliance_pct": 100.0},
            {"server": "api-west", "Request_Count": 4, "Mean_ms": 60.0, "P95_ms": 120.0, "SLA_Compliance_pct": 75.0},
        ]
    )

    return {
        "val_report": val_report,
        "desc_stats": desc_stats,
        "rate": rate,
        "mean": mean,
        "percentiles_df": percentiles_df,
        "sla_res": sla_res,
        "comp_metrics": comp_metrics,
        "group_df": group_df,
    }


# ---------------------------------------------------------------------------
# 1. JSON Report Tests
# ---------------------------------------------------------------------------


def test_generate_json_report_full(sample_analysis_objects: dict[str, object]) -> None:
    json_str = generate_json_report(
        source_kind="Upload CSV",
        response_column="latency_ms",
        validation_report=sample_analysis_objects["val_report"],  # type: ignore[arg-type]
        descriptive_stats=sample_analysis_objects["desc_stats"],  # type: ignore[arg-type]
        estimated_rate=sample_analysis_objects["rate"],  # type: ignore[arg-type]
        estimated_mean=sample_analysis_objects["mean"],  # type: ignore[arg-type]
        percentiles_df=sample_analysis_objects["percentiles_df"],  # type: ignore[arg-type]
        sla_result=sample_analysis_objects["sla_res"],  # type: ignore[arg-type]
        comparison_metrics=sample_analysis_objects["comp_metrics"],  # type: ignore[arg-type]
        group_comparison_df=sample_analysis_objects["group_df"],  # type: ignore[arg-type]
        group_column="server",
    )

    data = json.loads(json_str)
    assert isinstance(data, dict)
    assert data["report_metadata"]["data_source"] == "Upload CSV"
    assert data["report_metadata"]["response_time_column"] == "latency_ms"

    assert data["data_validation"]["total_rows"] == 10
    assert data["data_validation"]["valid_rows"] == 8
    assert data["data_validation"]["excluded_rows"] == 2

    assert data["descriptive_statistics"]["count"] == 8
    assert data["descriptive_statistics"]["mean_ms"] == 50.0

    assert data["exponential_model"]["estimated_rate_lambda_per_ms"] == 0.02

    assert len(data["percentile_analysis"]) == 5
    assert data["sla_evaluation"]["status"] == "FAIL"
    assert data["goodness_of_fit"]["ks_statistic"] == 0.125
    assert len(data["group_comparison"]["groups"]) == 2


def test_generate_json_report_empty_or_none_fields() -> None:
    # When optional fields are omitted, the report generates valid JSON with nulls
    json_str = generate_json_report(
        source_kind="Generate sample",
        response_column="response_time_ms",
    )
    data = json.loads(json_str)
    assert data["report_metadata"]["data_source"] == "Generate sample"
    assert data["data_validation"] is None
    assert data["descriptive_statistics"] is None
    assert data["exponential_model"] is None
    assert data["percentile_analysis"] is None
    assert data["sla_evaluation"] is None
    assert data["goodness_of_fit"] is None
    assert data["group_comparison"] is None


# ---------------------------------------------------------------------------
# 2. Markdown Report Tests
# ---------------------------------------------------------------------------


def test_generate_markdown_report_full(sample_analysis_objects: dict[str, object]) -> None:
    md_str = generate_markdown_report(
        source_kind="Upload CSV",
        response_column="latency_ms",
        validation_report=sample_analysis_objects["val_report"],  # type: ignore[arg-type]
        descriptive_stats=sample_analysis_objects["desc_stats"],  # type: ignore[arg-type]
        estimated_rate=sample_analysis_objects["rate"],  # type: ignore[arg-type]
        estimated_mean=sample_analysis_objects["mean"],  # type: ignore[arg-type]
        percentiles_df=sample_analysis_objects["percentiles_df"],  # type: ignore[arg-type]
        sla_result=sample_analysis_objects["sla_res"],  # type: ignore[arg-type]
        comparison_metrics=sample_analysis_objects["comp_metrics"],  # type: ignore[arg-type]
        group_comparison_df=sample_analysis_objects["group_df"],  # type: ignore[arg-type]
        group_column="server",
    )

    assert "# Server Response Time Performance & Modeling Report" in md_str
    assert "Data Ingestion & Quality Summary" in md_str
    assert "Total Rows Ingested" in md_str
    assert "Descriptive Latency Statistics" in md_str
    assert "Exponential Distribution Model Parameters" in md_str
    assert "Latency Percentiles (Observed vs. Theoretical)" in md_str
    assert "| P50 |" in md_str
    assert "Service Level Agreement (SLA) Compliance" in md_str
    assert "**[FAIL]**" in md_str
    assert "Distribution Model Fit Assessment" in md_str
    assert "Comparative Performance by server" in md_str


def test_generate_markdown_report_handles_missing_sections() -> None:
    md_str = generate_markdown_report(
        source_kind="Generate sample",
        response_column="rt",
    )
    assert "Validation report details not provided" in md_str
    assert "Descriptive statistics not available" in md_str
    assert "Exponential rate parameter not estimated" in md_str


# ---------------------------------------------------------------------------
# 3. CSV Summary Report Tests
# ---------------------------------------------------------------------------


def test_generate_summary_csv(sample_analysis_objects: dict[str, object]) -> None:
    csv_str = generate_summary_csv(
        descriptive_stats=sample_analysis_objects["desc_stats"],  # type: ignore[arg-type]
        estimated_rate=sample_analysis_objects["rate"],  # type: ignore[arg-type]
        estimated_mean=sample_analysis_objects["mean"],  # type: ignore[arg-type]
        percentiles_df=sample_analysis_objects["percentiles_df"],  # type: ignore[arg-type]
        sla_result=sample_analysis_objects["sla_res"],  # type: ignore[arg-type]
        comparison_metrics=sample_analysis_objects["comp_metrics"],  # type: ignore[arg-type]
    )

    df = pd.read_csv(io.StringIO(csv_str))
    assert isinstance(df, pd.DataFrame)
    assert set(["Category", "Metric", "Observed", "Theoretical", "Difference", "Unit"]).issubset(df.columns)

    categories = set(df["Category"])
    assert "Descriptive Statistics" in categories
    assert "Exponential Model" in categories
    assert "Latency Percentiles" in categories
    assert "SLA Compliance" in categories
    assert "Goodness of Fit" in categories
