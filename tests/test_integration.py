from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_management import (
    generate_sample_response_times,
    load_csv,
    validate_and_clean_response_times,
)
from distribution_modeling import calculate_distribution_comparison_metrics
from performance_analysis import (
    calculate_percentiles,
    compare_group_performance,
    evaluate_sla_compliance,
)
from reporting import (
    generate_json_report,
    generate_markdown_report,
    generate_summary_csv,
)
from statistical_analysis import (
    descriptive_statistics,
    estimate_exponential_rate,
    estimated_mean_response_time,
)


def test_full_pipeline_with_sample_csv() -> None:
    csv_path = Path("sample_response_times.csv")
    assert csv_path.exists(), "sample_response_times.csv must exist in workspace root."

    # 1. Ingest
    raw_df = load_csv(csv_path)
    assert len(raw_df) == 16
    assert "response_time_ms" in raw_df.columns
    assert "server" in raw_df.columns

    # 2. Validate and Clean
    cleaned_df, report = validate_and_clean_response_times(raw_df, "response_time_ms")
    assert report.valid_rows == 16
    assert report.invalid_rows == 0
    assert len(cleaned_df) == 16

    # 3. Descriptive Stats & Rate Estimation
    times = cleaned_df["response_time_ms"]
    stats = descriptive_statistics(times)
    rate = estimate_exponential_rate(times)
    mean = estimated_mean_response_time(rate)
    assert stats.count == 16
    assert stats.mean == pytest.approx(mean)
    assert rate > 0

    # 4. Goodness of Fit
    comp_metrics = calculate_distribution_comparison_metrics(times, rate)
    assert 0.0 <= comp_metrics.ks_statistic <= 1.0

    # 5. Performance Analysis: Percentiles & SLA
    pct_df = calculate_percentiles(times, rate)
    assert len(pct_df) == 5

    sla_result = evaluate_sla_compliance(times, sla_threshold_ms=150.0, target_compliance_percentage=75.0)
    assert sla_result.total_requests == 16
    assert isinstance(sla_result.is_compliant, bool)

    # 6. Group Performance Comparison (by server and by endpoint)
    group_server = compare_group_performance(cleaned_df, "response_time_ms", "server", sla_threshold_ms=150.0)
    assert len(group_server) == 3  # api-east, api-west, api-central

    group_endpoint = compare_group_performance(cleaned_df, "response_time_ms", "endpoint", sla_threshold_ms=150.0)
    assert len(group_endpoint) == 4  # /health, /orders, /products, /users

    # 7. Reporting: JSON, Markdown, CSV
    json_report = generate_json_report(
        source_kind="Upload CSV",
        response_column="response_time_ms",
        validation_report=report,
        descriptive_stats=stats,
        estimated_rate=rate,
        estimated_mean=mean,
        percentiles_df=pct_df,
        sla_result=sla_result,
        comparison_metrics=comp_metrics,
        group_comparison_df=group_server,
        group_column="server",
    )
    parsed_json = json.loads(json_report)
    assert parsed_json["data_validation"]["valid_rows"] == 16
    assert parsed_json["exponential_model"]["estimated_rate_lambda_per_ms"] == pytest.approx(rate)

    md_report = generate_markdown_report(
        source_kind="Upload CSV",
        response_column="response_time_ms",
        validation_report=report,
        descriptive_stats=stats,
        estimated_rate=rate,
        estimated_mean=mean,
        percentiles_df=pct_df,
        sla_result=sla_result,
        comparison_metrics=comp_metrics,
        group_comparison_df=group_server,
        group_column="server",
    )
    assert "# Server Response Time Performance & Modeling Report" in md_report
    assert "Comparative Performance by server" in md_report

    csv_summary = generate_summary_csv(
        descriptive_stats=stats,
        estimated_rate=rate,
        estimated_mean=mean,
        percentiles_df=pct_df,
        sla_result=sla_result,
        comparison_metrics=comp_metrics,
    )
    assert "Sample Count" in csv_summary
    assert "P95" in csv_summary


def test_full_pipeline_with_generated_sample_data() -> None:
    # Reproducible generated dataset
    count = 100
    expected_rate = 0.025
    raw_df = generate_sample_response_times(count=count, rate_per_ms=expected_rate, seed=42)

    cleaned_df, report = validate_and_clean_response_times(raw_df, "response_time_ms")
    assert report.valid_rows == count
    assert report.invalid_rows == 0

    times = cleaned_df["response_time_ms"]
    stats = descriptive_statistics(times)
    rate = estimate_exponential_rate(times)
    mean = estimated_mean_response_time(rate)

    pct_df = calculate_percentiles(times, rate)
    sla_result = evaluate_sla_compliance(times, sla_threshold_ms=60.0, target_compliance_percentage=80.0)

    json_report = generate_json_report(
        source_kind="Generate sample",
        response_column="response_time_ms",
        validation_report=report,
        descriptive_stats=stats,
        estimated_rate=rate,
        estimated_mean=mean,
        percentiles_df=pct_df,
        sla_result=sla_result,
    )
    data = json.loads(json_report)
    assert data["report_metadata"]["data_source"] == "Generate sample"
    assert data["descriptive_statistics"]["count"] == count


def test_full_pipeline_with_dirty_dataset() -> None:
    dirty_data = pd.DataFrame(
        {
            "server": ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9"],
            "latency": [" 25.4 ", "", "NaN", "invalid", -10.0, 0.0, np.inf, 48.2, 75.1],
        }
    )
    cleaned_df, report = validate_and_clean_response_times(dirty_data, "latency")
    assert report.total_rows == 9
    assert report.valid_rows == 3
    assert report.invalid_rows == 6

    times = cleaned_df["latency"]
    stats = descriptive_statistics(times)
    rate = estimate_exponential_rate(times)
    mean = estimated_mean_response_time(rate)

    pct_df = calculate_percentiles(times, rate)
    sla_result = evaluate_sla_compliance(times, sla_threshold_ms=50.0)
    groups = compare_group_performance(cleaned_df, "latency", "server")

    json_rep = generate_json_report(
        source_kind="Upload CSV",
        response_column="latency",
        validation_report=report,
        descriptive_stats=stats,
        estimated_rate=rate,
        estimated_mean=mean,
        percentiles_df=pct_df,
        sla_result=sla_result,
        group_comparison_df=groups,
        group_column="server",
    )
    parsed = json.loads(json_rep)
    assert parsed["data_validation"]["excluded_rows"] == 6
    assert parsed["descriptive_statistics"]["count"] == 3


def test_full_pipeline_single_observation() -> None:
    single_df = pd.DataFrame({"response_time_ms": [50.0], "server": ["node-1"]})
    cleaned_df, report = validate_and_clean_response_times(single_df, "response_time_ms")

    times = cleaned_df["response_time_ms"]
    stats = descriptive_statistics(times)
    assert stats.sample_variance is None
    assert stats.sample_standard_deviation is None

    rate = estimate_exponential_rate(times)
    pct_df = calculate_percentiles(times, rate)
    sla_result = evaluate_sla_compliance(times, sla_threshold_ms=100.0)

    md_report = generate_markdown_report(
        source_kind="Upload CSV",
        response_column="response_time_ms",
        validation_report=report,
        descriptive_stats=stats,
        estimated_rate=rate,
        percentiles_df=pct_df,
        sla_result=sla_result,
    )
    assert "N/A (n < 2)" in md_report
