"""Reporting module for server response time modeling and performance analysis.

Generates structured JSON analysis reports, executive Markdown summaries,
and tabular CSV summaries combining data quality, descriptive statistics,
exponential model parameters, percentile latency metrics, SLA evaluations,
and goodness-of-fit indicators.
"""

from __future__ import annotations

import datetime
import io
import json
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd

from data_management import ValidationReport
from distribution_modeling import DistributionComparisonMetrics
from performance_analysis import SLAComplianceResult
from statistical_analysis import DescriptiveStatistics


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively convert NumPy numbers, NaNs, and custom types to JSON primitives."""
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_sanitize_for_json(item) for item in obj]
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        if not np.isfinite(obj):
            return None
        return float(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, pd.DataFrame):
        return _sanitize_for_json(obj.to_dict(orient="records"))
    if isinstance(obj, pd.Series):
        return _sanitize_for_json(obj.to_dict())
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return _sanitize_for_json(obj.to_dict())
    return obj


def generate_json_report(
    source_kind: str,
    response_column: str,
    validation_report: ValidationReport | None = None,
    descriptive_stats: DescriptiveStatistics | None = None,
    estimated_rate: float | None = None,
    estimated_mean: float | None = None,
    percentiles_df: pd.DataFrame | None = None,
    sla_result: SLAComplianceResult | None = None,
    comparison_metrics: DistributionComparisonMetrics | dict[str, float] | None = None,
    group_comparison_df: pd.DataFrame | None = None,
    group_column: str | None = None,
) -> str:
    """Generate a comprehensive, structured JSON report of all analysis results.

    All fields are JSON-serializable. Missing or unavailable metrics are
    represented as null.
    """
    timestamp_str = datetime.datetime.now(datetime.timezone.utc).isoformat()

    report: dict[str, Any] = {
        "report_metadata": {
            "title": "Server Response Time Performance & Exponential Model Analysis",
            "generated_at_utc": timestamp_str,
            "data_source": str(source_kind),
            "response_time_column": str(response_column),
            "unit": "milliseconds (ms)",
            "rate_unit": "inverse milliseconds (1/ms)",
        }
    }

    # 1. Validation Report
    if validation_report is not None:
        report["data_validation"] = {
            "total_rows": int(validation_report.total_rows),
            "valid_rows": int(validation_report.valid_rows),
            "excluded_rows": int(validation_report.invalid_rows),
            "issue_counts": {
                k: int(v) for k, v in validation_report.issue_counts.items()
            },
        }
    else:
        report["data_validation"] = None

    # 2. Descriptive Statistics
    if descriptive_stats is not None:
        report["descriptive_statistics"] = {
            "count": int(descriptive_stats.count),
            "mean_ms": float(descriptive_stats.mean),
            "median_ms": float(descriptive_stats.median),
            "minimum_ms": float(descriptive_stats.minimum),
            "maximum_ms": float(descriptive_stats.maximum),
            "population_variance_ms2": float(descriptive_stats.population_variance),
            "population_std_ms": float(descriptive_stats.population_standard_deviation),
            "sample_variance_ms2": (
                float(descriptive_stats.sample_variance)
                if descriptive_stats.sample_variance is not None
                else None
            ),
            "sample_std_ms": (
                float(descriptive_stats.sample_standard_deviation)
                if descriptive_stats.sample_standard_deviation is not None
                else None
            ),
        }
    else:
        report["descriptive_statistics"] = None

    # 3. Exponential Model Parameters
    if estimated_rate is not None:
        report["exponential_model"] = {
            "estimated_rate_lambda_per_ms": float(estimated_rate),
            "estimated_mean_mu_ms": (
                float(estimated_mean)
                if estimated_mean is not None
                else (1.0 / float(estimated_rate) if estimated_rate > 0 else None)
            ),
            "scale_parameter_ms": (
                float(estimated_mean)
                if estimated_mean is not None
                else (1.0 / float(estimated_rate) if estimated_rate > 0 else None)
            ),
        }
    else:
        report["exponential_model"] = None

    # 4. Latency Percentiles
    if percentiles_df is not None and not percentiles_df.empty:
        report["percentile_analysis"] = percentiles_df.to_dict(orient="records")
    else:
        report["percentile_analysis"] = None

    # 5. SLA Compliance
    if sla_result is not None:
        report["sla_evaluation"] = sla_result.to_dict()
    else:
        report["sla_evaluation"] = None

    # 6. Distribution Goodness-of-Fit Metrics
    if comparison_metrics is not None:
        if isinstance(comparison_metrics, DistributionComparisonMetrics):
            report["goodness_of_fit"] = {
                "observed_mean_ms": float(comparison_metrics.observed_mean),
                "theoretical_mean_ms": float(comparison_metrics.theoretical_mean),
                "mean_difference_ms": float(comparison_metrics.observed_mean - comparison_metrics.theoretical_mean),
                "observed_median_ms": float(comparison_metrics.observed_median),
                "theoretical_median_ms": float(comparison_metrics.theoretical_median),
                "median_difference_ms": float(comparison_metrics.observed_median - comparison_metrics.theoretical_median),
                "observed_std_ms": float(comparison_metrics.observed_std),
                "theoretical_std_ms": float(comparison_metrics.theoretical_std),
                "std_difference_ms": float(comparison_metrics.observed_std - comparison_metrics.theoretical_std),
                "ks_statistic": float(comparison_metrics.ks_statistic),
                "max_cdf_distance": float(comparison_metrics.cdf_distance),
            }
        else:
            report["goodness_of_fit"] = _sanitize_for_json(comparison_metrics)
    else:
        report["goodness_of_fit"] = None

    # 7. Group Performance Comparison
    if group_comparison_df is not None and not group_comparison_df.empty:
        report["group_comparison"] = {
            "group_column": group_column or "group",
            "groups": group_comparison_df.to_dict(orient="records"),
        }
    else:
        report["group_comparison"] = None

    sanitized = _sanitize_for_json(report)
    return json.dumps(sanitized, indent=2)


def generate_markdown_report(
    source_kind: str,
    response_column: str,
    validation_report: ValidationReport | None = None,
    descriptive_stats: DescriptiveStatistics | None = None,
    estimated_rate: float | None = None,
    estimated_mean: float | None = None,
    percentiles_df: pd.DataFrame | None = None,
    sla_result: SLAComplianceResult | None = None,
    comparison_metrics: DistributionComparisonMetrics | dict[str, float] | None = None,
    group_comparison_df: pd.DataFrame | None = None,
    group_column: str | None = None,
) -> str:
    """Generate a clean, readable Markdown executive analysis report."""
    timestamp_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines: list[str] = [
        "# Server Response Time Performance & Modeling Report",
        "",
        f"**Generated:** {timestamp_str}  ",
        f"**Data Source:** {source_kind}  ",
        f"**Target Column:** `{response_column}` (milliseconds)  ",
        "",
        "---",
        "",
    ]

    # Section 1: Data Validation & Ingestion
    lines.extend(["## 1. Data Ingestion & Quality Summary", ""])
    if validation_report is not None:
        lines.append(
            f"- **Total Rows Ingested:** {validation_report.total_rows:,}  \n"
            f"- **Valid Observations Retained:** {validation_report.valid_rows:,}  \n"
            f"- **Excluded Rows:** {validation_report.invalid_rows:,}  "
        )
        if validation_report.invalid_rows > 0:
            lines.append("\n**Excluded Observations Breakdown:**\n")
            lines.append("| Issue Category | Count |")
            lines.append("| :--- | :--- |")
            for issue, count in validation_report.issue_counts.items():
                if count > 0:
                    lines.append(f"| {issue.capitalize()} | {count:,} |")
        lines.append("")
    else:
        lines.append("*Validation report details not provided.*\n")

    # Section 2: Descriptive Statistics
    lines.extend(["## 2. Descriptive Latency Statistics", ""])
    if descriptive_stats is not None:
        sample_std_str = (
            f"{descriptive_stats.sample_standard_deviation:.4f}"
            if descriptive_stats.sample_standard_deviation is not None
            else "N/A (n < 2)"
        )
        sample_var_str = (
            f"{descriptive_stats.sample_variance:.4f}"
            if descriptive_stats.sample_variance is not None
            else "N/A (n < 2)"
        )
        lines.append("| Metric | Value (ms) |")
        lines.append("| :--- | :--- |")
        lines.append(f"| Sample Count (n) | {descriptive_stats.count:,} |")
        lines.append(f"| Mean Response Time | {descriptive_stats.mean:.4f} |")
        lines.append(f"| Median Response Time | {descriptive_stats.median:.4f} |")
        lines.append(f"| Minimum Response Time | {descriptive_stats.minimum:.4f} |")
        lines.append(f"| Maximum Response Time | {descriptive_stats.maximum:.4f} |")
        lines.append(f"| Sample Std. Deviation | {sample_std_str} |")
        lines.append(f"| Sample Variance | {sample_var_str} |")
        lines.append(f"| Population Std. Deviation | {descriptive_stats.population_standard_deviation:.4f} |")
        lines.append("")
    else:
        lines.append("*Descriptive statistics not available.*\n")

    # Section 3: Exponential Model Estimation
    lines.extend(["## 3. Exponential Distribution Model Parameters", ""])
    if estimated_rate is not None:
        mean_val = estimated_mean if estimated_mean is not None else (1.0 / estimated_rate if estimated_rate > 0 else 0.0)
        lines.append(
            rf"- **Estimated Rate Parameter ($\hat{{\lambda}}$):** `{estimated_rate:.6f}` 1/ms  " + "\n"
            rf"- **Estimated Theoretical Mean ($\hat{{\mu}} = 1/\hat{{\lambda}}$):** `{mean_val:.4f}` ms  " + "\n"
            rf"- **Scale Parameter ($\beta = 1/\hat{{\lambda}}$):** `{mean_val:.4f}` ms  " + "\n"
            r"- **Estimation Method:** Maximum Likelihood Estimation (MLE) on validated observations."
        )
        lines.append("")
    else:
        lines.append("*Exponential rate parameter not estimated.*\n")

    # Section 4: Percentile Latency Analysis
    lines.extend(["## 4. Latency Percentiles (Observed vs. Theoretical)", ""])
    if percentiles_df is not None and not percentiles_df.empty:
        lines.append(
            "Percentiles evaluate tail latency and SLA boundaries. Observed percentiles "
            "use standard linear empirical interpolation; theoretical percentiles are derived from "
            "the exponential quantile function $Q(p) = -\\ln(1-p)/\\hat{\\lambda}$."
        )
        lines.append("")
        lines.append("| Percentile | Observed (ms) | Theoretical (ms) | Difference (ms) | Relative Diff (%) |")
        lines.append("| :--- | :--- | :--- | :--- | :--- |")
        for _, row in percentiles_df.iterrows():
            pct = row["Percentile"]
            obs = f"{float(row['Observed_ms']):.4f}"
            theo = f"{float(row['Theoretical_ms']):.4f}"
            diff = f"{float(row['Difference_ms']):+.4f}"
            rel_diff = f"{float(row['Relative_Difference_pct']):+.2f}%"
            lines.append(f"| {pct} | {obs} | {theo} | {diff} | {rel_diff} |")
        lines.append("")
    else:
        lines.append("*Percentile calculations not available.*\n")

    # Section 5: SLA Compliance
    lines.extend(["## 5. Service Level Agreement (SLA) Compliance", ""])
    if sla_result is not None:
        status_badge = "**[PASS]**" if sla_result.is_compliant else "**[FAIL]**"
        lines.append(f"### Status: {status_badge}\n")
        lines.append(f"- **Configured SLA Latency Threshold:** `{sla_result.sla_threshold_ms:.2f}` ms")
        lines.append(f"- **Target SLA Compliance:** `{sla_result.target_compliance_percentage:.2f}%`")
        lines.append(f"- **Actual Compliance Rate:** `{sla_result.actual_compliance_percentage:.2f}%`")
        lines.append(rf"- **Compliant Requests ($\le {sla_result.sla_threshold_ms:.2f}$ ms):** {sla_result.compliant_requests:,} / {sla_result.total_requests:,}")
        lines.append(f"- **Breached Requests ($> {sla_result.sla_threshold_ms:.2f}$ ms):** {sla_result.breached_requests:,}")
        lines.append("")
        if not sla_result.is_compliant:
            deficit = sla_result.target_compliance_percentage - sla_result.actual_compliance_percentage
            lines.append(f"> **Warning:** The server response times fell short of the SLA target by `{deficit:.2f}%`. Investigation into tail latency is recommended.")
            lines.append("")
    else:
        lines.append("*SLA compliance evaluation not configured.*\n")

    # Section 6: Model Goodness-of-Fit Evaluation
    lines.extend(["## 6. Distribution Model Fit Assessment", ""])
    if comparison_metrics is not None:
        if isinstance(comparison_metrics, DistributionComparisonMetrics):
            ks = comparison_metrics.ks_statistic
            lines.append("| Metric | Observed | Theoretical | Difference (Obs - Theo) |")
            lines.append("| :--- | :--- | :--- | :--- |")
            lines.append(f"| Mean (ms) | {comparison_metrics.observed_mean:.4f} | {comparison_metrics.theoretical_mean:.4f} | {comparison_metrics.observed_mean - comparison_metrics.theoretical_mean:+.4f} |")
            lines.append(f"| Median (ms) | {comparison_metrics.observed_median:.4f} | {comparison_metrics.theoretical_median:.4f} | {comparison_metrics.observed_median - comparison_metrics.theoretical_median:+.4f} |")
            lines.append(f"| Std. Deviation (ms) | {comparison_metrics.observed_std:.4f} | {comparison_metrics.theoretical_std:.4f} | {comparison_metrics.observed_std - comparison_metrics.theoretical_std:+.4f} |")
            lines.append(f"| KS Statistic (Max CDF Distance) | {ks:.4f} | 0.0000 | {ks:.4f} |")
            lines.append("")
            # Objective commentary based on KS distance
            if ks < 0.05:
                fit_comment = "The empirical distribution closely tracks the theoretical exponential model (KS distance < 0.05)."
            elif ks < 0.15:
                fit_comment = "The empirical distribution moderately aligns with the exponential model (KS distance < 0.15), with mild tail or median divergence."
            else:
                fit_comment = f"Notable divergence from an exponential distribution is observed (KS distance = {ks:.4f}). Real-world server delays may be multimodal or heavy-tailed."
            lines.append(f"**Fit Assessment:** {fit_comment}\n")
        else:
            lines.append(f"Goodness-of-fit metrics: {comparison_metrics}\n")
    else:
        lines.append("*Goodness-of-fit comparison metrics not available.*\n")

    # Section 7: Group Performance Breakdown (if present)
    if group_comparison_df is not None and not group_comparison_df.empty:
        col_label = group_column or "Group"
        lines.extend([f"## 7. Comparative Performance by {col_label}", ""])
        cols = list(group_comparison_df.columns)
        header_row = "| " + " | ".join(cols) + " |"
        sep_row = "| " + " | ".join([":---"] * len(cols)) + " |"
        lines.append(header_row)
        lines.append(sep_row)
        for _, row in group_comparison_df.iterrows():
            vals: list[str] = []
            for c in cols:
                val = row[c]
                if isinstance(val, (float, np.floating)):
                    vals.append(f"{val:.2f}")
                else:
                    vals.append(str(val))
            lines.append("| " + " | ".join(vals) + " |")
        lines.append("")

    return "\n".join(lines)


def generate_summary_csv(
    descriptive_stats: DescriptiveStatistics | None = None,
    estimated_rate: float | None = None,
    estimated_mean: float | None = None,
    percentiles_df: pd.DataFrame | None = None,
    sla_result: SLAComplianceResult | None = None,
    comparison_metrics: DistributionComparisonMetrics | None = None,
) -> str:
    """Generate a consolidated summary CSV comparing observed and theoretical metrics."""
    rows: list[dict[str, object]] = []

    # 1. Descriptive stats
    if descriptive_stats is not None:
        rows.append({
            "Category": "Descriptive Statistics",
            "Metric": "Sample Count",
            "Observed": descriptive_stats.count,
            "Theoretical": "N/A",
            "Difference": "N/A",
            "Unit": "count",
        })
        rows.append({
            "Category": "Descriptive Statistics",
            "Metric": "Mean Response Time",
            "Observed": round(descriptive_stats.mean, 4),
            "Theoretical": round(estimated_mean, 4) if estimated_mean else "N/A",
            "Difference": round(descriptive_stats.mean - estimated_mean, 4) if estimated_mean else "N/A",
            "Unit": "ms",
        })
        theo_median = (float(np.log(2.0) / estimated_rate)) if estimated_rate else None
        rows.append({
            "Category": "Descriptive Statistics",
            "Metric": "Median Response Time",
            "Observed": round(descriptive_stats.median, 4),
            "Theoretical": round(theo_median, 4) if theo_median else "N/A",
            "Difference": round(descriptive_stats.median - theo_median, 4) if theo_median else "N/A",
            "Unit": "ms",
        })
        rows.append({
            "Category": "Descriptive Statistics",
            "Metric": "Minimum Response Time",
            "Observed": round(descriptive_stats.minimum, 4),
            "Theoretical": "0.0",
            "Difference": round(descriptive_stats.minimum, 4),
            "Unit": "ms",
        })
        rows.append({
            "Category": "Descriptive Statistics",
            "Metric": "Maximum Response Time",
            "Observed": round(descriptive_stats.maximum, 4),
            "Theoretical": "Infinity",
            "Difference": "N/A",
            "Unit": "ms",
        })
        if descriptive_stats.sample_standard_deviation is not None:
            theo_std = round(1.0 / estimated_rate, 4) if estimated_rate else "N/A"
            diff_std = (
                round(descriptive_stats.sample_standard_deviation - (1.0 / estimated_rate), 4)
                if estimated_rate
                else "N/A"
            )
            rows.append({
                "Category": "Descriptive Statistics",
                "Metric": "Sample Standard Deviation",
                "Observed": round(descriptive_stats.sample_standard_deviation, 4),
                "Theoretical": theo_std,
                "Difference": diff_std,
                "Unit": "ms",
            })

    # 2. Exponential Model
    if estimated_rate is not None:
        rows.append({
            "Category": "Exponential Model",
            "Metric": "Estimated Rate (lambda)",
            "Observed": round(estimated_rate, 6),
            "Theoretical": "N/A",
            "Difference": "N/A",
            "Unit": "1/ms",
        })

    # 3. Percentiles
    if percentiles_df is not None and not percentiles_df.empty:
        for _, p_row in percentiles_df.iterrows():
            rows.append({
                "Category": "Latency Percentiles",
                "Metric": str(p_row["Percentile"]),
                "Observed": round(float(p_row["Observed_ms"]), 4),
                "Theoretical": round(float(p_row["Theoretical_ms"]), 4),
                "Difference": round(float(p_row["Difference_ms"]), 4),
                "Unit": "ms",
            })

    # 4. SLA Evaluation
    if sla_result is not None:
        rows.append({
            "Category": "SLA Compliance",
            "Metric": f"Compliance Rate (<= {sla_result.sla_threshold_ms:.1f} ms)",
            "Observed": f"{sla_result.actual_compliance_percentage:.2f}%",
            "Theoretical": f"{sla_result.target_compliance_percentage:.2f}% (Target)",
            "Difference": f"{sla_result.actual_compliance_percentage - sla_result.target_compliance_percentage:+.2f}%",
            "Unit": "percentage",
        })
        rows.append({
            "Category": "SLA Compliance",
            "Metric": "Breached Requests",
            "Observed": sla_result.breached_requests,
            "Theoretical": 0,
            "Difference": sla_result.breached_requests,
            "Unit": "count",
        })

    # 5. Goodness-of-Fit
    if comparison_metrics is not None and isinstance(comparison_metrics, DistributionComparisonMetrics):
        rows.append({
            "Category": "Goodness of Fit",
            "Metric": "Kolmogorov-Smirnov Statistic (D_ks)",
            "Observed": round(comparison_metrics.ks_statistic, 4),
            "Theoretical": "0.0",
            "Difference": round(comparison_metrics.ks_statistic, 4),
            "Unit": "dimensionless",
        })

    df = pd.DataFrame(rows)
    output = io.StringIO()
    df.to_csv(output, index=False)
    return output.getvalue()
