"""Performance analysis module for server response times.

Calculates observed and theoretical exponential percentiles, evaluates
Service Level Agreement (SLA) threshold compliance, and performs
group-level performance comparisons across servers or endpoints.

All response times are measured in milliseconds (ms), and exponential
rates in inverse milliseconds (1/ms).
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Sequence

import numpy as np
import pandas as pd

from statistical_analysis import _positive_observations, _positive_rate

DEFAULT_PERCENTILES: tuple[float, ...] = (50.0, 75.0, 90.0, 95.0, 99.0)


@dataclass(frozen=True)
class SLAComplianceResult:
    """Structured result of Service Level Agreement (SLA) compliance evaluation.

    Measures the proportion of requests completed within a specified
    time threshold (in ms) against an expected target percentage.
    """

    total_requests: int
    compliant_requests: int
    breached_requests: int
    actual_compliance_percentage: float
    target_compliance_percentage: float
    sla_threshold_ms: float
    is_compliant: bool
    status: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable dictionary representation."""
        return {
            "total_requests": int(self.total_requests),
            "compliant_requests": int(self.compliant_requests),
            "breached_requests": int(self.breached_requests),
            "actual_compliance_percentage": float(self.actual_compliance_percentage),
            "target_compliance_percentage": float(self.target_compliance_percentage),
            "sla_threshold_ms": float(self.sla_threshold_ms),
            "is_compliant": bool(self.is_compliant),
            "status": str(self.status),
        }

    def to_frame(self) -> pd.DataFrame:
        """Return SLA metrics formatted as a single-row display table."""
        return pd.DataFrame(
            [
                {
                    "SLA Threshold (ms)": f"{self.sla_threshold_ms:.2f}",
                    "Target Compliance (%)": f"{self.target_compliance_percentage:.2f}%",
                    "Actual Compliance (%)": f"{self.actual_compliance_percentage:.2f}%",
                    "Compliant Requests": self.compliant_requests,
                    "Breached Requests": self.breached_requests,
                    "Total Requests": self.total_requests,
                    "Status": self.status,
                }
            ]
        )


def theoretical_exponential_percentile(rate: Real, percentile: Real) -> float:
    """Calculate the theoretical exponential percentile in milliseconds.

    For an exponential distribution with rate lambda (1/ms), the quantile
    function is:
        Q(p) = -ln(1 - p) / lambda
    where p = percentile / 100.
    """
    valid_rate = _positive_rate(rate)
    if isinstance(percentile, (bool, np.bool_)) or not isinstance(percentile, Real):
        raise ValueError("Percentile must be a number between 0 and 100 exclusive.")
    pct = float(percentile)
    if not np.isfinite(pct) or pct <= 0.0 or pct >= 100.0:
        raise ValueError("Percentile must be strictly between 0 and 100 exclusive.")

    p = pct / 100.0
    # Q(p) = -log(1 - p) / rate = -log1p(-p) / rate
    value = float(-np.log1p(-p) / valid_rate)
    if not np.isfinite(value) or value < 0:
        raise ValueError("Calculated theoretical percentile is not finite.")
    return value


def calculate_percentiles(
    observations: object,
    rate: Real,
    percentiles: Sequence[Real] = DEFAULT_PERCENTILES,
) -> pd.DataFrame:
    """Calculate observed vs. theoretical exponential percentiles.

    Uses NumPy's standard linear interpolation convention (method='linear')
    for empirical sample percentiles.

    Returns a DataFrame with columns:
    - 'Percentile': e.g., 'P50', 'P75', 'P90', 'P95', 'P99'
    - 'Percentile_Value': numeric percentile (0-100)
    - 'Observed_ms': empirical response-time percentile in ms
    - 'Theoretical_ms': theoretical exponential percentile in ms
    - 'Difference_ms': Observed_ms - Theoretical_ms
    - 'Relative_Difference_pct': (Observed_ms - Theoretical_ms) / Theoretical_ms * 100
    """
    values = _positive_observations(observations)
    valid_rate = _positive_rate(rate)

    if not percentiles:
        raise ValueError("Percentiles sequence must not be empty.")

    clean_percentiles: list[float] = []
    for pct in percentiles:
        if isinstance(pct, (bool, np.bool_)) or not isinstance(pct, Real):
            raise ValueError("All percentiles must be numbers between 0 and 100 exclusive.")
        f_pct = float(pct)
        if not np.isfinite(f_pct) or f_pct <= 0.0 or f_pct >= 100.0:
            raise ValueError("All percentiles must be strictly between 0 and 100 exclusive.")
        clean_percentiles.append(f_pct)

    # Compute observed percentiles using standard linear interpolation
    observed_pcts = np.percentile(values, clean_percentiles, method="linear")

    rows: list[dict[str, object]] = []
    for pct, obs_val in zip(clean_percentiles, observed_pcts):
        theo_val = theoretical_exponential_percentile(valid_rate, pct)
        diff = float(obs_val - theo_val)
        rel_diff = float((diff / theo_val) * 100.0) if theo_val > 0 else 0.0

        pct_label = f"P{int(pct)}" if pct.is_integer() else f"P{pct:.1f}"
        rows.append(
            {
                "Percentile": pct_label,
                "Percentile_Value": pct,
                "Observed_ms": float(obs_val),
                "Theoretical_ms": theo_val,
                "Difference_ms": diff,
                "Relative_Difference_pct": rel_diff,
            }
        )

    return pd.DataFrame(rows)


def evaluate_sla_compliance(
    observations: object,
    sla_threshold_ms: Real,
    target_compliance_percentage: Real = 95.0,
) -> SLAComplianceResult:
    """Evaluate Service Level Agreement (SLA) compliance for response times.

    Parameters:
    - observations: non-empty 1D numeric array of finite positive numbers.
    - sla_threshold_ms: positive finite time threshold in milliseconds.
    - target_compliance_percentage: required compliance rate in percent (e.g. 95.0 for 95%).

    Returns an SLAComplianceResult object.
    """
    values = _positive_observations(observations)

    if isinstance(sla_threshold_ms, (bool, np.bool_)) or not isinstance(sla_threshold_ms, Real):
        raise ValueError("SLA threshold must be a finite positive number in milliseconds.")
    threshold = float(sla_threshold_ms)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("SLA threshold must be a finite positive number in milliseconds.")

    if (
        isinstance(target_compliance_percentage, (bool, np.bool_))
        or not isinstance(target_compliance_percentage, Real)
    ):
        raise ValueError("Target compliance percentage must be a number between 0 and 100.")
    target_pct = float(target_compliance_percentage)
    if not np.isfinite(target_pct) or target_pct <= 0.0 or target_pct > 100.0:
        raise ValueError("Target compliance percentage must be strictly greater than 0 and at most 100.")

    total = int(values.size)
    compliant = int(np.sum(values <= threshold))
    breached = total - compliant
    actual_pct = float((compliant / total) * 100.0)
    is_compliant = bool(actual_pct >= target_pct)
    status = "PASS" if is_compliant else "FAIL"

    return SLAComplianceResult(
        total_requests=total,
        compliant_requests=compliant,
        breached_requests=breached,
        actual_compliance_percentage=actual_pct,
        target_compliance_percentage=target_pct,
        sla_threshold_ms=threshold,
        is_compliant=is_compliant,
        status=status,
    )


def compare_group_performance(
    frame: pd.DataFrame,
    response_column: str,
    group_column: str,
    sla_threshold_ms: Real | None = None,
) -> pd.DataFrame:
    """Compare performance metrics across categories (e.g. server, endpoint).

    Computes request count, mean, median, P95, min, and max response times per group.
    If ``sla_threshold_ms`` is provided, also computes per-group SLA compliance percentage.

    Safely handles single-observation groups and missing/empty group labels.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")
    if frame.empty:
        raise ValueError("The dataset has no rows to compare.")
    if response_column not in frame.columns:
        raise ValueError(f"Response-time column '{response_column}' was not found in the dataset.")
    if group_column not in frame.columns:
        raise ValueError(f"Grouping column '{group_column}' was not found in the dataset.")

    # Validate SLA threshold if provided
    thresh_val: float | None = None
    if sla_threshold_ms is not None:
        if isinstance(sla_threshold_ms, (bool, np.bool_)) or not isinstance(sla_threshold_ms, Real):
            raise ValueError("SLA threshold must be a finite positive number in milliseconds.")
        thresh_val = float(sla_threshold_ms)
        if not np.isfinite(thresh_val) or thresh_val <= 0.0:
            raise ValueError("SLA threshold must be a finite positive number in milliseconds.")

    # Convert group column to string labels, handling NaNs/blanks
    df_clean = frame[[response_column, group_column]].copy()
    raw_group = df_clean[group_column]
    group_labels = raw_group.fillna("(missing)").astype(str).str.strip()
    group_labels = group_labels.replace("", "(empty)")
    df_clean["_group_label"] = group_labels

    # Ensure numeric response times
    df_clean[response_column] = pd.to_numeric(df_clean[response_column], errors="coerce")
    valid_mask = df_clean[response_column].notna() & np.isfinite(df_clean[response_column]) & (df_clean[response_column] > 0)
    df_valid = df_clean.loc[valid_mask]

    if df_valid.empty:
        raise ValueError("No valid positive response times found in the selected columns.")

    rows: list[dict[str, object]] = []
    for group_name, group_data in df_valid.groupby("_group_label", sort=True):
        times = group_data[response_column].to_numpy(dtype=float)
        n = int(times.size)
        mean_val = float(np.mean(times))
        median_val = float(np.median(times))
        p95_val = float(np.percentile(times, 95.0, method="linear"))
        min_val = float(np.min(times))
        max_val = float(np.max(times))

        row: dict[str, object] = {
            group_column: str(group_name),
            "Request_Count": n,
            "Mean_ms": mean_val,
            "Median_ms": median_val,
            "P95_ms": p95_val,
            "Min_ms": min_val,
            "Max_ms": max_val,
        }

        if thresh_val is not None:
            comp_count = int(np.sum(times <= thresh_val))
            comp_pct = float((comp_count / n) * 100.0)
            row["SLA_Compliance_pct"] = comp_pct

        rows.append(row)

    return pd.DataFrame(rows)
