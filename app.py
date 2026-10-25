"""Member 1 Streamlit interface for response-time data management."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from data_management import (
    CSVDataError,
    generate_sample_response_times,
    load_csv,
    validate_and_clean_response_times,
)
from statistical_analysis import (
    descriptive_statistics,
    estimate_exponential_rate,
    estimated_mean_response_time,
    exponential_cdf,
    survival_probability,
)
from distribution_modeling import (
    calculate_distribution_comparison_metrics,
    compute_ecdf_comparison,
    compute_histogram_data,
    generate_distribution_curves,
    interval_probability,
)
from performance_analysis import (
    calculate_percentiles,
    evaluate_sla_compliance,
)
from reporting import (
    generate_json_report,
    generate_markdown_report,
    generate_summary_csv,
)

st.set_page_config(
    page_title="Server Response Time Modeling",
    page_icon=":material/timer:",
    layout="wide",
)
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    [data-testid="stMetric"] {
        min-height: 6.25rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def show_formula_metric(
    column: st.delta_generator.DeltaGenerator,
    label: str,
    value: str,
    formula: str,
) -> None:
    with column:
        with st.container(border=True):
            st.markdown(f"**{label}**")
            st.caption(f"${formula}$")
            st.markdown(f"### {value}")


st.title("Server Response Time Modeling Using Exponential Distribution")
st.caption("Load, generate, validate, clean, and export response-time data.")

source_kind = st.radio(
    "Data source",
    ("Upload CSV", "Generate sample"),
    horizontal=True,
)

try:
    if source_kind == "Upload CSV":
        uploaded = st.file_uploader("Upload a CSV file", type="csv")
        if uploaded is None:
            st.info("Upload a CSV file to begin.")
            st.stop()
        raw_data = load_csv(uploaded)
    else:
        count = st.number_input(
            "Number of observations",
            min_value=1,
            value=100,
            step=1,
        )
        rate = st.number_input(
            "Exponential rate (1/ms)",
            min_value=0.000001,
            value=0.02,
            format="%.6f",
        )
        seed = st.number_input("Random seed", min_value=0, value=42, step=1)
        raw_data = generate_sample_response_times(int(count), float(rate), int(seed))
except (CSVDataError, ValueError) as error:
    st.error(str(error))
    st.stop()

st.session_state["member1_raw_data"] = raw_data
st.session_state["member1_source_kind"] = source_kind

st.subheader("Uploaded or generated dataset")
st.dataframe(raw_data, width="stretch", hide_index=True)

response_column = st.selectbox(
    "Response-time column (milliseconds)",
    list(raw_data.columns),
)

try:
    cleaned_data, report = validate_and_clean_response_times(
        raw_data,
        response_column,
    )
except (ValueError, TypeError) as error:
    st.error(f"Unable to validate this dataset: {error}")
    st.stop()

st.session_state["member1_cleaned_data"] = cleaned_data

if cleaned_data.empty:
    st.warning(
        "No valid response-time observations remain. "
        "Select another column or correct the input data."
    )
else:
    try:
        statistics = descriptive_statistics(cleaned_data[response_column])
        estimated_rate = estimate_exponential_rate(cleaned_data[response_column])
        estimated_mean = estimated_mean_response_time(estimated_rate)
    except ValueError as error:
        st.error(f"Statistical analysis could not be calculated: {error}")
    else:
        st.subheader("Statistical Analysis")
        st.caption(
            "Empirical statistics summarize validated observations. Model-based "
            "probabilities below use the fitted exponential distribution."
        )

        with st.container(border=True):
            st.markdown("**Dataset Summary**")
            summary_cols = st.columns(3)
            summary_cols[0].metric(
                "Valid observations",
                f"{statistics.count:,}",
                help="Number of finite, strictly positive response times used in this analysis.",
            )
            show_formula_metric(
                summary_cols[1],
                "Minimum (x_min) · ms",
                f"{statistics.minimum:.6g}",
                r"x_{\min}=\min(x_1,\ldots,x_n)",
            )
            show_formula_metric(
                summary_cols[2],
                "Maximum (x_max) · ms",
                f"{statistics.maximum:.6g}",
                r"x_{\max}=\max(x_1,\ldots,x_n)",
            )

        with st.container(border=True):
            st.markdown("**Descriptive Statistics · Empirical**")
            descriptive_cols = st.columns(3)
            show_formula_metric(
                descriptive_cols[0],
                "Mean (x̄) · ms",
                f"{statistics.mean:.6g}",
                r"\bar{x}=\frac{1}{n}\sum_{i=1}^{n}x_i",
            )
            show_formula_metric(
                descriptive_cols[1],
                "Median (x̃) · ms",
                f"{statistics.median:.6g}",
                r"\tilde{x}=\text{middle value of sorted observations}",
            )
            show_formula_metric(
                descriptive_cols[2],
                "Population variance (σ²) · ms²",
                f"{statistics.population_variance:.6g}",
                r"\sigma^2=\frac{1}{n}\sum_{i=1}^{n}(x_i-\bar{x})^2",
            )
            descriptive_cols = st.columns(3)
            show_formula_metric(
                descriptive_cols[0],
                "Population standard deviation (σ) · ms",
                f"{statistics.population_standard_deviation:.6g}",
                r"\sigma=\sqrt{\sigma^2}",
            )
            show_formula_metric(
                descriptive_cols[1],
                "Sample variance (s²) · ms²",
                (
                    f"{statistics.sample_variance:.6g}"
                    if statistics.sample_variance is not None
                    else "Not defined (n < 2)"
                ),
                r"s^2=\frac{1}{n-1}\sum_{i=1}^{n}(x_i-\bar{x})^2",
            )
            show_formula_metric(
                descriptive_cols[2],
                "Sample standard deviation (s) · ms",
                (
                    f"{statistics.sample_standard_deviation:.6g}"
                    if statistics.sample_standard_deviation is not None
                    else "Not defined (n < 2)"
                ),
                r"s=\sqrt{s^2}",
            )
            st.caption("Sample variance and standard deviation are undefined when n < 2.")

        with st.container(border=True):
            st.markdown("**Exponential Model Parameters · Fitted**")
            model_cols = st.columns(2)
            show_formula_metric(
                model_cols[0],
                "Estimated rate (λ̂) · 1/ms",
                f"{estimated_rate:.6g}",
                r"\hat{\lambda}=\frac{1}{\bar{x}}",
            )
            show_formula_metric(
                model_cols[1],
                "Estimated mean (E(X) = 1/λ̂) · ms",
                f"{estimated_mean:.6g}",
                r"E(X)=\frac{1}{\hat{\lambda}}=\bar{x}",
            )
            st.caption(
                r"$X$ is server response time, $x_i$ is an observed response time, "
                r"and $\lambda$ is the exponential rate in $1/\mathrm{ms}$. "
                r"With observed mean $\bar{x}$, $\hat{\lambda}=1/\bar{x}$."
            )
            st.caption("Theoretical moments for the fitted model:")
            st.latex(
                r"E(X)=\frac{1}{\lambda},\qquad "
                r"\operatorname{Var}(X)=\frac{1}{\lambda^2},\qquad "
                r"\sigma=\frac{1}{\lambda}"
            )

        st.divider()
        st.subheader("Exponential Distribution Probabilities")
        st.caption(
            r"Here, $X$ is server response time and $\lambda=\hat{\lambda}$ is the "
            r"estimated rate in $1/\mathrm{ms}$."
        )
        calc_col1, calc_col2 = st.columns(2)

        dataset_token = f"{source_kind}_{response_column}_{len(cleaned_data)}"
        if st.session_state.get("_prev_dataset_token") != dataset_token:
            st.session_state["_prev_dataset_token"] = dataset_token
            st.session_state.pop("prob_threshold_t", None)
            st.session_state.pop("prob_lower_a", None)
            st.session_state.pop("prob_upper_b", None)

        default_t = float(round(statistics.median, 2))
        default_a = float(round(statistics.minimum, 2))
        default_b = float(round(statistics.mean, 2))
        if default_a > default_b:
            default_b = default_a + 10.0

        with calc_col1, st.container(border=True):
            st.markdown("**Probability of Response Time Within Threshold**")
            threshold_t = st.number_input(
                "Time threshold t (ms)",
                min_value=0.0,
                value=default_t,
                step=1.0,
                key="prob_threshold_t",
            )
            prob_le = float(exponential_cdf(estimated_rate, threshold_t))
            prob_gt = float(survival_probability(estimated_rate, threshold_t))

            m_col1, m_col2 = st.columns(2)
            show_formula_metric(
                m_col1,
                "P(X ≤ t) · within threshold",
                f"{prob_le * 100:.2f}%",
                r"P(X\leq t)=1-e^{-\hat{\lambda}t}",
            )
            m_col1.caption(f"Decimal probability: {prob_le:.6f}")
            show_formula_metric(
                m_col2,
                "P(X > t) · exceeds threshold",
                f"{prob_gt * 100:.2f}%",
                r"P(X>t)=e^{-\hat{\lambda}t}",
            )
            m_col2.caption(f"Decimal probability: {prob_gt:.6f}")
            st.caption("These are model probabilities for a response time measured in milliseconds.")

        with calc_col2, st.container(border=True):
            st.markdown("**Probability of Response Time Within an Interval**")
            b_col1, b_col2 = st.columns(2)
            lower_a = b_col1.number_input(
                "Lower bound a (ms)",
                min_value=0.0,
                value=default_a,
                step=1.0,
                key="prob_lower_a",
            )
            upper_b = b_col2.number_input(
                "Upper bound b (ms)",
                min_value=0.0,
                value=default_b,
                step=1.0,
                key="prob_upper_b",
            )
            if lower_a <= upper_b:
                prob_ab = interval_probability(estimated_rate, lower_a, upper_b)
                st.markdown(f"**P({lower_a:g} ≤ X ≤ {upper_b:g} ms)**")
                st.caption(
                    r"$P(a\leq X\leq b)="
                    r"e^{-\hat{\lambda}a}-e^{-\hat{\lambda}b}$"
                )
                st.markdown(f"### {prob_ab * 100:.2f}%")
                st.caption(f"Decimal probability: {prob_ab:.6f}")
            else:
                st.error("Lower bound a must be less than or equal to upper bound b.")

        st.subheader("Exponential Distribution Graph")
        st.caption(
            r"For response times, $f(x)=\lambda e^{-\lambda x}$ and "
            r"$F(x)=P(X\leq x)=1-e^{-\lambda x}$, for $x\geq0$."
        )
        bins_label, bins_slider = st.columns([1, 4])
        current_bins = int(st.session_state.get("hist_bins_slider", 20))
        bins_label.markdown(f"**Histogram bins** · {current_bins}")
        with bins_slider:
            hist_bins = st.slider(
                "Histogram bins",
                min_value=5,
                max_value=50,
                value=20,
                step=1,
                key="hist_bins_slider",
                label_visibility="collapsed",
            )
        hist_df = compute_histogram_data(cleaned_data[response_column], bins=hist_bins)
        max_curve_time = float(max(float(statistics.maximum) * 1.05, estimated_mean * 3.5))
        curve_df = generate_distribution_curves(
            estimated_rate, max_time=max_curve_time, points=200
        )
        histogram_chart_data = hist_df.assign(
            series="Response time distribution",
            time_ms=hist_df["bin_center"],
            pdf=float("nan"),
        )
        pdf_chart_data = curve_df.assign(
            series="Probability density function (PDF)",
            bin_start=float("nan"),
            bin_end=float("nan"),
            bin_label="",
            count=float("nan"),
            density=float("nan"),
        )
        pdf_chart_data["time_ms"] = pdf_chart_data["time_ms"]
        density_chart_data = pd.concat(
            [histogram_chart_data, pdf_chart_data],
            ignore_index=True,
            sort=False,
        )

        chart_colors = {
            "domain": ["Response time distribution", "Probability density function (PDF)"],
            "range": ["#69a8ff", "#f2a65a"],
        }
        graph_col1, graph_col2 = st.columns(2)
        with graph_col1:
            st.markdown("**Response Time Distribution · Probability Density Function (PDF)**")
            st.caption(
                r"Bars show the validated response times; the curve is the fitted "
                r"$f(x)=\hat{\lambda}e^{-\hat{\lambda}x}$."
            )
            st.vega_lite_chart(
                density_chart_data,
                {
                    "width": "container",
                    "height": 320,
                    "layer": [
                        {
                            "transform": [
                                {"filter": "datum.series === 'Response time distribution'"}
                            ],
                            "mark": {"type": "bar", "opacity": 0.55},
                            "encoding": {
                                "x": {
                                    "field": "bin_start",
                                    "type": "quantitative",
                                    "title": "Response Time (ms)",
                                },
                                "x2": {"field": "bin_end"},
                                "y": {
                                    "field": "density",
                                    "type": "quantitative",
                                    "title": "Probability Density (1/ms)",
                                },
                                "color": {
                                    "field": "series",
                                    "type": "nominal",
                                    "scale": chart_colors,
                                    "legend": {"title": None},
                                },
                                "tooltip": [
                                    {"field": "bin_label", "type": "nominal", "title": "Bin"},
                                    {"field": "count", "type": "quantitative", "title": "Count"},
                                    {
                                        "field": "density",
                                        "type": "quantitative",
                                        "title": "Probability Density (1/ms)",
                                        "format": ".4f",
                                    },
                                ],
                            },
                        },
                        {
                            "transform": [
                                {"filter": "datum.series === 'Probability density function (PDF)'"}
                            ],
                            "mark": {"type": "line", "strokeWidth": 3},
                            "encoding": {
                                "x": {
                                    "field": "time_ms",
                                    "type": "quantitative",
                                    "title": "Response Time (ms)",
                                },
                                "y": {
                                    "field": "pdf",
                                    "type": "quantitative",
                                    "title": "Probability Density (1/ms)",
                                },
                                "color": {
                                    "field": "series",
                                    "type": "nominal",
                                    "scale": chart_colors,
                                    "legend": {"title": None},
                                },
                                "tooltip": [
                                    {
                                        "field": "time_ms",
                                        "type": "quantitative",
                                        "title": "Time (ms)",
                                        "format": ".2f",
                                    },
                                    {
                                        "field": "pdf",
                                        "type": "quantitative",
                                        "title": "Probability Density (1/ms)",
                                        "format": ".5f",
                                    },
                                ],
                            },
                        },
                    ],
                    "resolve": {"scale": {"y": "shared"}},
                    "config": {
                        "axis": {"labelColor": "#cbd5e1", "titleColor": "#e2e8f0"},
                        "legend": {"labelColor": "#cbd5e1"},
                    },
                },
                width="stretch",
            )
        with graph_col2:
            st.markdown("**Cumulative Distribution Function (CDF)**")
            st.caption(r"Fitted model: $F(x)=1-e^{-\hat{\lambda}x}$.")
            st.vega_lite_chart(
                curve_df,
                {
                    "width": "container",
                    "height": 320,
                    "mark": {"type": "line", "color": "#69a8ff", "strokeWidth": 3},
                    "encoding": {
                        "x": {
                            "field": "time_ms",
                            "type": "quantitative",
                            "title": "Response Time (ms)",
                        },
                        "y": {
                            "field": "cdf",
                            "type": "quantitative",
                            "title": "Cumulative Probability",
                            "scale": {"domain": [0, 1]},
                            "axis": {"format": ".0%"},
                        },
                        "tooltip": [
                            {
                                "field": "time_ms",
                                "type": "quantitative",
                                "title": "Time (ms)",
                                "format": ".2f",
                            },
                            {
                                "field": "cdf",
                                "type": "quantitative",
                                "title": "Probability",
                                "format": ".3%",
                            },
                        ],
                    },
                    "config": {
                        "axis": {"labelColor": "#cbd5e1", "titleColor": "#e2e8f0"}
                    },
                },
                width="stretch",
            )

        st.subheader("Observed Data vs. Theoretical Distribution")
        st.caption(
            r"Observed values come from the validated dataset; the theoretical "
            r"curve comes from the exponential distribution with rate "
            r"$\hat{\lambda}$ estimated from those values."
        )
        st.markdown("**Observed Probability vs. Theoretical Probability**")
        st.caption(
            "The observed curve is the fraction of data at or below each response "
            "time; the theoretical curve is the fitted model's cumulative probability."
        )
        sorted_observations = (
            cleaned_data[response_column].sort_values().reset_index(drop=True)
        )
        ecdf_df = compute_ecdf_comparison(sorted_observations, estimated_rate)
        cdf_comparison_data = pd.concat(
            [
                ecdf_df[["time_ms", "Empirical CDF"]]
                .rename(columns={"Empirical CDF": "probability"})
                .assign(series="Observed probability"),
                ecdf_df[["time_ms", "Theoretical CDF"]]
                .rename(columns={"Theoretical CDF": "probability"})
                .assign(series="Theoretical probability"),
            ],
            ignore_index=True,
        )
        st.vega_lite_chart(
            cdf_comparison_data,
            {
                "width": "container",
                "height": 320,
                "mark": {"type": "line", "strokeWidth": 2.5},
                "encoding": {
                    "x": {
                        "field": "time_ms",
                        "type": "quantitative",
                        "title": "Response Time (ms)",
                    },
                    "y": {
                        "field": "probability",
                        "type": "quantitative",
                        "title": "Cumulative Probability",
                        "scale": {"domain": [0, 1]},
                        "axis": {"format": ".0%"},
                    },
                    "color": {
                        "field": "series",
                        "type": "nominal",
                        "scale": {
                            "domain": ["Observed probability", "Theoretical probability"],
                            "range": ["#69a8ff", "#f2a65a"],
                        },
                        "legend": {"title": None},
                    },
                    "tooltip": [
                        {
                            "field": "series",
                            "type": "nominal",
                            "title": "Curve",
                        },
                        {
                            "field": "time_ms",
                            "type": "quantitative",
                            "title": "Time (ms)",
                            "format": ".2f",
                        },
                        {
                            "field": "probability",
                            "type": "quantitative",
                            "title": "Probability",
                            "format": ".3%",
                        },
                    ],
                },
                "config": {
                    "axis": {"labelColor": "#cbd5e1", "titleColor": "#e2e8f0"},
                    "legend": {"labelColor": "#cbd5e1"},
                },
            },
            width="stretch",
        )

        st.markdown("**Observed Data vs. Theoretical Distribution · Summary**")
        comparison_metrics = calculate_distribution_comparison_metrics(
            cleaned_data[response_column], estimated_rate
        )
        st.dataframe(comparison_metrics.to_frame(), width="stretch", hide_index=True)

        percentiles_df = calculate_percentiles(
            cleaned_data[response_column],
            estimated_rate,
        )
        default_sla_thresh = float(
            round(
                percentiles_df.loc[
                    percentiles_df["Percentile"] == "P95",
                    "Observed_ms",
                ].iloc[0],
                2,
            )
        )
        sla_result = evaluate_sla_compliance(
            cleaned_data[response_column],
            sla_threshold_ms=max(default_sla_thresh, 0.000001),
            target_compliance_percentage=95.0,
        )
        group_df = None
        selected_group_col = None

        json_report_str = generate_json_report(
            source_kind=source_kind,
            response_column=response_column,
            validation_report=report,
            descriptive_stats=statistics,
            estimated_rate=estimated_rate,
            estimated_mean=estimated_mean,
            percentiles_df=percentiles_df,
            sla_result=sla_result,
            comparison_metrics=comparison_metrics,
            group_comparison_df=group_df,
            group_column=selected_group_col,
        )
        md_report_str = generate_markdown_report(
            source_kind=source_kind,
            response_column=response_column,
            validation_report=report,
            descriptive_stats=statistics,
            estimated_rate=estimated_rate,
            estimated_mean=estimated_mean,
            percentiles_df=percentiles_df,
            sla_result=sla_result,
            comparison_metrics=comparison_metrics,
            group_comparison_df=group_df,
            group_column=selected_group_col,
        )
        summary_csv_str = generate_summary_csv(
            descriptive_stats=statistics,
            estimated_rate=estimated_rate,
            estimated_mean=estimated_mean,
            percentiles_df=percentiles_df,
            sla_result=sla_result,
            comparison_metrics=comparison_metrics,
        )


st.divider()
st.subheader("Export & Download Results")
dl_col1, dl_col2, dl_col3, dl_col4 = st.columns(4)
with dl_col1:
    st.download_button(
        "Download cleaned CSV",
        cleaned_data.to_csv(index=False).encode("utf-8"),
        "cleaned_response_times.csv",
        "text/csv",
        disabled=cleaned_data.empty,
        key="download_cleaned_csv",
    )
with dl_col2:
    st.download_button(
        "Download Analysis Report (JSON)",
        json_report_str.encode("utf-8") if not cleaned_data.empty and "json_report_str" in locals() else b"",
        "response_time_analysis.json",
        "application/json",
        disabled=cleaned_data.empty,
        key="download_json_report",
    )
with dl_col3:
    st.download_button(
        "Download Executive Report (MD)",
        md_report_str.encode("utf-8") if not cleaned_data.empty and "md_report_str" in locals() else b"",
        "response_time_report.md",
        "text/markdown",
        disabled=cleaned_data.empty,
        key="download_md_report",
    )
with dl_col4:
    st.download_button(
        "Download Metrics Summary (CSV)",
        summary_csv_str.encode("utf-8") if not cleaned_data.empty and "summary_csv_str" in locals() else b"",
        "performance_metrics_summary.csv",
        "text/csv",
        disabled=cleaned_data.empty,
        key="download_summary_csv",
    )
