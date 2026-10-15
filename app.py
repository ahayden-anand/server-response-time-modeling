"""Member 1 Streamlit interface for response-time data management."""

from __future__ import annotations

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
    format_interval_label,
    generate_distribution_curves,
    interval_probability,
)

st.set_page_config(
    page_title="Server Response Time Modeling",
    page_icon=":material/timer:",
    layout="wide",
)
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

st.subheader("Validation report")
st.write(
    f"Valid rows: {report.valid_rows} of {report.total_rows}; "
    f"excluded rows: {report.invalid_rows}."
)
st.dataframe(report.to_frame(), width="stretch", hide_index=True)

st.subheader("Cleaned dataset")
st.dataframe(cleaned_data, width="stretch", hide_index=True)

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
        st.subheader("Statistical analysis")
        first_row = st.columns(4)
        first_row[0].metric("Valid observations", statistics.count)
        first_row[1].metric("Mean response time (ms)", f"{statistics.mean:.6g}")
        first_row[2].metric("Median response time (ms)", f"{statistics.median:.6g}")
        first_row[3].metric("Minimum response time (ms)", f"{statistics.minimum:.6g}")

        second_row = st.columns(4)
        second_row[0].metric("Maximum response time (ms)", f"{statistics.maximum:.6g}")
        second_row[1].metric(
            "Population variance (ms²)",
            f"{statistics.population_variance:.6g}",
        )
        second_row[2].metric(
            "Population standard deviation (ms)",
            f"{statistics.population_standard_deviation:.6g}",
        )
        second_row[3].metric(
            "Sample variance (ms²)",
            (
                f"{statistics.sample_variance:.6g}"
                if statistics.sample_variance is not None
                else "Not defined (n < 2)"
            ),
        )

        third_row = st.columns(3)
        third_row[0].metric(
            "Sample standard deviation (ms)",
            (
                f"{statistics.sample_standard_deviation:.6g}"
                if statistics.sample_standard_deviation is not None
                else "Not defined (n < 2)"
            ),
        )
        third_row[1].metric("Estimated rate λ (1/ms)", f"{estimated_rate:.6g}")
        third_row[2].metric("Estimated mean 1/λ (ms)", f"{estimated_mean:.6g}")
        st.markdown(
            "For an exponential model, the maximum-likelihood rate estimate is "
            r"$\hat{\lambda}=1/\bar{x}$, where $\bar{x}$ is the mean of the "
            "validated observations above. The estimate therefore uses the "
            "uploaded or generated **cleaned dataset**, not the sample generator's "
            "configured rate."
        )

        st.divider()
        st.subheader("Probability Calculator")
        calc_col1, calc_col2 = st.columns([1, 2])

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

        with calc_col1:
            st.markdown("**Threshold Probability: $P(X \\le t)$ and $P(X > t)$**")
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
            m_col1.metric("P(X ≤ t)", f"{prob_le:.4f}", f"{prob_le * 100:.2f}%")
            m_col2.metric("P(X > t)", f"{prob_gt:.4f}", f"{prob_gt * 100:.2f}%")

        with calc_col2:
            st.markdown("**Interval Probability: $P(a \\le X \\le b)$**")
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
                interval_label = format_interval_label(lower_a, upper_b)
                st.metric(
                    interval_label,
                    f"{prob_ab:.4f}",
                    f"{prob_ab * 100:.2f}%",
                )
            else:
                st.error("Lower bound a must be less than or equal to upper bound b.")

        st.subheader("Interactive Distribution Graphs")
        hist_bins = st.slider(
            "Histogram bins",
            min_value=5,
            max_value=50,
            value=20,
            step=1,
            key="hist_bins_slider",
        )
        hist_df = compute_histogram_data(cleaned_data[response_column], bins=hist_bins)
        st.markdown("**Observed Response-Time Histogram**")
        st.bar_chart(hist_df, x="bin_label", y="count")

        max_curve_time = float(max(float(statistics.maximum) * 1.05, estimated_mean * 3.5))
        curve_df = generate_distribution_curves(
            estimated_rate, max_time=max_curve_time, points=200
        )
        graph_col1, graph_col2 = st.columns(2)
        with graph_col1:
            st.markdown("**Theoretical Exponential PDF ($f(t) = \\lambda e^{-\\lambda t}$)**")
            st.line_chart(curve_df, x="time_ms", y="pdf")
        with graph_col2:
            st.markdown("**Theoretical Exponential CDF ($F(t) = 1 - e^{-\\lambda t}$)**")
            st.line_chart(curve_df, x="time_ms", y="cdf")

        st.subheader("Observed vs. Theoretical Distribution Comparison")
        st.markdown("**Empirical CDF vs. Theoretical CDF**")
        sorted_observations = (
            cleaned_data[response_column].sort_values().reset_index(drop=True)
        )
        ecdf_df = compute_ecdf_comparison(sorted_observations, estimated_rate)
        st.line_chart(ecdf_df, x="time_ms", y=["Empirical CDF", "Theoretical CDF"])

        st.markdown("**Distribution Comparison Metrics**")
        comparison_metrics = calculate_distribution_comparison_metrics(
            cleaned_data[response_column], estimated_rate
        )
        st.dataframe(comparison_metrics.to_frame(), width="stretch", hide_index=True)


st.download_button(
    "Download cleaned CSV",
    cleaned_data.to_csv(index=False).encode("utf-8"),
    "cleaned_response_times.csv",
    "text/csv",
    disabled=cleaned_data.empty,
)
