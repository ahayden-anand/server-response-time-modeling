
"""Professional Streamlit dashboard for Member 1: Data Management."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from data_management import (
    CSVDataError,
    generate_sample_response_times,
    load_csv,
    validate_and_clean_response_times,
)

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Response Time Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# Styling
# --------------------------------------------------
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
        max-width: 1450px;
    }

    .hero {
        padding: 1.8rem 2rem;
        border-radius: 18px;
        background: linear-gradient(120deg, #102a43, #176b87);
        color: white;
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        color: white;
        font-size: 2rem;
        margin-bottom: 0.45rem;
    }

    .hero p {
        color: #e0f2fe;
        font-size: 1rem;
        margin-bottom: 0;
    }

    .section-label {
        color: #2680a5;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.3rem;
    }

    div[data-testid="stMetric"] {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
        padding: 1rem;
        border-radius: 12px;
    }

    div[data-testid="stMetricLabel"] {
        font-weight: 600;
    }

    .footer {
        text-align: center;
        color: #8492a6;
        font-size: 0.85rem;
        padding-top: 1.5rem;
    }

    div.stDownloadButton > button {
        width: 100%;
        border-radius: 10px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div style="font-size:0.8rem; letter-spacing:0.14em;
                    font-weight:700; margin-bottom:0.6rem;">
            DATA MANAGEMENT MODULE · MEMBER 01
        </div>
        <h1>Server Response Time Modeling</h1>
        <p>
            Upload datasets, generate sample observations, validate input
            quality, and export clean response-time data.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Sidebar controls
# --------------------------------------------------
with st.sidebar:
    st.title("📊 Data Workspace")
    st.caption("Member 1 · Data Management")

    st.divider()

    source_kind = st.radio(
        "Choose data source",
        ("Upload CSV", "Generate sample"),
        help="Upload your own dataset or create simulated response-time data.",
    )

    raw_data = None

    if source_kind == "Upload CSV":
        uploaded = st.file_uploader(
            "Upload response-time CSV",
            type=["csv"],
            help="Choose a CSV file containing a response-time column.",
        )

        if uploaded is not None:
            try:
                raw_data = load_csv(uploaded)
            except CSVDataError as error:
                st.error(str(error))

    else:
        st.markdown("#### Sample generator")

        count = st.number_input(
            "Number of observations",
            min_value=1,
            max_value=100000,
            value=100,
            step=10,
        )

        rate = st.number_input(
            "Exponential rate λ (1/ms)",
            min_value=0.000001,
            max_value=1000.0,
            value=0.02,
            format="%.6f",
            help="The rate is measured in inverse milliseconds.",
        )

        seed = st.number_input(
            "Random seed",
            min_value=0,
            max_value=2147483647,
            value=42,
            step=1,
        )

        if st.button("Generate dataset", type="primary", use_container_width=True):
            try:
                st.session_state["member1_generated_data"] = (
                    generate_sample_response_times(
                        int(count),
                        float(rate),
                        int(seed),
                    )
                )
            except ValueError as error:
                st.error(str(error))

        raw_data = st.session_state.get("member1_generated_data")

    st.divider()
    st.caption("Response-time units: milliseconds (ms)")
    st.caption("Invalid observations are excluded from cleaned data.")

# --------------------------------------------------
# Empty state
# --------------------------------------------------
if raw_data is None:
    st.info(
        "👈 **Start by choosing a data source.** Upload a CSV file or "
        "generate sample observations from the sidebar."
    )

    st.markdown("### What you can do")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 📁 Upload")
        st.write("Load and preview an existing response-time dataset.")

    with col2:
        st.markdown("#### 🧹 Validate")
        st.write("Identify missing, invalid, zero, negative, and infinite values.")

    with col3:
        st.markdown("#### 📥 Export")
        st.write("Download the cleaned dataset for further analysis.")

    st.stop()

# --------------------------------------------------
# Data preview and validation
# --------------------------------------------------
st.markdown('<div class="section-label">Dataset overview</div>', unsafe_allow_html=True)
st.subheader("Your response-time data")

st.caption(
    f"Source: {source_kind} · "
    f"{len(raw_data):,} original rows · "
    f"{len(raw_data.columns):,} columns"
)

with st.expander("Preview original dataset", expanded=False):
    st.dataframe(
        raw_data.head(100),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("Showing up to the first 100 rows.")

response_column = st.selectbox(
    "Select the response-time column",
    options=list(raw_data.columns),
    help="Choose the column containing response times measured in milliseconds.",
)

try:
    cleaned_data, report = validate_and_clean_response_times(
        raw_data,
        response_column,
    )
except (ValueError, TypeError) as error:
    st.error(f"Unable to validate this dataset: {error}")
    st.stop()

# --------------------------------------------------
# Summary metrics
# --------------------------------------------------
st.markdown('<div class="section-label">Data quality</div>', unsafe_allow_html=True)
st.subheader("Validation summary")

valid_rows = report.valid_rows
total_rows = report.total_rows
invalid_rows = report.invalid_rows
valid_percent = (valid_rows / total_rows * 100) if total_rows else 0

m1, m2, m3, m4 = st.columns(4)

m1.metric("Total rows", f"{total_rows:,}")
m2.metric("Valid rows", f"{valid_rows:,}")
m3.metric("Excluded rows", f"{invalid_rows:,}")
m4.metric("Data retained", f"{valid_percent:.1f}%")

if invalid_rows == 0:
    st.success("All rows contain valid, positive, finite response-time values.")
elif valid_rows == 0:
    st.error("No valid response times remain. Select another column or correct your dataset.")
else:
    st.warning(
        f"{invalid_rows:,} row(s) contain invalid response-time values "
        "and are excluded from the cleaned dataset."
    )

# --------------------------------------------------
# Tabs
# --------------------------------------------------
tab_clean, tab_issues, tab_info = st.tabs(
    ["🧹 Cleaned data", "🔎 Validation details", "ℹ️ Dataset information"]
)

with tab_clean:
    st.markdown("#### Cleaned response-time dataset")
    st.caption("Only rows with valid, positive, finite response times are retained.")

    st.dataframe(
        cleaned_data,
        use_container_width=True,
        hide_index=True,
        height=380,
    )

    st.download_button(
        label="⬇️ Download cleaned CSV",
        data=cleaned_data.to_csv(index=False).encode("utf-8"),
        file_name="cleaned_response_times.csv",
        mime="text/csv",
        disabled=cleaned_data.empty,
        use_container_width=True,
    )

with tab_issues:
    st.markdown("#### Validation issue breakdown")
    issue_table = report.to_frame()

    st.dataframe(
        issue_table,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "A single row may appear in multiple issue categories. "
        "The excluded-row total counts each invalid row only once."
    )

with tab_info:
    st.markdown("#### Dataset information")

    info1, info2 = st.columns(2)

    with info1:
        st.write("**Selected response-time column**")
        st.code(str(response_column))

        st.write("**Measurement unit**")
        st.write("Milliseconds (ms)")

    with info2:
        st.write("**Number of columns**")
        st.write(f"{len(raw_data.columns):,}")

        st.write("**Number of rows removed**")
        st.write(f"{invalid_rows:,}")

    st.write("**Columns in the dataset**")
    st.write(", ".join(str(column) for column in raw_data.columns))

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.divider()
st.markdown(
    """
    <div class="footer">
        Server Response Time Modeling Using Exponential Distribution
        <br>
        Member 1 — Data Management Module
    </div>
    """,
    unsafe_allow_html=True,
)
