"""Member 1 Streamlit interface for response-time data management."""

from __future__ import annotations

import streamlit as st

from data_management import CSVDataError, generate_sample_response_times, load_csv, validate_and_clean_response_times

st.set_page_config(page_title="Server Response Time Modeling", page_icon=":material/timer:", layout="wide")
st.title("Server Response Time Modeling Using Exponential Distribution")
st.caption("load, generate, validate, clean, and export response-time data.")

source_kind = st.radio("Data source", ("Upload CSV", "Generate sample"), horizontal=True)
try:
    if source_kind == "Upload CSV":
        uploaded = st.file_uploader("Upload a CSV file", type="csv")
        if uploaded is None:
            st.info("Upload a CSV file to begin.")
            st.stop()
        raw_data = load_csv(uploaded)
    else:
        count = st.number_input("Number of observations", min_value=1, value=100, step=1)
        rate = st.number_input("Exponential rate (1/ms)", min_value=0.000001, value=0.02, format="%.6f")
        seed = st.number_input("Random seed", min_value=0, value=42, step=1)
        raw_data = generate_sample_response_times(int(count), float(rate), int(seed))
except (CSVDataError, ValueError) as error:
    st.error(str(error))
    st.stop()

st.session_state["member1_raw_data"] = raw_data
st.session_state["member1_source_kind"] = source_kind
st.subheader("Uploaded or generated dataset")
st.dataframe(raw_data, use_container_width=True, hide_index=True)
response_column = st.selectbox("Response-time column (milliseconds)", list(raw_data.columns))
cleaned_data, report = validate_and_clean_response_times(raw_data, response_column)
st.session_state["member1_cleaned_data"] = cleaned_data

st.subheader("Validation report")
st.write(f"Valid rows: {report.valid_rows} of {report.total_rows}; excluded rows: {report.invalid_rows}.")
st.dataframe(report.to_frame(), use_container_width=True, hide_index=True)
st.subheader("Cleaned dataset")
st.dataframe(cleaned_data, use_container_width=True, hide_index=True)
if cleaned_data.empty:
    st.warning("No valid response-time observations remain. Select another column or correct the input data.")
st.download_button("Download cleaned CSV", cleaned_data.to_csv(index=False).encode("utf-8"), "cleaned_response_times.csv", "text/csv", disabled=cleaned_data.empty)
