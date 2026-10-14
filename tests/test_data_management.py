from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_management import CSVDataError, generate_sample_response_times, load_csv, validate_and_clean_response_times


def test_load_csv_rejects_empty_and_header_only_files(tmp_path: Path) -> None:
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    with pytest.raises(CSVDataError, match="empty"):
        load_csv(empty)
    header_only = tmp_path / "headers.csv"
    header_only.write_text("response_time_ms,server\n")
    with pytest.raises(CSVDataError, match="no data rows"):
        load_csv(header_only)


def test_load_csv_reads_valid_data_and_rejects_malformed_content(tmp_path: Path) -> None:
    valid = tmp_path / "valid.csv"
    valid.write_text("response_time_ms,server\n12.5,api-east\n")
    assert load_csv(valid).to_dict("list") == {"response_time_ms": [12.5], "server": ["api-east"]}

    malformed = tmp_path / "malformed.csv"
    malformed.write_text('response_time_ms,server\n"12.5,api-east\n')
    with pytest.raises(CSVDataError, match="not a valid CSV"):
        load_csv(malformed)


def test_validation_reports_each_invalid_category_and_preserves_columns() -> None:
    frame = pd.DataFrame({"response_ms": [10, "", "not-a-number", -1, 0, "NaN", np.inf, 25], "server": list("abcdefgh")})
    cleaned, report = validate_and_clean_response_times(frame, "response_ms")
    assert cleaned.to_dict("list") == {"response_ms": [10.0, 25.0], "server": ["a", "h"]}
    assert report.valid_rows == 2
    assert report.invalid_rows == 6
    assert report.issue_counts == {"missing": 1, "non_numeric": 1, "nan": 1, "infinite": 1, "zero": 1, "negative": 1}


def test_validation_reports_native_nan_and_requires_selected_column() -> None:
    frame = pd.DataFrame({"latency_ms": [np.nan, 12], "server": ["east", "west"]})
    cleaned, report = validate_and_clean_response_times(frame, "latency_ms")
    assert len(cleaned) == 1
    assert report.issue_counts["missing"] == 1
    assert report.issue_counts["nan"] == 1
    with pytest.raises(ValueError, match="not found"):
        validate_and_clean_response_times(frame, "response_time_ms")


def test_generated_samples_are_reproducible_and_positive() -> None:
    first = generate_sample_response_times(count=4, rate_per_ms=0.5, seed=7)
    second = generate_sample_response_times(count=4, rate_per_ms=0.5, seed=7)
    pd.testing.assert_frame_equal(first, second)
    assert (first["response_time_ms"] > 0).all()


@pytest.mark.parametrize("count, rate", [(0, 1), (-2, 1), (3, 0), (3, -1), (3, np.inf)])
def test_generator_rejects_invalid_parameters(count: int, rate: float) -> None:
    with pytest.raises(ValueError):
        generate_sample_response_times(count=count, rate_per_ms=rate)
