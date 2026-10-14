"""CSV loading, sample generation, and response-time data validation.

All response times in this module are measured in milliseconds (ms). When an
exponential rate is used, its unit is inverse milliseconds (1/ms).
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from os import PathLike
from typing import IO

import numpy as np
import pandas as pd


class CSVDataError(ValueError):
    """Raised when a CSV file cannot be used as a tabular dataset."""


@dataclass(frozen=True)
class ValidationReport:
    """Structured result of response-time validation.

    ``issue_rows`` uses the original DataFrame index labels.
    """

    response_column: str
    total_rows: int
    valid_rows: int
    issue_rows: dict[str, list[int]]

    @property
    def invalid_rows(self) -> int:
        """Return the number of rows excluded from the cleaned dataset."""
        return self.total_rows - self.valid_rows

    @property
    def issue_counts(self) -> dict[str, int]:
        """Return the number of observations in every validation category."""
        return {issue: len(rows) for issue, rows in self.issue_rows.items()}

    def to_frame(self) -> pd.DataFrame:
        """Return issue counts and original row indexes as a display table."""
        return pd.DataFrame(
            {
                "issue": list(self.issue_rows),
                "count": [len(rows) for rows in self.issue_rows.values()],
                "row_indexes": [", ".join(map(str, rows)) or "—" for rows in self.issue_rows.values()],
            }
        )


def load_csv(source: str | PathLike[str] | IO[bytes] | IO[str]) -> pd.DataFrame:
    """Read a non-empty CSV file and provide clear, user-facing errors.

    Automatic NA conversion is disabled so validation can distinguish blank
    values from the literal text ``NaN``.
    """
    try:
        frame = pd.read_csv(source, keep_default_na=False, on_bad_lines="error")
    except pd.errors.EmptyDataError as error:
        raise CSVDataError("The CSV file is empty. Add a header row and at least one data row.") from error
    except (pd.errors.ParserError, UnicodeDecodeError) as error:
        raise CSVDataError(f"The file is not a valid CSV: {error}") from error
    except (OSError, ValueError, TypeError) as error:
        raise CSVDataError(f"The CSV file could not be read: {error}") from error

    if frame.columns.empty:
        raise CSVDataError("The CSV file has no column headers.")
    if frame.empty:
        raise CSVDataError("The CSV file has headers but no data rows.")
    if any(not str(column).strip() for column in frame.columns):
        raise CSVDataError("Every CSV column must have a non-empty header.")
    return frame


def generate_sample_response_times(
    count: int = 100,
    rate_per_ms: float = 0.02,
    seed: int | None = 42,
    response_column: str = "response_time_ms",
) -> pd.DataFrame:
    """Generate reproducible exponentially distributed response times in ms.

    ``rate_per_ms`` is a rate in inverse milliseconds (1/ms), so the scale is
    ``1 / rate_per_ms`` milliseconds.
    """
    if isinstance(count, bool) or not isinstance(count, Integral) or count <= 0:
        raise ValueError("Sample count must be a positive integer.")
    try:
        rate = float(rate_per_ms)
    except (TypeError, ValueError) as error:
        raise ValueError("Rate must be a finite positive number in 1/ms.") from error
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("Rate must be a finite positive number in 1/ms.")
    if not isinstance(response_column, str) or not response_column.strip():
        raise ValueError("Response-time column name must be a non-empty string.")

    values = np.random.default_rng(seed).exponential(scale=1 / rate, size=count)
    return pd.DataFrame({response_column: values})


def validate_and_clean_response_times(
    frame: pd.DataFrame, response_column: str
) -> tuple[pd.DataFrame, ValidationReport]:
    """Keep finite, strictly positive observations and report rejected rows.

    Every original column is retained for accepted rows. The selected
    response-time column becomes numeric float data in milliseconds.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")
    if frame.empty:
        raise ValueError("The dataset has no rows to validate.")
    if response_column not in frame.columns:
        raise ValueError(f"Response-time column '{response_column}' was not found in the dataset.")

    source = frame[response_column]
    text = source.astype("string").str.strip()
    missing = source.isna() | text.eq("")
    nan_values = source.isna() | text.str.casefold().eq("nan")
    numeric = pd.to_numeric(source, errors="coerce")
    non_numeric = ~(missing | nan_values) & numeric.isna()
    infinite = numeric.notna() & ~np.isfinite(numeric)
    zero = numeric.eq(0)
    negative = numeric.lt(0)
    invalid = missing | nan_values | non_numeric | infinite | zero | negative

    issues = {
        "missing": frame.index[missing].tolist(),
        "non_numeric": frame.index[non_numeric].tolist(),
        "nan": frame.index[nan_values].tolist(),
        "infinite": frame.index[infinite].tolist(),
        "zero": frame.index[zero].tolist(),
        "negative": frame.index[negative].tolist(),
    }
    cleaned = frame.loc[~invalid].copy()
    cleaned[response_column] = numeric.loc[~invalid].astype(float)
    report = ValidationReport(response_column, len(frame), len(cleaned), issues)
    return cleaned.reset_index(drop=True), report
