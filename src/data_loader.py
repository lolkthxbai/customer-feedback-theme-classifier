"""Utilities for loading and validating complaint data."""

from __future__ import annotations

import logging
from typing import BinaryIO

import pandas as pd

from src.config import RANDOM_STATE

logger = logging.getLogger(__name__)


def load_csv(file: BinaryIO) -> pd.DataFrame:
    """
    Load uploaded CSV file into a pandas DataFrame.
    Raise a clear error if the file cannot be loaded.
    """
    try:
        complaints_df = pd.read_csv(file)
    except pd.errors.EmptyDataError as error:
        raise ValueError("The uploaded file is empty.") from error
    except pd.errors.ParserError as error:
        raise ValueError("The uploaded file could not be parsed as a valid CSV.") from error
    except UnicodeDecodeError as error:
        raise ValueError("The uploaded file encoding could not be read. Try saving it as UTF-8 CSV.") from error
    except Exception as error:
        raise ValueError(f"Unexpected error while loading file: {error}") from error

    logger.info("Loaded CSV with %s rows and %s columns", len(complaints_df), len(complaints_df.columns))
    return complaints_df


def validate_required_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    """
    Check that required columns exist in the DataFrame.
    Raise ValueError if any required columns are missing.
    """
    missing_columns: list[str] = []

    for column in required_columns:
        if column not in df.columns:
            missing_columns.append(column)

    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Missing required column(s): {missing_text}")


def limit_rows(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    """
    Limit the dataset size for faster training.
    Use random sampling with a fixed random state if the dataset is larger than max_rows.
    """
    if max_rows <= 0:
        raise ValueError("Maximum rows must be greater than zero.")

    if len(df) <= max_rows:
        return df.copy()

    logger.info("Sampling %s rows from %s available rows", max_rows, len(df))
    return df.sample(n=max_rows, random_state=RANDOM_STATE).reset_index(drop=True)
