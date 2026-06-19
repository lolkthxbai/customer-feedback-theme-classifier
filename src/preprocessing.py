"""Text cleaning and training-data preparation."""

from __future__ import annotations

import re

import pandas as pd


def clean_text(text: str) -> str:
    """
    Clean one text string.
    Lowercase the text, strip whitespace, replace line breaks, and remove extra spaces.
    """
    if pd.isna(text):
        return ""

    clean_value = str(text).lower().replace("\n", " ").replace("\r", " ").strip()
    return re.sub(r"\s+", " ", clean_value)


def prepare_training_data(
    df: pd.DataFrame,
    text_column: str,
    target_column: str,
    min_text_length: int,
) -> pd.DataFrame:
    """
    Remove rows with missing text or missing target labels.
    Clean the text column.
    Remove very short complaint narratives.
    Return a cleaned DataFrame.
    """
    cleaned_df = df.copy()

    cleaned_df[text_column] = cleaned_df[text_column].apply(clean_text)
    cleaned_df[target_column] = cleaned_df[target_column].astype("string").str.strip()

    has_text = cleaned_df[text_column].str.len() >= min_text_length
    has_target = cleaned_df[target_column].notna() & (cleaned_df[target_column] != "")

    cleaned_df = cleaned_df.loc[has_text & has_target].reset_index(drop=True)
    return cleaned_df
