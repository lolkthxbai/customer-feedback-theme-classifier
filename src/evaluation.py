"""Evaluation and insight helpers."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def evaluate_model(y_test, predictions) -> dict:
    """
    Return accuracy score and classification report.
    """
    report_dict = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "classification_report": report_dict,
        "classification_report_df": pd.DataFrame(report_dict).transpose(),
        "confusion_matrix": confusion_matrix(y_test, predictions),
    }


def get_top_categories(df: pd.DataFrame, column: str, top_n: int = 10) -> pd.DataFrame:
    """
    Return the top category counts for charts.
    """
    if column not in df.columns:
        return pd.DataFrame(columns=[column, "count"])

    category_counts = (
        df[column]
        .dropna()
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .value_counts()
        .head(top_n)
        .rename_axis(column)
        .reset_index(name="count")
    )
    return category_counts
