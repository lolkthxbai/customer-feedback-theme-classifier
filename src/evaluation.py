"""Evaluation and insight helpers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def evaluate_model(y_test, predictions, labels: list[str] | None = None) -> dict:
    """
    Return accuracy score, classification report, and confusion matrix.
    """
    evaluation_labels = labels or sorted(set(y_test).union(predictions))
    report_dict = classification_report(
        y_test,
        predictions,
        labels=evaluation_labels,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "classification_report": report_dict,
        "classification_report_df": pd.DataFrame(report_dict).transpose(),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=evaluation_labels),
        "labels": evaluation_labels,
    }


def create_confusion_matrix_figure(confusion_matrix_data, labels: list[str]):
    """
    Create a labeled Matplotlib confusion matrix figure.
    """
    if not labels:
        raise ValueError("At least one category label is required for a confusion matrix.")

    chart_size = min(max(7, len(labels) * 1.1), 16)
    figure, axis = plt.subplots(figsize=(chart_size, chart_size * 0.8))
    image = axis.imshow(confusion_matrix_data, cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    axis.set(
        xticks=range(len(labels)),
        yticks=range(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted category",
        ylabel="Actual category",
        title="Confusion Matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    maximum_count = confusion_matrix_data.max() if confusion_matrix_data.size else 0
    threshold = maximum_count / 2
    for row_index, row in enumerate(confusion_matrix_data):
        for column_index, value in enumerate(row):
            axis.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )

    figure.tight_layout()
    return figure


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
