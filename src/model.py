"""Model construction, training, persistence, and prediction helpers."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import (
    RANDOM_STATE,
    TEST_SIZE,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    TFIDF_STOP_WORDS,
)


def build_model_pipeline() -> Pipeline:
    """
    Build and return a scikit-learn Pipeline using TfidfVectorizer and LogisticRegression.
    """
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words=TFIDF_STOP_WORDS,
                    max_features=TFIDF_MAX_FEATURES,
                    ngram_range=TFIDF_NGRAM_RANGE,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def train_model(
    df: pd.DataFrame,
    text_column: str,
    target_column: str,
    test_size: float = TEST_SIZE,
) -> tuple[Pipeline, pd.Series, pd.Series, list[str]]:
    """
    Split the data, train the model, and return the trained pipeline, y_test, predictions, and class labels.
    """
    if df.empty:
        raise ValueError("No training rows are available after cleaning.")

    target_count = df[target_column].nunique()
    if target_count < 2:
        raise ValueError("The target column must contain at least two categories for training.")

    text_series = df[text_column]
    target_series = df[target_column]
    stratify_labels = target_series if target_series.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        text_series,
        target_series,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=stratify_labels,
    )

    model_pipeline = build_model_pipeline()
    model_pipeline.fit(X_train, y_train)
    predictions = model_pipeline.predict(X_test)
    class_labels = list(model_pipeline.named_steps["classifier"].classes_)

    return model_pipeline, y_test, pd.Series(predictions, index=y_test.index), class_labels


def save_model(model: Pipeline, model_path: str) -> None:
    """
    Save trained model pipeline using joblib.
    """
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(model_path: str) -> Pipeline:
    """
    Load saved model pipeline using joblib.
    """
    return joblib.load(model_path)


def predict_theme(model: Pipeline, text: str) -> str:
    """
    Predict the category/theme for a new complaint narrative.
    """
    return str(model.predict([text])[0])


def predict_theme_with_confidence(model: Pipeline, text: str) -> dict:
    """
    Predict category and return confidence scores if the classifier supports predict_proba.
    """
    predicted_theme = predict_theme(model, text)
    prediction_details: dict = {"prediction": predicted_theme, "confidence_scores": {}}

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([text])[0]
        class_labels = model.classes_
        prediction_details["confidence_scores"] = {
            str(label): float(probability)
            for label, probability in zip(class_labels, probabilities)
        }

    return prediction_details


def predict_themes_with_confidence(model: Pipeline, texts: list[str]) -> pd.DataFrame:
    """
    Predict categories for multiple feedback messages and return top confidence scores.
    """
    if not texts:
        raise ValueError("At least one feedback message is required for batch prediction.")

    predictions = model.predict(texts)
    prediction_results = pd.DataFrame(
        {
            "Feedback": texts,
            "Predicted category": [str(prediction) for prediction in predictions],
        }
    )

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(texts)
        prediction_results["Confidence"] = probabilities.max(axis=1)

    return prediction_results
