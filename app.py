"""Streamlit dashboard for classifying customer complaint themes."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from src.api_client import api_results_to_dataframe, fetch_complaints_from_api
from src.config import (
    DEFAULT_TARGET_COLUMN,
    MAX_ROWS_FOR_TRAINING,
    MIN_TEXT_LENGTH,
    MODEL_PATH,
    OPTIONAL_TARGET_COLUMNS,
    TEST_SIZE,
    TEXT_COLUMN,
)
from src.data_loader import limit_rows, load_csv, validate_required_columns
from src.evaluation import create_confusion_matrix_figure, evaluate_model, get_top_categories
from src.model import (
    load_model,
    predict_theme_with_confidence,
    predict_themes_with_confidence,
    save_model,
    train_model,
)
from src.preprocessing import clean_text, prepare_training_data

logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="Customer Feedback Theme Classifier",
    page_icon="📊",
    layout="wide",
)

st.title("Customer Feedback Theme Classifier")
st.write(
    "This app uses public consumer complaint data to train a machine learning model "
    "that classifies customer complaint narratives into product or issue categories."
)
st.info(
    "This project uses public CFPB complaint data for educational analysis. "
    "Complaint narratives are consumer-submitted, not independently verified, "
    "and should be interpreted as exploratory feedback signals."
)

if "model_pipeline" not in st.session_state:
    st.session_state.model_pipeline = None
if "evaluation_results" not in st.session_state:
    st.session_state.evaluation_results = None
if "batch_prediction_results" not in st.session_state:
    st.session_state.batch_prediction_results = None
if "model_metadata" not in st.session_state:
    st.session_state.model_metadata = {}

st.sidebar.header("Controls")
target_column = st.sidebar.selectbox(
    "Target column",
    OPTIONAL_TARGET_COLUMNS,
    index=OPTIONAL_TARGET_COLUMNS.index(DEFAULT_TARGET_COLUMN),
)
max_rows = st.sidebar.number_input(
    "Maximum rows to train on",
    min_value=100,
    max_value=MAX_ROWS_FOR_TRAINING,
    value=5000,
    step=500,
)
test_size = st.sidebar.slider(
    "Test size",
    min_value=0.1,
    max_value=0.4,
    value=float(TEST_SIZE),
    step=0.05,
)
save_trained_model = st.sidebar.checkbox("Save trained model", value=True)
use_api = st.sidebar.checkbox("Fetch sample data from CFPB API")
train_requested = st.sidebar.button("Train Model", type="primary")

with st.sidebar.expander("Saved model"):
    model_file = Path(MODEL_PATH)
    if model_file.exists():
        st.write(f"Model file found: {MODEL_PATH}")
        if st.button("Load Saved Model"):
            try:
                loaded_model, loaded_metadata = load_model(MODEL_PATH)
                st.session_state.model_pipeline = loaded_model
                st.session_state.model_metadata = loaded_metadata
                st.session_state.evaluation_results = None
                st.session_state.batch_prediction_results = None
                st.success("Saved model loaded.")
            except Exception as error:
                st.error(f"Could not load saved model: {error}")
    else:
        st.write("No saved model file yet.")

    model_metadata = st.session_state.model_metadata
    if model_metadata:
        st.write(f"Model: {model_metadata.get('model_type', 'Unknown')}")
        if "target_column" in model_metadata:
            st.write(f"Target: {model_metadata['target_column']}")
        if "training_rows" in model_metadata:
            st.write(f"Training rows: {model_metadata['training_rows']:,}")
        if "accuracy" in model_metadata:
            st.write(f"Accuracy: {model_metadata['accuracy']:.2%}")
        if "trained_at_utc" in model_metadata:
            st.write(f"Trained: {model_metadata['trained_at_utc']}")


def render_classification_tools() -> None:
    """Render single and batch classification controls for a loaded model."""
    if st.session_state.model_pipeline is None:
        return

    st.subheader("Classify New Feedback")
    user_input = st.text_area("Paste a customer complaint or feedback message")
    if st.button("Classify Feedback"):
        if not user_input.strip():
            st.error("Please enter a complaint or feedback message.")
        else:
            cleaned_input = clean_text(user_input)
            prediction_result = predict_theme_with_confidence(
                st.session_state.model_pipeline,
                cleaned_input,
            )
            predicted_theme = prediction_result["prediction"]
            st.success(f"Predicted category: {predicted_theme}")

            confidence_scores = prediction_result.get("confidence_scores", {})
            if confidence_scores:
                st.write("Top confidence scores")
                top_predictions = sorted(
                    confidence_scores.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )[:5]
                for label, probability in top_predictions:
                    st.write(f"{label}: {probability:.2%}")

    st.subheader("Batch Classify Feedback")
    batch_input = st.text_area(
        "Paste one feedback message per line",
        key="batch_feedback_input",
    )
    if st.button("Create Prediction File"):
        feedback_messages = [
            message.strip()
            for message in batch_input.splitlines()
            if message.strip()
        ]
        if not feedback_messages:
            st.error("Enter at least one feedback message.")
        else:
            cleaned_feedback = [clean_text(message) for message in feedback_messages]
            batch_results = predict_themes_with_confidence(
                st.session_state.model_pipeline,
                cleaned_feedback,
            )
            batch_results["Feedback"] = feedback_messages
            st.session_state.batch_prediction_results = batch_results

    batch_prediction_results = st.session_state.batch_prediction_results
    if batch_prediction_results is not None:
        st.dataframe(batch_prediction_results, use_container_width=True)
        st.download_button(
            "Download Predictions CSV",
            data=batch_prediction_results.to_csv(index=False).encode("utf-8"),
            file_name="feedback_theme_predictions.csv",
            mime="text/csv",
        )


uploaded_file = st.file_uploader("Upload CFPB complaint CSV", type=["csv"])

complaints_df: pd.DataFrame | None = None

if use_api:
    try:
        with st.spinner("Fetching sample CFPB records..."):
            api_response = fetch_complaints_from_api(size=100)
            complaints_df = api_results_to_dataframe(api_response)
        st.success(f"Fetched {len(complaints_df):,} sample records from the CFPB API.")
    except RuntimeError as error:
        st.error(str(error))

if uploaded_file is not None:
    try:
        complaints_df = load_csv(uploaded_file)
        required_columns = [TEXT_COLUMN, target_column]
        validate_required_columns(complaints_df, required_columns)
    except ValueError as error:
        st.error(str(error))
        complaints_df = None

if complaints_df is not None and not complaints_df.empty:
    st.subheader("Dataset Preview")
    st.write(f"Rows loaded: {len(complaints_df):,}")
    st.dataframe(complaints_df.head(20), use_container_width=True)

    if TEXT_COLUMN in complaints_df.columns:
        missing_narratives = complaints_df[TEXT_COLUMN].isna().sum()
        st.write(f"Missing complaint narratives: {missing_narratives:,}")

    required_columns = [TEXT_COLUMN, target_column]
    try:
        validate_required_columns(complaints_df, required_columns)
    except ValueError as error:
        st.error(str(error))
    else:
        limited_df = limit_rows(complaints_df, int(max_rows))
        clean_df = prepare_training_data(
            limited_df,
            text_column=TEXT_COLUMN,
            target_column=target_column,
            min_text_length=MIN_TEXT_LENGTH,
        )

        metric_columns = st.columns(3)
        metric_columns[0].metric("Rows before cleaning", f"{len(limited_df):,}")
        metric_columns[1].metric("Rows after cleaning", f"{len(clean_df):,}")
        metric_columns[2].metric("Target categories", f"{clean_df[target_column].nunique():,}")

        if len(clean_df) < 100:
            st.warning("Fewer than 100 usable rows remain. Model results may be unstable.")

        if train_requested:
            try:
                with st.spinner("Training model..."):
                    model_pipeline, y_test, predictions, labels = train_model(
                        clean_df,
                        text_column=TEXT_COLUMN,
                        target_column=target_column,
                        test_size=float(test_size),
                    )
                    evaluation_results = evaluate_model(y_test, predictions, labels)
                    model_metadata = {
                        "model_type": "TF-IDF + Logistic Regression",
                        "target_column": target_column,
                        "training_rows": int(len(clean_df)),
                        "accuracy": float(evaluation_results["accuracy"]),
                        "trained_at_utc": datetime.now(timezone.utc).strftime(
                            "%Y-%m-%d %H:%M UTC"
                        ),
                    }
                    st.session_state.model_pipeline = model_pipeline
                    st.session_state.evaluation_results = evaluation_results
                    st.session_state.batch_prediction_results = None
                    st.session_state.model_metadata = model_metadata

                    if save_trained_model:
                        save_model(model_pipeline, MODEL_PATH, model_metadata)

                st.success("Model training complete.")
                if save_trained_model:
                    st.write(f"Saved model to `{MODEL_PATH}`.")
            except ValueError as error:
                st.error(f"Training error: {error}")
            except Exception as error:
                st.error(f"Unexpected model training error: {error}")

        st.subheader("Model Performance")
        if st.session_state.evaluation_results:
            evaluation_results = st.session_state.evaluation_results
            st.metric("Accuracy", f"{evaluation_results['accuracy']:.2%}")
            st.dataframe(
                evaluation_results["classification_report_df"],
                use_container_width=True,
            )
            st.subheader("Confusion Matrix")
            confusion_matrix_figure = create_confusion_matrix_figure(
                evaluation_results["confusion_matrix"],
                evaluation_results["labels"],
            )
            st.pyplot(confusion_matrix_figure, use_container_width=True)
        else:
            st.caption("Train a model to see accuracy and classification metrics.")

        render_classification_tools()

        st.subheader("Business Insights")
        chart_columns = [
            "Product",
            "Issue",
            "State",
            "Company",
            "Submitted via",
            "Company response to consumer",
        ]

        for column in chart_columns:
            if column in clean_df.columns:
                top_categories = get_top_categories(clean_df, column)
                if not top_categories.empty:
                    chart_df = top_categories.set_index(column)
                    st.write(f"Top {column}")
                    st.bar_chart(chart_df)
else:
    st.write("Upload a CFPB complaint CSV to begin.")
    render_classification_tools()

with st.sidebar.expander("Data source"):
    st.write("Download public data from the CFPB Consumer Complaint Database.")
    st.write("CSV upload is the primary workflow. API sample fetch is optional.")
