"""Streamlit dashboard for classifying customer complaint themes."""

from __future__ import annotations

import logging
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
from src.model import predict_theme_with_confidence, save_model, train_model
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
                    st.session_state.model_pipeline = model_pipeline
                    st.session_state.evaluation_results = evaluation_results

                    if save_trained_model:
                        save_model(model_pipeline, MODEL_PATH)

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

        st.subheader("Classify New Feedback")
        user_input = st.text_area("Paste a customer complaint or feedback message")
        if st.button("Classify Feedback"):
            if not user_input.strip():
                st.error("Please enter a complaint or feedback message.")
            elif st.session_state.model_pipeline is None:
                st.error("Please train a model before classifying new feedback.")
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

with st.sidebar.expander("Data source"):
    st.write("Download public data from the CFPB Consumer Complaint Database.")
    st.write("CSV upload is the primary workflow. API sample fetch is optional.")

with st.sidebar.expander("Saved model"):
    model_file = Path(MODEL_PATH)
    if model_file.exists():
        st.write(f"Model file found: `{MODEL_PATH}`")
    else:
        st.write("No saved model file yet.")
