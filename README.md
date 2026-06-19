# Customer Feedback Theme Classifier

Customer Feedback Theme Classifier is a Python and Streamlit machine learning app that classifies public consumer complaint narratives into business-relevant themes. It uses CFPB consumer complaint data, cleans narrative text, trains a TF-IDF and Logistic Regression classifier, and turns complaint records into model metrics and business insight charts.

## Project Snapshot

- Built with public Consumer Financial Protection Bureau complaint data.
- Tested with a CFPB CSV containing 19,191 complaint records.
- Trained a scikit-learn text classification pipeline.
- Predicted complaint categories such as `Product` or `Issue`.
- Achieved about 73.8% accuracy during local testing with `Product` as the target label.
- Packaged as a GitHub-ready portfolio project with modular Python code and a Streamlit dashboard.

## Why This Project Matters

Customer feedback often arrives as messy, unstructured text. Support teams, product teams, and operations leaders need a way to identify recurring themes without reading every complaint manually.

This project demonstrates how basic natural language processing can help convert customer narratives into structured categories, making it easier to spot common problems, prioritize fixes, and communicate business trends.

## Dataset

This app is designed for the public [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/).

The expected CSV columns include:

- `Consumer complaint narrative`
- `Product`
- `Issue`
- `Sub-product`
- `Sub-issue`
- `Company`
- `State`
- `Submitted via`
- `Company response to consumer`
- `Timely response?`
- `Date received`

The project does not include downloaded complaint data in the repository. CSV files should be downloaded locally from CFPB and uploaded through the Streamlit app.

### Try It With Sample Data

For a quick local smoke test, upload [`data/sample_complaints.csv`](data/sample_complaints.csv). It contains synthetic complaint-style records only; it is included to confirm the upload, training, prediction, and chart workflows work end to end.

The sample is intentionally small. Its accuracy and category counts are not meaningful model results. Use a public CFPB export for analysis.

## Features

- Upload a CFPB complaint CSV file.
- Validate that required columns exist.
- Remove missing or blank complaint narratives.
- Clean narrative text by lowercasing, trimming whitespace, and removing extra line breaks.
- Limit training rows for faster experimentation.
- Train a TF-IDF and Logistic Regression text classifier.
- Choose either `Product` or `Issue` as the target label.
- Evaluate model accuracy and classification metrics.
- Display a confusion matrix to identify commonly confused categories.
- Save the trained model locally with joblib.
- Paste a new complaint or feedback message and predict its likely category.
- Display prediction confidence scores when available.
- Show business insight charts for top products, issues, states, companies, submission channels, and response categories.
- Optionally fetch a small CFPB API sample for preview/demo use.

## Tech Stack

- Python 3.11+
- Streamlit
- pandas
- NumPy
- scikit-learn
- TF-IDF vectorization
- Logistic Regression
- matplotlib
- joblib
- requests

## How It Works

1. Upload a public CFPB complaint CSV.
2. Validate required columns such as `Consumer complaint narrative` and the selected target label.
3. Clean and filter complaint narratives.
4. Split the dataset into training and test sets.
5. Convert text into TF-IDF features.
6. Train a Logistic Regression classifier.
7. Evaluate predictions with accuracy and a classification report.
8. Use the trained model to classify new complaint text.
9. Review charts that summarize recurring complaint themes.

## Model

The main model is a scikit-learn Pipeline:

```python
Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_features=10000,
        ngram_range=(1, 2)
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])
```

## Local Results

During local testing, the app was run against a public CFPB CSV with 19,191 rows. Using `Product` as the target label and the default train/test split, the model reached approximately 73.8% accuracy.

Results will vary depending on the date range, filters, target label, and number of rows used for training.

## Installation

Clone the repository:

```bash
git clone https://github.com/lolkthxbai/customer-feedback-theme-classifier.git
cd customer-feedback-theme-classifier
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the App

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in your terminal, usually:

```text
http://localhost:8501
```

## Testing

Run the automated tests from the project root:

```bash
pytest -q
```

## Project Structure

```text
customer-feedback-theme-classifier/
├── README.md
├── requirements.txt
├── .gitignore
├── app.py
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── evaluation.py
│   └── api_client.py
├── data/
│   └── .gitkeep
│   └── sample_complaints.csv
├── models/
│   └── .gitkeep
└── notebooks/
    └── exploratory_analysis.ipynb
```

## Module Overview

- `app.py`: Streamlit dashboard and user workflow.
- `src/config.py`: Shared constants such as column names, model path, and training settings.
- `src/data_loader.py`: CSV loading, required-column validation, and row limiting.
- `src/preprocessing.py`: Text cleaning and training-data preparation.
- `src/model.py`: Pipeline construction, training, prediction, model saving, and model loading.
- `src/evaluation.py`: Accuracy, classification report, confusion matrix, and top-category summaries.
- `src/api_client.py`: Optional CFPB API sample fetch and conversion to a DataFrame.

## Data Privacy

This project should only be used with public CFPB data or synthetic sample data.

Do not upload or commit:

- private employer data
- proprietary customer data
- downloaded CFPB CSV files
- trained `.joblib` model files

The `.gitignore` file excludes downloaded CSV files and `models/*.joblib`. The only CSV tracked by the repository is the synthetic `data/sample_complaints.csv` smoke-test dataset.

## Limitations

- CFPB complaint narratives are consumer-submitted and not independently verified.
- The dataset is not representative of all consumer experiences.
- The model is intended for educational and exploratory analysis, not production decision-making.
- Accuracy depends heavily on dataset size, date range, class balance, and selected target label.
- Short or vague feedback messages may produce low-confidence predictions.

## Future Improvements

- Add a Multinomial Naive Bayes comparison model.
- Add downloadable prediction results.
- Add screenshots to this README.
- Add unit tests with pytest.
- Deploy the app to Streamlit Community Cloud.
- Expand the exploratory notebook with charts and data quality checks.
