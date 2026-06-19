import joblib
import pandas as pd

from src.model import (
    load_model,
    predict_theme_with_confidence,
    predict_themes_with_confidence,
    save_model,
    train_model,
)


def test_train_model_and_predict_theme_with_confidence(tmp_path):
    dataframe = pd.DataFrame(
        {
            "Consumer complaint narrative": [
                "My card payment was not applied on time.",
                "The credit card website declined my payment.",
                "I need a replacement credit card immediately.",
                "My checking account deposit is still missing.",
                "The bank charged a fee on my savings account.",
                "I cannot see my checking account balance online.",
            ],
            "Product": [
                "Credit card",
                "Credit card",
                "Credit card",
                "Checking or savings account",
                "Checking or savings account",
                "Checking or savings account",
            ],
        }
    )

    model, _, _, labels = train_model(
        dataframe,
        text_column="Consumer complaint narrative",
        target_column="Product",
        test_size=0.33,
    )
    prediction = predict_theme_with_confidence(model, "My checking account has an unexpected fee.")

    assert set(labels) == {"Checking or savings account", "Credit card"}
    assert prediction["prediction"] in labels
    assert set(prediction["confidence_scores"]) == set(labels)

    batch_predictions = predict_themes_with_confidence(
        model,
        [
            "My checking account has an unexpected fee.",
            "My credit card payment was declined.",
        ],
    )

    assert batch_predictions.columns.tolist() == [
        "Feedback",
        "Predicted category",
        "Confidence",
    ]
    assert len(batch_predictions) == 2
    assert set(batch_predictions["Predicted category"]).issubset(set(labels))
    assert batch_predictions["Confidence"].between(0, 1).all()

    metadata = {
        "target_column": "Product",
        "training_rows": 6,
        "accuracy": 0.75,
        "model_type": "TF-IDF + Logistic Regression",
    }
    model_path = tmp_path / "saved_model.joblib"
    save_model(model, str(model_path), metadata)
    loaded_model, loaded_metadata = load_model(str(model_path))

    assert loaded_metadata == metadata
    assert predict_theme_with_confidence(
        loaded_model,
        "My checking account has an unexpected fee.",
    )["prediction"] in labels

    legacy_model_path = tmp_path / "legacy_model.joblib"
    joblib.dump(model, legacy_model_path)
    legacy_model, legacy_metadata = load_model(str(legacy_model_path))

    assert legacy_metadata["artifact_format"] == "legacy"
    assert predict_theme_with_confidence(
        legacy_model,
        "My credit card payment was declined.",
    )["prediction"] in labels
