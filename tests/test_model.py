import pandas as pd

from src.model import predict_theme_with_confidence, predict_themes_with_confidence, train_model


def test_train_model_and_predict_theme_with_confidence():
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
