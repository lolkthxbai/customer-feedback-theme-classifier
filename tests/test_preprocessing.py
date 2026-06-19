import pandas as pd

from src.preprocessing import clean_text, prepare_training_data


def test_clean_text_normalizes_case_and_whitespace():
    assert clean_text("  My ACCOUNT\n has   an issue.  ") == "my account has an issue."


def test_prepare_training_data_removes_invalid_rows():
    dataframe = pd.DataFrame(
        {
            "Consumer complaint narrative": [
                "A valid complaint narrative that is long enough.",
                "short",
                None,
            ],
            "Product": ["Credit card", "Mortgage", "Checking or savings account"],
        }
    )

    cleaned_dataframe = prepare_training_data(
        dataframe,
        text_column="Consumer complaint narrative",
        target_column="Product",
        min_text_length=20,
    )

    assert cleaned_dataframe.shape[0] == 1
    assert cleaned_dataframe.loc[0, "Consumer complaint narrative"] == "a valid complaint narrative that is long enough."
