from io import BytesIO

import pytest

from src.data_loader import load_csv, validate_required_columns


def test_load_csv_reads_uploaded_content():
    uploaded_file = BytesIO(b"Consumer complaint narrative,Product\nA complaint message,Credit card\n")

    dataframe = load_csv(uploaded_file)

    assert dataframe.shape == (1, 2)
    assert dataframe.loc[0, "Product"] == "Credit card"


def test_validate_required_columns_reports_missing_columns():
    dataframe = load_csv(BytesIO(b"Consumer complaint narrative\nA complaint message\n"))

    with pytest.raises(ValueError, match="Missing required column\\(s\\): Product"):
        validate_required_columns(dataframe, ["Consumer complaint narrative", "Product"])
