import pandas as pd
import pytest

from src.models.model_validation import (
    validate_temporal_split,
)


def test_valid_temporal_split(tmp_path):

    df = pd.DataFrame({
        "date": [
            "2020-01-04",
            "2020-01-05",
            "2020-01-06",
            "2020-01-07",
        ],
        "daily_outcome": [100, 200, 300, 400],
    })

    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)

    result = validate_temporal_split(path)

    assert result["valid_temporal_split"]
    assert result["train_records"] == 3
    assert result["test_records"] == 1


def test_missing_date_column(tmp_path):

    df = pd.DataFrame({
        "daily_outcome": [100, 200],
    })

    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)

    with pytest.raises(ValueError):
        validate_temporal_split(path)


def test_insufficient_dates(tmp_path):

    df = pd.DataFrame({
        "date": [
            "2020-01-07",
            "2020-01-07",
        ],
    })

    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)

    with pytest.raises(ValueError):
        validate_temporal_split(path)