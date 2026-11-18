import pandas as pd

from src.models.temporal_audit import (
    audit_temporal_features,
)


def test_temporal_audit_valid_data(tmp_path):

    dates = pd.date_range(
        "2020-01-01",
        periods=5,
        freq="D",
    )

    df = pd.DataFrame({
        "atmId": ["ATM001"] * 5,
        "date": dates,
        "daily_outcome": [
            100, 200, 300, 400, 500
        ],
        "outcome_lag_1": [
            None, 100, 200, 300, 400
        ],
        "outcome_lag_2": [
            None, None, 100, 200, 300
        ],
        "outcome_rolling_mean_3": [
            None, None, None, 200, 300
        ],
    })

    path = tmp_path / "temporal_data.csv"
    df.to_csv(path, index=False)

    results = audit_temporal_features(path)

    assert results["duplicated_atm_dates"] == 0
    assert results["temporal_gaps"] == 0
    assert results["lag_1_mismatches"] == 0
    assert results["lag_2_mismatches"] == 0
    assert results["rolling_mean_mismatches"] == 0