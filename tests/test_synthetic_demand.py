import numpy as np
import pandas as pd

from src.simulation.synthetic_demand import (
    DENOMINATIONS,
    TARGET_COLUMNS,
    SyntheticDemandConfig,
    generate_synthetic_demand,
)


def create_sample_data():
    return pd.DataFrame({
        "atmId": ["ATM001", "ATM001", "ATM002"],
        "date": [
            "2020-01-01",
            "2020-01-02",
            "2020-01-01",
        ],
        "daily_outcome": [100, 250, 400],
    })


def test_synthetic_demand_is_reproducible():

    df = create_sample_data()

    config = SyntheticDemandConfig(
        units_to_cop=1000,
        seed=42,
    )

    first = generate_synthetic_demand(df, config)
    second = generate_synthetic_demand(df, config)

    pd.testing.assert_frame_equal(first, second)


def test_synthetic_demand_preserves_amount():

    df = create_sample_data()

    result = generate_synthetic_demand(df)

    reconstructed = sum(
        result[f"notes_{denomination}"] * denomination
        for denomination in DENOMINATIONS
    )

    assert (
        reconstructed == result["synthetic_amount_cop"]
    ).all()


def test_synthetic_demand_is_order_independent():

    df = create_sample_data()

    shuffled = df.sample(
        frac=1,
        random_state=123,
    ).reset_index(drop=True)

    original = generate_synthetic_demand(df)
    reordered = generate_synthetic_demand(shuffled)

    columns = ["atmId", "date"] + TARGET_COLUMNS

    original = original[columns].sort_values(
        ["atmId", "date"]
    ).reset_index(drop=True)

    reordered = reordered[columns].sort_values(
        ["atmId", "date"]
    ).reset_index(drop=True)

    pd.testing.assert_frame_equal(
        original,
        reordered,
    )


def test_synthetic_demand_rejects_missing_columns():

    df = pd.DataFrame({
        "daily_outcome": [100, 200, 300],
    })

    try:
        generate_synthetic_demand(df)
    except ValueError as error:
        assert "Faltan columnas" in str(error)
    else:
        raise AssertionError(
            "Se esperaba ValueError por columnas faltantes."
        )


def test_synthetic_demand_nonnegative_notes():

    result = generate_synthetic_demand(
        create_sample_data()
    )

    assert np.all(
        result[TARGET_COLUMNS].to_numpy() >= 0
    )