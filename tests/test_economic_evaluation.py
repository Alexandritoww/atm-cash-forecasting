import pandas as pd
import pytest

from src.simulation.economic_evaluation import (
    EconomicConfig,
    evaluate_economic_costs,
)


def test_economic_cost_calculation(tmp_path):

    data = pd.DataFrame({
        "scenario": ["Random Forest"],
        "safety_stock": [100],
        "shortage_rate_pct": [25.0],
        "total_unmet_cop": [1_000_000],
        "total_remaining_cop": [2_000_000],
        "total_replenished_cop": [3_000_000],
    })

    path = tmp_path / "results.csv"
    data.to_csv(path, index=False)

    config = EconomicConfig(
        shortage_rate=0.10,
        holding_rate=0.001,
        replenishment_rate=0.002,
    )

    result = evaluate_economic_costs(path, config)

    assert result.iloc[0]["cost_shortage"] == 100_000
    assert result.iloc[0]["cost_holding"] == 2_000
    assert result.iloc[0]["cost_replenishment"] == 6_000
    assert result.iloc[0]["total_cost"] == 108_000


def test_invalid_economic_rate():

    with pytest.raises(ValueError):
        EconomicConfig(shortage_rate=-0.1)


def test_invalid_negative_amount(tmp_path):

    data = pd.DataFrame({
        "scenario": ["Naive"],
        "safety_stock": [20],
        "shortage_rate_pct": [10],
        "total_unmet_cop": [-100],
        "total_remaining_cop": [1000],
        "total_replenished_cop": [2000],
    })

    path = tmp_path / "invalid.csv"
    data.to_csv(path, index=False)

    with pytest.raises(ValueError):
        evaluate_economic_costs(path)