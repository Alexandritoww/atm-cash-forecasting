import pandas as pd
import pytest

from src.simulation.sensitivity_analysis import (
    run_sensitivity_analysis,
)


@pytest.fixture
def sample_data(tmp_path):

    df = pd.DataFrame({
        "scenario": ["Naive", "Random Forest", "Oracle"],
        "safety_stock": [200, 500, 100],
        "shortage_rate_pct": [35.0, 25.0, 5.0],
        "total_unmet_cop": [1000, 500, 0],
        "total_remaining_cop": [500, 1500, 2000],
        "total_replenished_cop": [1000, 2000, 3000],
    })

    path = tmp_path / "scenarios.csv"
    df.to_csv(path, index=False)

    return path


def test_number_of_combinations(sample_data):

    result = run_sensitivity_analysis(
        sample_data,
        max_shortage_pct=30,
    )

    assert len(result) == 36


def test_oracle_is_excluded(sample_data):

    result = run_sensitivity_analysis(
        sample_data,
        max_shortage_pct=30,
    )

    assert (
        result["best_scenario"] == "Random Forest"
    ).all()


def test_no_feasible_policy(sample_data):

    result = run_sensitivity_analysis(
        sample_data,
        max_shortage_pct=10,
    )

    assert not result["feasible"].any()


def test_invalid_rates(sample_data):

    with pytest.raises(ValueError):
        run_sensitivity_analysis(
            sample_data,
            shortage_rates=(-0.1,),
        )