import pandas as pd
import pytest

from src.simulation.service_level_analysis import (
    analyze_service_levels,
)


@pytest.fixture
def sample_data(tmp_path):

    df = pd.DataFrame({
        "scenario": [
            "Naive",
            "Random Forest",
            "Oracle",
        ],
        "safety_stock": [200, 500, 100],
        "shortage_rate_pct": [70.0, 25.0, 5.0],
        "total_unmet_cop": [1000, 500, 0],
        "total_remaining_cop": [500, 1500, 2000],
        "total_replenished_cop": [1000, 2000, 3000],
    })

    path = tmp_path / "service_levels.csv"
    df.to_csv(path, index=False)

    return path


def test_number_of_evaluations(sample_data):

    result = analyze_service_levels(
        sample_data,
        service_levels=(30, 60, 100),
    )

    assert len(result) == 108


def test_service_levels_are_preserved(sample_data):

    result = analyze_service_levels(
        sample_data,
        service_levels=(30, 60, 100),
    )

    assert set(result["service_limit_pct"]) == {
        30, 60, 100
    }


def test_oracle_is_not_selected(sample_data):

    result = analyze_service_levels(
        sample_data,
        service_levels=(30, 60, 100),
    )

    selected = result["best_scenario"].dropna()

    assert "Oracle" not in selected.values


def test_empty_service_levels(sample_data):

    with pytest.raises(ValueError):
        analyze_service_levels(
            sample_data,
            service_levels=(),
        )