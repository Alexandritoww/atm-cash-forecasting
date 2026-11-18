import pandas as pd
import pytest

from src.simulation.policy_selection import (
    select_best_policy,
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
        "shortage_rate_pct": [35.0, 8.0, 4.0],
        "total_cost": [1000, 1500, 2000],
    })

    path = tmp_path / "policies.csv"
    df.to_csv(path, index=False)

    return path


def test_select_best_feasible_policy(sample_data):

    result = select_best_policy(
        sample_data,
        max_shortage_pct=10,
    )

    assert result["feasible"] is True

    assert (
        result["best_policy"]["scenario"]
        == "Random Forest"
    )


def test_no_feasible_policy(sample_data):

    result = select_best_policy(
        sample_data,
        max_shortage_pct=2,
    )

    assert result["feasible"] is False

    assert "closest_policy" in result


def test_invalid_service_limit(sample_data):

    with pytest.raises(ValueError):
        select_best_policy(
            sample_data,
            max_shortage_pct=120,
        )


def test_cheapest_policy_within_limit(sample_data):

    result = select_best_policy(
        sample_data,
        max_shortage_pct=40,
    )

    assert result["feasible"] is True

    assert (
        result["best_policy"]["scenario"]
        == "Naive"
    )