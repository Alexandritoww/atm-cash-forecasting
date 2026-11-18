import pytest
from src.simulation.denomination_simulator import (
    SimulationConfig,
    DENOMINATIONS,
    dispense_cash,
)

from src.simulation.denomination_simulator import (
    SimulationConfig,
    DENOMINATIONS,
)


def test_denominations_are_correct():
    assert DENOMINATIONS == (
        10_000,
        20_000,
        50_000,
        100_000,
    )


def test_default_configuration():
    config = SimulationConfig()

    assert config.random_seed == 42
    assert config.denominations == DENOMINATIONS


def test_invalid_denominations():
    with pytest.raises(ValueError):
        SimulationConfig(
            denominations=(10_000, -20_000)
        )


def test_duplicate_denominations():
    with pytest.raises(ValueError):
        SimulationConfig(
            denominations=(10_000, 10_000)
        )

def test_dispense_valid_amount():
    inventory = {
        10_000: 100,
        20_000: 100,
        50_000: 100,
        100_000: 100,
    }

    result = dispense_cash(180_000, inventory)

    total = sum(
        denomination * quantity
        for denomination, quantity in result.items()
    )

    assert total == 180_000
    assert sum(result.values()) == 4


def test_dispense_respects_inventory():
    inventory = {
        10_000: 1,
        20_000: 1,
        50_000: 0,
        100_000: 0,
    }

    result = dispense_cash(30_000, inventory)

    assert result[10_000] == 1
    assert result[20_000] == 1


def test_dispense_insufficient_cash():
    inventory = {
        10_000: 1,
        20_000: 0,
        50_000: 0,
        100_000: 0,
    }

    with pytest.raises(ValueError):
        dispense_cash(100_000, inventory)


def test_dispense_invalid_amount():
    inventory = {10_000: 10}

    with pytest.raises(ValueError):
        dispense_cash(-10_000, inventory)


def test_dispense_does_not_modify_inventory():
    inventory = {
        10_000: 10,
        20_000: 10,
        50_000: 10,
        100_000: 10,
    }

    original = inventory.copy()

    dispense_cash(180_000, inventory)

    assert inventory == original