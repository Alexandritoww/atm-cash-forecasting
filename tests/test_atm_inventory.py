import pytest

from src.simulation.atm_inventory import ATMInventory


@pytest.fixture
def atm():
    return ATMInventory(
        atm_id="ATM001",
        inventory={
            10_000: 100,
            20_000: 100,
            50_000: 100,
            100_000: 100,
        },
    )


def test_initial_cash(atm):
    assert atm.total_cash() == 18_000_000


def test_withdraw_updates_inventory(atm):
    initial_cash = atm.total_cash()

    dispensed = atm.withdraw(180_000)

    assert atm.total_cash() == initial_cash - 180_000
    assert sum(
        denomination * quantity
        for denomination, quantity in dispensed.items()
    ) == 180_000


def test_failed_withdraw_keeps_inventory(atm):
    original = atm.get_inventory()

    with pytest.raises(ValueError):
        atm.withdraw(20_000_000)

    assert atm.get_inventory() == original


def test_low_stock():
    atm = ATMInventory(
        atm_id="ATM002",
        inventory={
            10_000: 10,
            20_000: 50,
            50_000: 5,
            100_000: 100,
        },
        low_stock_threshold=20,
    )

    assert atm.low_stock() == {
        10_000: 10,
        50_000: 5,
    }


def test_inventory_is_independent():
    original = {
        10_000: 100,
        20_000: 100,
        50_000: 100,
        100_000: 100,
    }

    atm = ATMInventory("ATM003", original)
    atm.withdraw(100_000)

    assert original[100_000] == 100

def test_replenish_updates_inventory(atm):
    initial_cash = atm.total_cash()

    replenished = atm.replenish({
        10_000: 50,
        50_000: 20,
    })

    assert replenished == 1_500_000
    assert atm.total_cash() == initial_cash + 1_500_000
    assert atm.get_inventory()[10_000] == 150
    assert atm.get_inventory()[50_000] == 120


def test_replenish_exceeds_capacity(atm):
    original = atm.get_inventory()

    with pytest.raises(ValueError):
        atm.replenish({
            10_000: 50,
            100_000: 2000,
        })

    assert atm.get_inventory() == original


def test_replenish_invalid_quantity(atm):
    original = atm.get_inventory()

    with pytest.raises(ValueError):
        atm.replenish({
            20_000: -10,
        })

    assert atm.get_inventory() == original


def test_initial_inventory_exceeds_capacity():
    with pytest.raises(ValueError):
        ATMInventory(
            atm_id="ATM004",
            inventory={
                10_000: 2001,
                20_000: 100,
                50_000: 100,
                100_000: 100,
            },
        )
from datetime import datetime


def test_withdraw_registers_transaction():

    atm = ATMInventory(
        atm_id="ATM001",
        inventory={
            10_000: 100,
            20_000: 100,
            50_000: 100,
            100_000: 100,
        },
    )

    timestamp = datetime(2026, 10, 17, 8, 0)

    initial_cash = atm.total_cash()

    atm.withdraw(180_000, timestamp=timestamp)

    history = atm.get_transaction_history()

    assert len(history) == 1
    assert history[0].transaction_type == "withdrawal"
    assert history[0].amount == 180_000
    assert history[0].balance_before == initial_cash
    assert history[0].balance_after == initial_cash - 180_000
    assert history[0].timestamp == timestamp


def test_replenish_registers_transaction():

    atm = ATMInventory(
        atm_id="ATM002",
        inventory={
            10_000: 100,
            20_000: 100,
            50_000: 100,
            100_000: 100,
        },
    )

    initial_cash = atm.total_cash()

    atm.replenish({
        10_000: 50,
        50_000: 20,
    })

    history = atm.get_transaction_history()

    assert len(history) == 1
    assert history[0].transaction_type == "replenishment"
    assert history[0].amount == 1_500_000
    assert history[0].balance_before == initial_cash
    assert history[0].balance_after == initial_cash + 1_500_000


def test_failed_withdraw_does_not_register_transaction():

    atm = ATMInventory(
        atm_id="ATM003",
        inventory={
            10_000: 10,
            20_000: 10,
            50_000: 10,
            100_000: 10,
        },
    )

    original_inventory = atm.get_inventory()

    with pytest.raises(ValueError):
        atm.withdraw(100_000_000)

    assert atm.get_inventory() == original_inventory
    assert len(atm.get_transaction_history()) == 0


def test_failed_replenish_does_not_register_transaction():

    atm = ATMInventory(
        atm_id="ATM004",
        inventory={
            10_000: 100,
            20_000: 100,
            50_000: 100,
            100_000: 100,
        },
    )

    original_inventory = atm.get_inventory()

    with pytest.raises(ValueError):
        atm.replenish({
            10_000: 50,
            100_000: 2000,
        })

    assert atm.get_inventory() == original_inventory
    assert len(atm.get_transaction_history()) == 0


def test_transaction_history_returns_independent_copies():

    atm = ATMInventory(
        atm_id="ATM005",
        inventory={
            10_000: 100,
            20_000: 100,
            50_000: 100,
            100_000: 100,
        },
    )

    atm.withdraw(100_000)

    history = atm.get_transaction_history()
    history[0].denominations[100_000] = 999

    original_history = atm.get_transaction_history()

    assert original_history[0].denominations[100_000] == 1