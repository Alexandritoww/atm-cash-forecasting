from datetime import datetime

import pytest

from src.simulation.transaction import ATMTransaction


def test_valid_withdrawal_transaction():

    transaction = ATMTransaction(
        atm_id="ATM001",
        transaction_type="withdrawal",
        amount=150_000,
        denominations={
            50_000: 1,
            100_000: 1,
        },
        balance_before=1_000_000,
        balance_after=850_000,
    )

    assert transaction.amount == 150_000
    assert transaction.balance_after == 850_000
    assert isinstance(transaction.timestamp, datetime)


def test_valid_replenishment_transaction():

    transaction = ATMTransaction(
        atm_id="ATM001",
        transaction_type="replenishment",
        amount=500_000,
        denominations={
            50_000: 10,
        },
        balance_before=1_000_000,
        balance_after=1_500_000,
    )

    assert transaction.transaction_type == "replenishment"
    assert transaction.balance_after == 1_500_000


def test_invalid_transaction_type():

    with pytest.raises(ValueError):
        ATMTransaction(
            atm_id="ATM001",
            transaction_type="transfer",
            amount=100_000,
            denominations={100_000: 1},
            balance_before=500_000,
            balance_after=400_000,
        )


def test_incorrect_transaction_balance():

    with pytest.raises(ValueError):
        ATMTransaction(
            atm_id="ATM001",
            transaction_type="withdrawal",
            amount=100_000,
            denominations={100_000: 1},
            balance_before=500_000,
            balance_after=450_000,
        )


def test_incorrect_denomination_amount():

    with pytest.raises(ValueError):
        ATMTransaction(
            atm_id="ATM001",
            transaction_type="withdrawal",
            amount=200_000,
            denominations={50_000: 2},
            balance_before=500_000,
            balance_after=300_000,
        )