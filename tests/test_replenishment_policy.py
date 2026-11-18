import pytest

from src.simulation.replenishment_policy import (
    ReplenishmentPolicy,
)


# ============================================================
# TEST 1: VALIDAR EL CÁLCULO DE REABASTECIMIENTO
# ============================================================

def test_replenishment_calculation():
    """
    Verifica que la política calcule correctamente
    la cantidad de billetes que debemos recargar.
    """

    policy = ReplenishmentPolicy(
        safety_stock=20,
        cassette_capacity=2000,
    )

    inventory = {
        10_000: 100,
        20_000: 50,
        50_000: 40,
        100_000: 30,
    }

    predicted_demand = {
        10_000: 120,
        20_000: 80,
        50_000: 150,
        100_000: 40,
    }

    result = policy.calculate_replenishment(
        inventory,
        predicted_demand,
    )

    expected = {
        10_000: 40,
        20_000: 50,
        50_000: 130,
        100_000: 30,
    }

    assert result == expected


# ============================================================
# TEST 2: RESPETAR LA CAPACIDAD MÁXIMA DEL CASETE
# ============================================================

def test_replenishment_respects_capacity():
    """
    Verifica que ninguna recarga supere la
    capacidad máxima de 2000 billetes por casete.
    """

    policy = ReplenishmentPolicy(
        safety_stock=20,
        cassette_capacity=2000,
    )

    inventory = {
        10_000: 1900,
        20_000: 1900,
        50_000: 1900,
        100_000: 1900,
    }

    # Demanda superior a la capacidad del casete.
    predicted_demand = {
        10_000: 3000,
        20_000: 3000,
        50_000: 3000,
        100_000: 3000,
    }

    result = policy.calculate_replenishment(
        inventory,
        predicted_demand,
    )

    # Solo hay espacio para 100 billetes por casete.
    expected = {
        10_000: 100,
        20_000: 100,
        50_000: 100,
        100_000: 100,
    }

    assert result == expected

    # Verificar que no se supere la capacidad.
    for denomination, quantity in result.items():
        assert (
            inventory[denomination] + quantity
            <= policy.cassette_capacity
        )


# ============================================================
# TEST 3: VALIDAR DEMANDA NEGATIVA
# ============================================================

def test_invalid_predicted_demand():
    """
    Verifica que la política rechace valores
    negativos en la demanda prevista.
    """

    policy = ReplenishmentPolicy(
        safety_stock=20,
        cassette_capacity=2000,
    )

    inventory = {
        10_000: 100,
        20_000: 100,
        50_000: 100,
        100_000: 100,
    }

    predicted_demand = {
        10_000: -5,
        20_000: 100,
        50_000: 100,
        100_000: 100,
    }

    with pytest.raises(ValueError):
        policy.calculate_replenishment(
            inventory,
            predicted_demand,
        )