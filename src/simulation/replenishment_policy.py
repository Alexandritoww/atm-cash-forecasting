from dataclasses import dataclass

from src.simulation.denomination_simulator import DENOMINATIONS


@dataclass(frozen=True)
class ReplenishmentPolicy:
    """
    Política de reabastecimiento basada en
    demanda prevista por denominación.
    """

    safety_stock: int = 20
    cassette_capacity: int = 2000

    def __post_init__(self):
        if (
            isinstance(self.safety_stock, bool)
            or not isinstance(self.safety_stock, int)
            or self.safety_stock < 0
        ):
            raise ValueError(
                "El stock de seguridad debe ser un entero no negativo."
            )

        if (
            isinstance(self.cassette_capacity, bool)
            or not isinstance(self.cassette_capacity, int)
            or self.cassette_capacity <= 0
        ):
            raise ValueError(
                "La capacidad del casete debe ser positiva."
            )

    def calculate_replenishment(
        self,
        inventory: dict[int, int],
        predicted_demand: dict[int, int],
    ) -> dict[int, int]:
        """
        Calcula cuántos billetes agregar por denominación.

        No modifica el inventario recibido.
        """

        if set(inventory) != set(DENOMINATIONS):
            raise ValueError(
                "El inventario debe contener todas las denominaciones."
            )

        if set(predicted_demand) != set(DENOMINATIONS):
            raise ValueError(
                "La demanda prevista debe contener todas las denominaciones."
            )

        replenishment = {}

        for denomination in DENOMINATIONS:

            current = inventory[denomination]
            predicted = predicted_demand[denomination]

            if (
                isinstance(current, bool)
                or not isinstance(current, int)
                or current < 0
                or current > self.cassette_capacity
            ):
                raise ValueError(
                    f"Inventario inválido para {denomination}."
                )

            if (
                isinstance(predicted, bool)
                or not isinstance(predicted, int)
                or predicted < 0
            ):
                raise ValueError(
                    f"Demanda inválida para {denomination}."
                )

            target_stock = predicted + self.safety_stock

            # Nunca superar la capacidad del casete.
            target_stock = min(
                target_stock,
                self.cassette_capacity,
            )

            quantity = max(
                0,
                target_stock - current,
            )

            replenishment[denomination] = quantity

        return replenishment