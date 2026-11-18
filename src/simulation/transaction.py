from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class ATMTransaction:
    """
    Representa un movimiento de efectivo de un ATM.

    Registra:
    - Cajero donde ocurrió la operación.
    - Tipo de movimiento.
    - Monto monetario.
    - Billetes involucrados.
    - Saldo anterior y posterior.
    - Fecha y hora de la operación.
    """

    atm_id: str
    transaction_type: str
    amount: int
    denominations: dict[int, int]
    balance_before: int
    balance_after: int
    timestamp: datetime = field(
        default_factory=datetime.now
    )

    def __post_init__(self):
        """Valida la información de la transacción."""

        if not isinstance(self.atm_id, str) or not self.atm_id.strip():
            raise ValueError(
                "El identificador del ATM es obligatorio."
            )

        if self.transaction_type not in ("withdrawal", "replenishment"):
            raise ValueError(
                "El tipo debe ser withdrawal o replenishment."
            )

        for value in (
            self.amount,
            self.balance_before,
            self.balance_after,
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(
                    "Los valores monetarios deben ser enteros."
                )

        if self.amount <= 0:
            raise ValueError(
                "El monto de la transacción debe ser positivo."
            )

        if self.balance_before < 0 or self.balance_after < 0:
            raise ValueError(
                "Los saldos no pueden ser negativos."
            )

        if not isinstance(self.denominations, dict):
            raise ValueError(
                "Las denominaciones deben ser un diccionario."
            )

        from src.simulation.denomination_simulator import DENOMINATIONS

        for denomination, quantity in self.denominations.items():
            if (
                isinstance(denomination, bool)
                or not isinstance(denomination, int)
                or denomination not in DENOMINATIONS
                or isinstance(quantity, bool)
                or not isinstance(quantity, int)
                or quantity < 0
            ):
                raise ValueError(
                    "Las denominaciones y cantidades deben ser válidas."
                )

        calculated_amount = sum(
            denomination * quantity
            for denomination, quantity in self.denominations.items()
        )

        if calculated_amount != self.amount:
            raise ValueError(
                "El monto no coincide con los billetes registrados."
            )

        expected_balance = (
            self.balance_before - self.amount
            if self.transaction_type == "withdrawal"
            else self.balance_before + self.amount
        )

        if self.balance_after != expected_balance:
            raise ValueError(
                "El saldo posterior no coincide con la operación."
            )

        if not isinstance(self.timestamp, datetime):
            raise ValueError(
                "La fecha debe ser un objeto datetime."
            )