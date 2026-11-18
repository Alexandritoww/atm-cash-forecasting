from dataclasses import dataclass, field
from datetime import datetime

from src.simulation.denomination_simulator import (
    DENOMINATIONS,
    dispense_cash,
)

from src.simulation.transaction import ATMTransaction


@dataclass
class ATMInventory:
    """
    Representa el inventario de efectivo de un cajero ATM.

    Funcionalidades:
    - Validar el inventario inicial.
    - Calcular el efectivo total disponible.
    - Procesar retiros de efectivo.
    - Detectar inventario bajo.
    - Consultar el inventario.
    - Reabastecer los casetes.
    - Registrar automáticamente las transacciones.
    - Consultar el historial de movimientos.

    Todos los valores monetarios se expresan en COP.
    """

    atm_id: str
    inventory: dict[int, int]
    low_stock_threshold: int = 20
    cassette_capacity: int = 2000

    transaction_history: list[ATMTransaction] = field(
        default_factory=list,
        init=False,
        repr=False,
    )

    # ========================================================
    # 1. VALIDACIÓN DEL INVENTARIO INICIAL
    # ========================================================

    def __post_init__(self):
        """
        Valida los datos iniciales del cajero ATM.
        """

        # Validar identificador del cajero.
        if (
            not isinstance(self.atm_id, str)
            or not self.atm_id.strip()
        ):
            raise ValueError(
                "El identificador del ATM es obligatorio."
            )

        # Validar umbral de inventario bajo.
        if (
            isinstance(self.low_stock_threshold, bool)
            or not isinstance(self.low_stock_threshold, int)
            or self.low_stock_threshold < 0
        ):
            raise ValueError(
                "El umbral debe ser un entero no negativo."
            )

        # Validar capacidad máxima del casete.
        if (
            isinstance(self.cassette_capacity, bool)
            or not isinstance(self.cassette_capacity, int)
            or self.cassette_capacity <= 0
        ):
            raise ValueError(
                "La capacidad del casete debe ser un entero positivo."
            )

        # Validar estructura del inventario.
        if not isinstance(self.inventory, dict):
            raise ValueError(
                "El inventario debe ser un diccionario."
            )

        # Validar denominaciones permitidas.
        if set(self.inventory.keys()) != set(DENOMINATIONS):
            raise ValueError(
                "El inventario debe contener las cuatro "
                "denominaciones permitidas."
            )

        # Validar cantidad de billetes y capacidad.
        for denomination, quantity in self.inventory.items():

            if (
                isinstance(quantity, bool)
                or not isinstance(quantity, int)
                or quantity < 0
            ):
                raise ValueError(
                    f"Cantidad inválida para {denomination}."
                )

            if quantity > self.cassette_capacity:
                raise ValueError(
                    f"El inventario de {denomination} "
                    "supera la capacidad del casete."
                )

        # Crear una copia para proteger el inventario interno.
        self.inventory = self.inventory.copy()

    # ========================================================
    # 2. CALCULAR EFECTIVO TOTAL
    # ========================================================

    def total_cash(self) -> int:
        """
        Calcula el efectivo total disponible en el ATM.

        Returns:
            int: Valor total en COP.
        """

        return sum(
            denomination * quantity
            for denomination, quantity in self.inventory.items()
        )

    # ========================================================
    # 3. PROCESAR RETIROS
    # ========================================================

    def withdraw(
        self,
        amount: int,
        timestamp: datetime | None = None,
    ) -> dict[int, int]:
        """
        Procesa un retiro y registra la transacción.

        Args:
            amount:
                Monto solicitado en COP.

            timestamp:
                Fecha y hora de la operación.
                Si no se proporciona, utiliza la actual.

        Returns:
            dict[int, int]:
                Billetes dispensados por denominación.
        """

        # 3.1. Obtener el saldo antes del retiro.
        balance_before = self.total_cash()

        # 3.2. Calcular una combinación válida de billetes.
        dispensed = dispense_cash(
            amount=amount,
            inventory=self.inventory,
        )

        # 3.3. Calcular el saldo esperado después del retiro.
        balance_after = balance_before - amount

        # 3.4. Crear y validar el registro de la transacción.
        transaction = ATMTransaction(
            atm_id=self.atm_id,
            transaction_type="withdrawal",
            amount=amount,
            denominations=dispensed.copy(),
            balance_before=balance_before,
            balance_after=balance_after,
            timestamp=(
                timestamp
                if timestamp is not None
                else datetime.now()
            ),
        )

        # 3.5. Actualizar el inventario.
        for denomination, quantity in dispensed.items():
            self.inventory[denomination] -= quantity

        # 3.6. Registrar la transacción.
        self.transaction_history.append(transaction)

        return dispensed

    # ========================================================
    # 4. DETECTAR INVENTARIO BAJO
    # ========================================================

    def low_stock(self) -> dict[int, int]:
        """
        Identifica denominaciones con inventario bajo.

        Se considera inventario bajo cuando la cantidad
        es menor o igual al umbral configurado.

        Returns:
            dict[int, int]:
                Denominaciones con inventario bajo.
        """

        return {
            denomination: quantity
            for denomination, quantity in self.inventory.items()
            if quantity <= self.low_stock_threshold
        }

    # ========================================================
    # 5. CONSULTAR INVENTARIO
    # ========================================================

    def get_inventory(self) -> dict[int, int]:
        """
        Devuelve una copia del inventario actual.

        Returns:
            dict[int, int]:
                Cantidad de billetes disponibles
                por denominación.
        """

        return self.inventory.copy()

    # ========================================================
    # 6. REABASTECER EFECTIVO
    # ========================================================

    def replenish(
        self,
        replenishment: dict[int, int],
        timestamp: datetime | None = None,
    ) -> int:
        """
        Recarga los casetes del ATM y registra el movimiento.

        Args:
            replenishment:
                Cantidad de billetes que se agregarán
                por denominación.

            timestamp:
                Fecha y hora de la recarga.

        Returns:
            int: Valor monetario total recargado en COP.
        """

        # 6.1. Validar estructura de la recarga.
        if (
            not isinstance(replenishment, dict)
            or not replenishment
        ):
            raise ValueError(
                "La recarga debe ser un diccionario no vacío."
            )

        # 6.2. Validar denominaciones, cantidades y capacidad.
        for denomination, quantity in replenishment.items():

            if (
                isinstance(denomination, bool)
                or not isinstance(denomination, int)
                or denomination not in self.inventory
            ):
                raise ValueError(
                    f"Denominación no permitida: {denomination}"
                )

            if (
                isinstance(quantity, bool)
                or not isinstance(quantity, int)
                or quantity < 0
            ):
                raise ValueError(
                    f"Cantidad inválida para {denomination}."
                )

            new_quantity = (
                self.inventory[denomination] + quantity
            )

            if new_quantity > self.cassette_capacity:
                raise ValueError(
                    f"La recarga de {denomination} "
                    "supera la capacidad máxima del casete."
                )

        # 6.3. Calcular el valor total de la recarga.
        total_replenished = sum(
            denomination * quantity
            for denomination, quantity in replenishment.items()
        )

        if total_replenished <= 0:
            raise ValueError(
                "La recarga debe agregar efectivo."
            )

        # 6.4. Calcular saldos.
        balance_before = self.total_cash()
        balance_after = balance_before + total_replenished

        # 6.5. Crear y validar la transacción.
        transaction = ATMTransaction(
            atm_id=self.atm_id,
            transaction_type="replenishment",
            amount=total_replenished,
            denominations=replenishment.copy(),
            balance_before=balance_before,
            balance_after=balance_after,
            timestamp=(
                timestamp
                if timestamp is not None
                else datetime.now()
            ),
        )

        # 6.6. Actualizar inventario.
        for denomination, quantity in replenishment.items():
            self.inventory[denomination] += quantity

        # 6.7. Registrar la transacción.
        self.transaction_history.append(transaction)

        return total_replenished

    # ========================================================
    # 7. CONSULTAR HISTORIAL DE TRANSACCIONES
    # ========================================================

    def get_transaction_history(self) -> list[ATMTransaction]:
        """
        Devuelve el historial de transacciones del ATM.

        Se generan copias independientes de los registros
        para proteger los diccionarios internos.

        Returns:
            list[ATMTransaction]:
                Lista de movimientos registrados.
        """

        return [
            ATMTransaction(
                atm_id=transaction.atm_id,
                transaction_type=transaction.transaction_type,
                amount=transaction.amount,
                denominations=transaction.denominations.copy(),
                balance_before=transaction.balance_before,
                balance_after=transaction.balance_after,
                timestamp=transaction.timestamp,
            )
            for transaction in self.transaction_history
        ]