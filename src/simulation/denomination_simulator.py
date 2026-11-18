from dataclasses import dataclass


# ============================================================
# 1. DENOMINACIONES DISPONIBLES
# ============================================================

# Denominaciones utilizadas en nuestra simulación (COP).
DENOMINATIONS = (
    10_000,
    20_000,
    50_000,
    100_000,
)


# ============================================================
# 2. CONFIGURACIÓN DEL SIMULADOR
# ============================================================

@dataclass(frozen=True)
class SimulationConfig:
    """
    Configuración general del simulador de denominaciones.

    Attributes:
        denominations:
            Denominaciones de billetes disponibles.

        random_seed:
            Semilla para garantizar reproducibilidad
            en futuros experimentos.
    """

    denominations: tuple[int, ...] = DENOMINATIONS
    random_seed: int = 42

    def __post_init__(self):
        """Valida la configuración del simulador."""

        if not self.denominations:
            raise ValueError(
                "Debe existir al menos una denominación."
            )

        if any(
            isinstance(d, bool)
            or not isinstance(d, int)
            or d <= 0
            for d in self.denominations
        ):
            raise ValueError(
                "Las denominaciones deben ser enteros positivos."
            )

        if len(set(self.denominations)) != len(
            self.denominations
        ):
            raise ValueError(
                "No puede haber denominaciones repetidas."
            )

        if (
            isinstance(self.random_seed, bool)
            or not isinstance(self.random_seed, int)
        ):
            raise ValueError(
                "La semilla debe ser un número entero."
            )


# ============================================================
# 3. ALGORITMO DE DISPENSACIÓN DE EFECTIVO
# ============================================================

def dispense_cash(
    amount: int,
    inventory: dict[int, int],
) -> dict[int, int]:
    """
    Calcula una combinación válida de billetes para un retiro.

    Objetivos:
        1. Minimizar el número total de billetes.
        2. Respetar el inventario disponible.
        3. No modificar el inventario original.
        4. Rechazar retiros que no puedan completarse.

    Args:
        amount:
            Monto solicitado en pesos colombianos.

        inventory:
            Diccionario con la cantidad disponible
            de billetes por denominación.

    Returns:
        Diccionario con los billetes que deben
        dispensarse por denominación.

    Raises:
        ValueError:
            Cuando el monto, inventario o combinación
            solicitada no es válida.
    """

    # --------------------------------------------------------
    # 3.1. VALIDACIÓN DEL MONTO
    # --------------------------------------------------------

    if isinstance(amount, bool) or not isinstance(amount, int):
        raise ValueError(
            "El monto debe ser un número entero."
        )

    if amount <= 0:
        raise ValueError(
            "El monto solicitado debe ser positivo."
        )

    if amount % min(DENOMINATIONS) != 0:
        raise ValueError(
            "El monto debe ser múltiplo de 10.000 COP."
        )

    # --------------------------------------------------------
    # 3.2. VALIDACIÓN DEL INVENTARIO
    # --------------------------------------------------------

    if not isinstance(inventory, dict) or not inventory:
        raise ValueError(
            "El inventario debe ser un diccionario no vacío."
        )

    for denomination, quantity in inventory.items():

        if (
            isinstance(denomination, bool)
            or not isinstance(denomination, int)
            or denomination not in DENOMINATIONS
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

    # --------------------------------------------------------
    # 3.3. VALIDACIÓN DE EFECTIVO TOTAL
    # --------------------------------------------------------

    total_available = sum(
        denomination * quantity
        for denomination, quantity in inventory.items()
    )

    if amount > total_available:
        raise ValueError(
            "El cajero no tiene suficiente efectivo."
        )

    # --------------------------------------------------------
    # 3.4. PREPARACIÓN DEL ALGORITMO
    # --------------------------------------------------------

    denominations = sorted(
        inventory.keys(),
        reverse=True
    )

    best_solution = None
    best_count = float("inf")

    # --------------------------------------------------------
    # 3.5. BÚSQUEDA DE COMBINACIONES
    # --------------------------------------------------------

    def search(
        index: int,
        remaining: int,
        selected: dict[int, int],
        count: int,
    ):
        nonlocal best_solution, best_count

        # Encontramos una combinación válida
        if remaining == 0:

            if count < best_count:
                best_solution = selected.copy()
                best_count = count

            return

        # No quedan denominaciones por evaluar
        if index >= len(denominations):
            return

        # Poda: ya usamos demasiados billetes
        if count >= best_count:
            return

        denomination = denominations[index]

        # Máximo número de billetes que podemos utilizar
        max_notes = min(
            inventory[denomination],
            remaining // denomination
        )

        # Evaluamos combinaciones desde la mayor cantidad
        for quantity in range(max_notes, -1, -1):

            selected[denomination] = quantity

            new_remaining = (
                remaining - denomination * quantity
            )

            new_count = count + quantity

            search(
                index + 1,
                new_remaining,
                selected,
                new_count,
            )

        # Limpiamos el estado para retroceder
        selected.pop(denomination, None)

    # --------------------------------------------------------
    # 3.6. EJECUCIÓN DEL ALGORITMO
    # --------------------------------------------------------

    search(
        index=0,
        remaining=amount,
        selected={},
        count=0,
    )

    # --------------------------------------------------------
    # 3.7. VALIDACIÓN DEL RESULTADO
    # --------------------------------------------------------

    if best_solution is None:
        raise ValueError(
            "No es posible dispensar el monto solicitado "
            "con las denominaciones disponibles."
        )

    # --------------------------------------------------------
    # 3.8. RESULTADO FINAL
    # --------------------------------------------------------

    return {
        denomination: best_solution.get(denomination, 0)
        for denomination in sorted(inventory.keys())
    }