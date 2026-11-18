from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class EconomicConfig:
    """
    Parámetros económicos hipotéticos.

    shortage_rate:
        Penalización por unidad monetaria no atendida.

    holding_rate:
        Costo de mantener efectivo restante durante
        el horizonte simulado.

    replenishment_rate:
        Costo proporcional del efectivo recargado.
    """

    shortage_rate: float = 0.10
    holding_rate: float = 0.001
    replenishment_rate: float = 0.002

    def __post_init__(self):
        for name in (
            "shortage_rate",
            "holding_rate",
            "replenishment_rate",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0 <= value < float("inf")
            ):
                raise ValueError(
                    f"{name} debe ser un número finito no negativo."
                )


def evaluate_economic_costs(
    data_path: str | Path,
    config: EconomicConfig | None = None,
) -> pd.DataFrame:
    """
    Calcula los costos hipotéticos de cada política
    y nivel de stock de seguridad.
    """

    if config is None:
        config = EconomicConfig()

    df = pd.read_csv(data_path)

    required_columns = [
        "scenario",
        "safety_stock",
        "shortage_rate_pct",
        "total_unmet_cop",
        "total_remaining_cop",
        "total_replenished_cop",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Faltan columnas: {missing}"
        )

    for col in (
        "total_unmet_cop",
        "total_remaining_cop",
        "total_replenished_cop",
    ):
        if df[col].isna().any() or (df[col] < 0).any():
            raise ValueError(
                f"Valores inválidos en {col}"
            )

    df["cost_shortage"] = (
        df["total_unmet_cop"]
        * config.shortage_rate
    )

    df["cost_holding"] = (
        df["total_remaining_cop"]
        * config.holding_rate
    )

    df["cost_replenishment"] = (
        df["total_replenished_cop"]
        * config.replenishment_rate
    )

    df["total_cost"] = (
        df["cost_shortage"]
        + df["cost_holding"]
        + df["cost_replenishment"]
    )

    return df.sort_values(
        "total_cost"
    ).reset_index(drop=True)