from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


def run_sensitivity_analysis(
    data_path: str | Path,
    max_shortage_pct: float = 30.0,
    shortage_rates=(0.05, 0.10, 0.20, 0.40),
    holding_rates=(0.0005, 0.001, 0.002),
    replenishment_rates=(0.001, 0.002, 0.005),
) -> pd.DataFrame:
    """
    Analiza la sensibilidad de la política seleccionada
    ante diferentes supuestos económicos.

    Solo considera políticas implementables:
    Naive y Random Forest.
    """

    df = pd.read_csv(data_path)

    required = [
        "scenario",
        "safety_stock",
        "shortage_rate_pct",
        "total_unmet_cop",
        "total_remaining_cop",
        "total_replenished_cop",
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Faltan columnas: {missing}")

    if (
        not np.isfinite(max_shortage_pct)
        or not 0 <= max_shortage_pct <= 100
    ):
        raise ValueError("Límite de servicio inválido.")

    rates = (
        list(shortage_rates)
        + list(holding_rates)
        + list(replenishment_rates)
    )

    if any(
        not np.isfinite(rate) or rate < 0
        for rate in rates
    ):
        raise ValueError("Las tasas deben ser finitas y no negativas.")

    # Oracle es una referencia teórica, no implementable.
    df = df[df["scenario"].isin(["Naive", "Random Forest"])].copy()

    for column in [
        "shortage_rate_pct",
        "total_unmet_cop",
        "total_remaining_cop",
        "total_replenished_cop",
    ]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

        if (
            not np.isfinite(df[column].to_numpy()).all()
            or (df[column] < 0).any()
        ):
            raise ValueError(f"Valores inválidos en {column}")

    if (df["shortage_rate_pct"] > 100).any():
        raise ValueError("Porcentaje de faltantes inválido.")

    if df.empty:
        raise ValueError("No existen políticas implementables.")

    results = []

    for shortage, holding, replenishment in product(
        shortage_rates,
        holding_rates,
        replenishment_rates,
    ):

        scenarios = df.copy()

        scenarios["cost_shortage"] = (
            scenarios["total_unmet_cop"] * shortage
        )

        scenarios["cost_holding"] = (
            scenarios["total_remaining_cop"] * holding
        )

        scenarios["cost_replenishment"] = (
            scenarios["total_replenished_cop"] * replenishment
        )

        scenarios["total_cost"] = (
            scenarios["cost_shortage"]
            + scenarios["cost_holding"]
            + scenarios["cost_replenishment"]
        )

        feasible = scenarios[
            scenarios["shortage_rate_pct"] <= max_shortage_pct
        ].copy()

        result = {
            "shortage_rate": shortage,
            "holding_rate": holding,
            "replenishment_rate": replenishment,
            "max_shortage_pct": max_shortage_pct,
            "feasible": not feasible.empty,
            "best_scenario": None,
            "best_safety_stock": np.nan,
            "best_shortage_pct": np.nan,
            "best_total_cost": np.nan,
        }

        if not feasible.empty:

            best = feasible.sort_values(
                ["total_cost", "shortage_rate_pct"]
            ).iloc[0]

            result.update({
                "best_scenario": best["scenario"],
                "best_safety_stock": int(best["safety_stock"]),
                "best_shortage_pct": best["shortage_rate_pct"],
                "best_total_cost": best["total_cost"],
            })

        results.append(result)

    return pd.DataFrame(results)