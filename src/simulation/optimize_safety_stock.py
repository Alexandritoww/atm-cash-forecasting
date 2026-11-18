from pathlib import Path

import pandas as pd

from src.simulation.operational_simulator import (
    simulate_replenishment,
)


def optimize_safety_stock(
    data_path: str | Path,
    model_path: str | Path,
    safety_levels=(0, 20, 50, 100, 200, 500),
    cassette_capacity=2000,
    initial_stock=100,
):
    """
    Evalúa diferentes niveles de stock de seguridad.

    Utiliza los mismos datos, modelos e inventario
    inicial para garantizar comparaciones consistentes.
    """

    results = []

    for safety_stock in safety_levels:

        print(
            f"\nEvaluando stock de seguridad: "
            f"{safety_stock} billetes"
        )

        simulation = simulate_replenishment(
            data_path=data_path,
            model_path=model_path,
            safety_stock=safety_stock,
            cassette_capacity=cassette_capacity,
            initial_stock=initial_stock,
        )

        summary = (
            simulation.groupby("scenario")
            .agg(
                total_replenished_cop=(
                    "replenished_cop", "sum"
                ),
                total_unmet_cop=(
                    "unmet_cop", "sum"
                ),
                total_remaining_cop=(
                    "remaining_cop", "sum"
                ),
                total_unmet_notes=(
                    "unmet_notes", "sum"
                ),
                atms_with_shortage=(
                    "has_shortage", "sum"
                ),
                total_atms=(
                    "atmId", "nunique"
                ),
            )
            .reset_index()
        )

        summary["safety_stock"] = safety_stock

        summary["shortage_rate_pct"] = (
            summary["atms_with_shortage"]
            / summary["total_atms"]
            * 100
        )

        results.append(summary)

    final_results = pd.concat(
        results,
        ignore_index=True,
    )

    return final_results