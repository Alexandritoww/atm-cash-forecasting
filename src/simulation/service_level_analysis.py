from pathlib import Path

import pandas as pd

from src.simulation.sensitivity_analysis import (
    run_sensitivity_analysis,
)


def analyze_service_levels(
    data_path: str | Path,
    service_levels=(30, 40, 60, 80, 100),
) -> pd.DataFrame:
    """
    Compara la selección económica de políticas
    bajo diferentes límites de faltantes.

    Cada límite se evalúa con las mismas
    combinaciones de parámetros económicos.
    """

    if not service_levels:
        raise ValueError(
            "Debe especificarse al menos un nivel de servicio."
        )

    results = []

    for level in service_levels:

        print(
            f"Evaluando límite de faltantes: {level}%"
        )

        sensitivity = run_sensitivity_analysis(
            data_path=data_path,
            max_shortage_pct=level,
        )

        sensitivity["service_limit_pct"] = level

        results.append(sensitivity)

    return pd.concat(
        results,
        ignore_index=True,
    )