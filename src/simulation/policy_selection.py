from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "scenario",
    "safety_stock",
    "shortage_rate_pct",
    "total_cost",
]


def select_best_policy(
    data_path: str | Path,
    max_shortage_pct: float = 10.0,
) -> dict:
    """
    Selecciona la política de menor costo que cumple
    una restricción máxima de ATM con faltantes.

    Si ninguna política cumple, devuelve un resultado
    explícito de inviabilidad.
    """

    if (
        isinstance(max_shortage_pct, bool)
        or not isinstance(max_shortage_pct, (int, float))
        or not np.isfinite(max_shortage_pct)
        or not 0 <= max_shortage_pct <= 100
    ):
        raise ValueError(
            "max_shortage_pct debe estar entre 0 y 100."
        )

    df = pd.read_csv(data_path)

    missing = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Faltan columnas: {missing}"
        )

    if df.empty:
        raise ValueError("El archivo no contiene escenarios.")

    for column in ("shortage_rate_pct", "total_cost"):
        values = pd.to_numeric(df[column], errors="coerce")

        if (
            not np.isfinite(values.to_numpy()).all()
            or (values < 0).any()
        ):
            raise ValueError(
                f"Valores inválidos en {column}."
            )

        df[column] = values

    if (df["shortage_rate_pct"] > 100).any():
        raise ValueError(
            "El porcentaje de faltantes no puede superar 100."
        )

    feasible = df[
        df["shortage_rate_pct"] <= max_shortage_pct
    ].copy()

    if feasible.empty:
        closest = df.sort_values(
            ["shortage_rate_pct", "total_cost"]
        ).iloc[0]

        return {
            "feasible": False,
            "max_shortage_pct": max_shortage_pct,
            "message": (
                "Ninguna política cumple el nivel "
                "de servicio solicitado."
            ),
            "closest_policy": closest.to_dict(),
        }

    best = feasible.sort_values(
        ["total_cost", "shortage_rate_pct"]
    ).iloc[0]

    return {
        "feasible": True,
        "max_shortage_pct": max_shortage_pct,
        "message": "Política viable encontrada.",
        "best_policy": best.to_dict(),
        "feasible_count": len(feasible),
    }