from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.simulation.service_level_analysis import (
    analyze_service_levels,
)


BASE_DIR = Path(__file__).resolve().parent

INPUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "safety_stock_optimization.csv"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "service_level_analysis.csv"
)

FIGURES_DIR = BASE_DIR / "reports" / "figures"


def main():

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nIniciando análisis de niveles de servicio...")

    results = analyze_service_levels(
        data_path=INPUT_PATH,
        service_levels=(30, 40, 60, 80, 100),
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\n========== RESUMEN GENERAL ==========")

    print(f"Evaluaciones realizadas: {len(results)}")

    # Mantener solo combinaciones viables.
    feasible = results[
        results["feasible"]
    ].copy()

    print(f"Evaluaciones viables: {len(feasible)}")

    if feasible.empty:
        print("No se encontraron políticas viables.")
        return

    # Frecuencia de selección por nivel de servicio.
    frequency = pd.crosstab(
        feasible["service_limit_pct"],
        feasible["best_scenario"],
    )

    print("\n========== POLÍTICAS SELECCIONADAS ==========")
    print(frequency.to_string())

    # Gráfico 1: Frecuencia de políticas.
    ax = frequency.plot(
        kind="bar",
        figsize=(10, 6),
    )

    ax.set_title(
        "Selección de políticas por límite de faltantes"
    )
    ax.set_xlabel(
        "Máximo permitido de ATM con faltantes (%)"
    )
    ax.set_ylabel(
        "Combinaciones económicas"
    )
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Política")

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "service_level_policy_comparison.png",
        dpi=200,
    )
    plt.close()

    # Gráfico 2: Stock seleccionado.
    stock_frequency = pd.crosstab(
        feasible["service_limit_pct"],
        feasible["best_safety_stock"],
    )

    ax = stock_frequency.plot(
        kind="bar",
        figsize=(10, 6),
    )

    ax.set_title(
        "Stock seleccionado por límite de faltantes"
    )
    ax.set_xlabel(
        "Máximo permitido de ATM con faltantes (%)"
    )
    ax.set_ylabel(
        "Combinaciones económicas"
    )
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Billetes de seguridad")

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "service_level_stock_comparison.png",
        dpi=200,
    )
    plt.close()

    print("\n========== RESUMEN DE STOCK ==========")
    print(stock_frequency.to_string())

    print(f"\nResultados guardados en: {OUTPUT_PATH}")
    print(f"Gráficos guardados en: {FIGURES_DIR}")


if __name__ == "__main__":
    main()