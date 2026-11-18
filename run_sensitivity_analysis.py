from pathlib import Path

import matplotlib.pyplot as plt

from src.simulation.sensitivity_analysis import (
    run_sensitivity_analysis,
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
    / "sensitivity_analysis.csv"
)

FIGURES_DIR = BASE_DIR / "reports" / "figures"


def main():

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Iniciando análisis de sensibilidad...")

    results = run_sensitivity_analysis(
        data_path=INPUT_PATH,
        max_shortage_pct=30.0,
    )

    results.to_csv(OUTPUT_PATH, index=False)

    print("\n========== RESUMEN ==========")
    print(f"Combinaciones evaluadas: {len(results)}")

    feasible = results[results["feasible"]].copy()

    print(f"Combinaciones viables: {len(feasible)}")

    if feasible.empty:
        print("Ninguna combinación produjo una política viable.")
        return

    print("\nPolíticas seleccionadas:")
    print(
        feasible["best_scenario"]
        .value_counts()
        .to_string()
    )

    print("\nStocks seleccionados:")
    print(
        feasible["best_safety_stock"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # Gráfico: frecuencia de políticas seleccionadas.
    counts = feasible["best_scenario"].value_counts()

    plt.figure(figsize=(8, 5))
    counts.plot(kind="bar")

    plt.title("Frecuencia de selección por política")
    plt.xlabel("Política")
    plt.ylabel("Número de combinaciones económicas")
    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "sensitivity_policy_frequency.png",
        dpi=200,
    )
    plt.close()

    # Gráfico: stock de seguridad seleccionado.
    stock_counts = (
        feasible["best_safety_stock"]
        .value_counts()
        .sort_index()
    )

    plt.figure(figsize=(8, 5))
    stock_counts.plot(kind="bar")

    plt.title("Frecuencia del stock de seguridad seleccionado")
    plt.xlabel("Stock de seguridad (billetes)")
    plt.ylabel("Número de combinaciones económicas")
    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "sensitivity_safety_stock.png",
        dpi=200,
    )
    plt.close()

    print(f"\nResultados guardados en: {OUTPUT_PATH}")
    print(f"Gráficos guardados en: {FIGURES_DIR}")


if __name__ == "__main__":
    main()