from pathlib import Path

import matplotlib.pyplot as plt

from src.simulation.economic_evaluation import (
    EconomicConfig,
    evaluate_economic_costs,
)


BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "safety_stock_optimization.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "reports"
    / "figures"
)

RESULTS_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "economic_evaluation.csv"
)


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = EconomicConfig(
        shortage_rate=0.10,
        holding_rate=0.001,
        replenishment_rate=0.002,
    )

    results = evaluate_economic_costs(
        data_path=DATA_PATH,
        config=config,
    )

    results.to_csv(
        RESULTS_PATH,
        index=False,
    )

    print("\n========== EVALUACIÓN ECONÓMICA ==========")

    print(
        results[
            [
                "scenario",
                "safety_stock",
                "shortage_rate_pct",
                "cost_shortage",
                "cost_holding",
                "cost_replenishment",
                "total_cost",
            ]
        ].to_string(index=False)
    )

    # Gráfico 1: Costo total frente al stock.
    plt.figure(figsize=(10, 6))

    for scenario, group in results.groupby("scenario"):
        group = group.sort_values("safety_stock")

        plt.plot(
            group["safety_stock"],
            group["total_cost"] / 1_000_000,
            marker="o",
            label=scenario,
        )

    plt.title("Costo total por política de abastecimiento")
    plt.xlabel("Stock de seguridad (billetes)")
    plt.ylabel("Costo hipotético (millones COP)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "economic_cost_comparison.png",
        dpi=200,
    )
    plt.close()

    # Gráfico 2: Faltantes frente al inventario restante.
    plt.figure(figsize=(10, 6))

    for scenario, group in results.groupby("scenario"):
        plt.scatter(
            group["total_remaining_cop"] / 1_000_000,
            group["shortage_rate_pct"],
            label=scenario,
            s=65,
        )

    plt.title("Relación entre inventario y faltantes")
    plt.xlabel("Efectivo restante (millones COP sintéticos)")
    plt.ylabel("ATM con faltantes (%)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "inventory_shortage_tradeoff.png",
        dpi=200,
    )
    plt.close()

    # Mejor configuración entre las evaluadas.
    best = results.iloc[0]

    print("\n========== MEJOR CONFIGURACIÓN ==========")
    print(f"Política: {best['scenario']}")
    print(f"Stock de seguridad: {best['safety_stock']}")
    print(f"Costo total: {best['total_cost']:,.2f} COP sintéticos")
    print(
        f"ATM con faltantes: "
        f"{best['shortage_rate_pct']:.2f}%"
    )

    print(f"\nResultados guardados en: {RESULTS_PATH}")
    print(f"Gráficos guardados en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()