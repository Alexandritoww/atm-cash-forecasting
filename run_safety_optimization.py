from pathlib import Path

from src.simulation.optimize_safety_stock import (
    optimize_safety_stock,
)


BASE_DIR = Path(__file__).resolve().parent


def main():

    data_path = (
        BASE_DIR
        / "data"
        / "processed"
        / "atm_denomination_demand.csv"
    )

    model_path = (
        BASE_DIR
        / "models"
        / "random_forest_multioutput.joblib"
    )

    output_path = (
        BASE_DIR
        / "data"
        / "processed"
        / "safety_stock_optimization.csv"
    )

    print("Iniciando optimización de stock de seguridad...")

    results = optimize_safety_stock(
        data_path=data_path,
        model_path=model_path,
        safety_levels=(0, 20, 50, 100, 200, 500),
        cassette_capacity=2000,
        initial_stock=100,
    )

    results.to_csv(
        output_path,
        index=False,
    )

    print("\n========== RESULTADOS ==========")

    print(
        results[
            [
                "scenario",
                "safety_stock",
                "shortage_rate_pct",
                "total_unmet_cop",
                "total_remaining_cop",
                "total_replenished_cop",
            ]
        ].to_string(index=False)
    )

    print(f"\nResultados guardados en: {output_path}")


if __name__ == "__main__":
    main()