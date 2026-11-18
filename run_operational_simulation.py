from pathlib import Path

from src.simulation.operational_simulator import simulate_replenishment


BASE_DIR = Path(__file__).resolve().parent


def main():
    data_path = (
        BASE_DIR / "data" / "processed"
        / "atm_denomination_demand.csv"
    )

    model_path = (
        BASE_DIR / "models"
        / "random_forest_multioutput.joblib"
    )

    results = simulate_replenishment(
        data_path=data_path,
        model_path=model_path,
        safety_stock=20,
        cassette_capacity=2000,
        initial_stock=100,
    )

    output_path = (
        BASE_DIR / "data" / "processed"
        / "operational_simulation.csv"
    )

    results.to_csv(output_path, index=False)

    summary = results.groupby("scenario").agg(
        total_replenished_cop=("replenished_cop", "sum"),
        total_unmet_cop=("unmet_cop", "sum"),
        total_remaining_cop=("remaining_cop", "sum"),
        total_unmet_notes=("unmet_notes", "sum"),
        atms_with_shortage=("has_shortage", "sum"),
    )

    print("\n===== SIMULACIÓN OPERATIVA =====")
    print(summary.to_string())
    print(f"\nResultados guardados en: {output_path}")


if __name__ == "__main__":
    main()