from pathlib import Path

from src.models.evaluate_models import evaluate_models


BASE_DIR = Path(__file__).resolve().parent


def main():

    print("Iniciando evaluación comparativa...")

    evaluate_models(
        data_path=(
            BASE_DIR
            / "data"
            / "processed"
            / "atm_denomination_demand.csv"
        ),
        rf_path=(
            BASE_DIR
            / "models"
            / "random_forest_multioutput.joblib"
        ),
        nn_path=(
            BASE_DIR
            / "models"
            / "multitask_atm_network.pt"
        ),
        output_path=(
            BASE_DIR
            / "data"
            / "processed"
            / "model_comparison.csv"
        ),
    )

    print("\nEvaluación completada.")


if __name__ == "__main__":
    main()