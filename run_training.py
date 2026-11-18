from pathlib import Path

from src.models.train_baseline import train_baseline


# Directorio principal del proyecto
BASE_DIR = Path(__file__).resolve().parent

# Dataset procesado
DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "atm_daily_demand.csv"
)

# Ubicación donde guardaremos el modelo
MODEL_PATH = (
    BASE_DIR
    / "models"
    / "random_forest_baseline.joblib"
)


def main():
    print("Iniciando entrenamiento ATM Cash Forecasting...")

    train_baseline(
        data_path=DATA_PATH,
        model_path=MODEL_PATH,
    )

    print("\nEntrenamiento completado.")


if __name__ == "__main__":
    main()
