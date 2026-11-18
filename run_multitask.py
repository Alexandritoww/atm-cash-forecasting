from pathlib import Path

from src.models.train_multitask import train_multitask


BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "atm_denomination_demand.csv"
)

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "multitask_atm_network.pt"
)


def main():
    print("Iniciando entrenamiento de red multitarea...")

    train_multitask(
        data_path=DATA_PATH,
        model_path=MODEL_PATH,
        epochs=150,
    )

    print("\nEntrenamiento completado.")


if __name__ == "__main__":
    main()