from pathlib import Path

from src.models.model_validation import (
    validate_temporal_split,
)


BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "atm_denomination_demand.csv"
)


def main():

    print("\n===== VALIDACIÓN TEMPORAL =====")

    result = validate_temporal_split(
        data_path=DATA_PATH,
    )

    for key, value in result.items():
        print(f"{key}: {value}")

    if result["valid_temporal_split"]:
        print("\nPartición temporal propuesta: VÁLIDA")
    else:
        print("\nPartición temporal propuesta: INVÁLIDA")


if __name__ == "__main__":
    main()