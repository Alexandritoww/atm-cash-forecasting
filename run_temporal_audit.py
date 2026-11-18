from pathlib import Path

from src.models.temporal_audit import (
    audit_temporal_features,
)


BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "atm_daily_demand.csv"
)


def main():

    print("\n========== AUDITORÍA TEMPORAL ==========")

    results = audit_temporal_features(
        data_path=DATA_PATH,
    )

    for key, value in results.items():
        print(f"{key}: {value}")

    print("\nAuditoría diagnóstica completada.")


if __name__ == "__main__":
    main()