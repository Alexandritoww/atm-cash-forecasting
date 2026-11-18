from pathlib import Path

from src.simulation.synthetic_demand import (
    SyntheticDemandConfig,
    generate_synthetic_demand,
)

from src.models.train_multioutput import (
    train_multioutput,
)

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

INPUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "atm_daily_demand.csv"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "atm_denomination_demand.csv"
)

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "random_forest_multioutput.joblib"
)


def main():

    print("1. Cargando demanda diaria...")

    df = pd.read_csv(INPUT_PATH)

    print(f"Registros: {len(df)}")

    print("\n2. Generando denominaciones sintéticas...")

    config = SyntheticDemandConfig(
        units_to_cop=1000,
        seed=42,
    )

    synthetic = generate_synthetic_demand(
        df,
        config=config,
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    synthetic.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(f"Dataset generado: {OUTPUT_PATH}")

    print("\n3. Entrenando modelo multisalida...")

    train_multioutput(
        data_path=OUTPUT_PATH,
        model_path=MODEL_PATH,
    )

    print("\nProceso completado correctamente.")


if __name__ == "__main__":
    main()