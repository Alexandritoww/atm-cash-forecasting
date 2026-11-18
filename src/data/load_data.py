from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "atmId",
    "transactionTime",
    "totalOutcome",
    "totalIncome",
    "numberOutcomeTransaction",
    "numberIncomeTransaction",
]


def load_atm_data(file_path: str | Path) -> pd.DataFrame:
    """
    Carga y valida los datos originales de los ATM.
    """

    file_path = Path(file_path)

    if not file_path.is_file():
        raise FileNotFoundError(
            f"No se encontró el archivo: {file_path}"
        )

    df = pd.read_csv(file_path)

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Columnas faltantes: {missing_columns}"
        )

    if df.empty:
        raise ValueError(
            "El dataset está vacío."
        )

    return df