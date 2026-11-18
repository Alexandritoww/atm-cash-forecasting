import pandas as pd


NUMERIC_COLUMNS = [
    "totalOutcome",
    "totalIncome",
    "numberOutcomeTransaction",
    "numberIncomeTransaction",
]


def preprocess_atm_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia y agrega los movimientos por ATM y día.

    No convierte los valores monetarios originales
    a pesos colombianos.
    """

    data = df.copy()

    # Convertir la fecha.
    data["transactionTime"] = pd.to_datetime(
        data["transactionTime"],
        errors="coerce",
    )

    # Convertir columnas numéricas.
    for column in NUMERIC_COLUMNS:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    # Eliminar registros incompletos.
    data = data.dropna(
        subset=[
            "atmId",
            "transactionTime",
            *NUMERIC_COLUMNS,
        ]
    )

    # Eliminar valores negativos no válidos.
    for column in NUMERIC_COLUMNS:
        data = data[data[column] >= 0]

    # Crear fecha sin hora.
    data["date"] = data["transactionTime"].dt.normalize()

    # Agregar por cajero y día.
    daily = (
        data.groupby(
            ["atmId", "date"],
            as_index=False,
        )
        .agg(
            daily_outcome=("totalOutcome", "sum"),
            daily_income=("totalIncome", "sum"),
            outcome_transactions=(
                "numberOutcomeTransaction",
                "sum",
            ),
            income_transactions=(
                "numberIncomeTransaction",
                "sum",
            ),
        )
    )

    daily = daily.sort_values(
        ["atmId", "date"]
    ).reset_index(drop=True)

    return daily