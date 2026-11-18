from pathlib import Path

import pandas as pd


def validate_temporal_split(
    data_path: str | Path,
    date_column: str = "date",
    test_date: str | None = None,
) -> dict:
    """
    Verifica que el conjunto de entrenamiento
    contenga únicamente fechas anteriores
    al conjunto de evaluación.
    """

    df = pd.read_csv(data_path)

    if date_column not in df.columns:
        raise ValueError(
            f"No existe la columna {date_column}"
        )

    df[date_column] = pd.to_datetime(
        df[date_column],
        errors="coerce",
    )

    if df[date_column].isna().any():
        raise ValueError(
            "Existen fechas inválidas."
        )

    if test_date is None:
        cutoff = df[date_column].max()
    else:
        cutoff = pd.Timestamp(test_date)

    train = df[df[date_column] < cutoff]
    test = df[df[date_column] == cutoff]

    if train.empty:
        raise ValueError(
            "No hay registros de entrenamiento."
        )

    if test.empty:
        raise ValueError(
            "No hay registros de evaluación."
        )

    train_max = train[date_column].max()
    test_min = test[date_column].min()

    valid = train_max < test_min

    return {
        "valid_temporal_split": bool(valid),
        "train_records": len(train),
        "test_records": len(test),
        "train_max_date": str(train_max.date()),
        "test_min_date": str(test_min.date()),
        "train_unique_dates": train[date_column].nunique(),
        "test_unique_dates": test[date_column].nunique(),
    }