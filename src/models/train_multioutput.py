from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from src.simulation.synthetic_demand import (
    TARGET_COLUMNS,
)


FEATURE_COLUMNS = [
    "day_of_week",
    "day_of_month",
    "is_weekend",
    "outcome_lag_1",
    "outcome_lag_2",
    "outcome_rolling_mean_3",
]

DENOMINATIONS = np.array(
    [10_000, 20_000, 50_000, 100_000],
    dtype=int,
)


def train_multioutput(
    data_path: str | Path,
    model_path: str | Path,
):
    """
    Entrena Random Forest multisalida para estimar
    cantidades de billetes por denominación.

    La evaluación se realiza en la última fecha
    disponible y el entrenamiento utiliza
    exclusivamente fechas anteriores.
    """

    df = pd.read_csv(data_path)

    # 1. Validar columnas.
    required_columns = (
        ["atmId", "date"]
        + FEATURE_COLUMNS
        + list(TARGET_COLUMNS)
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Faltan columnas: {missing_columns}"
        )

    # 2. Validar fechas.
    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    if df["date"].isna().any():
        raise ValueError(
            "Existen fechas inválidas."
        )

    if df["atmId"].isna().any():
        raise ValueError(
            "Existen identificadores ATM vacíos."
        )

    # 3. Verificar duplicados.
    duplicated = df.duplicated(
        subset=["atmId", "date"]
    )

    if duplicated.any():
        raise ValueError(
            "Existen registros duplicados por ATM y fecha."
        )

    # 4. Validar variables y objetivos.
    numeric_columns = (
        FEATURE_COLUMNS
        + list(TARGET_COLUMNS)
    )

    numeric_data = df[numeric_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if not np.isfinite(
        numeric_data.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Existen valores faltantes o no finitos."
        )

    df[numeric_columns] = numeric_data

    targets = df[list(TARGET_COLUMNS)].to_numpy(
        dtype=float
    )

    if (targets < 0).any():
        raise ValueError(
            "Las cantidades de billetes no pueden ser negativas."
        )

    if not np.allclose(
        targets,
        np.rint(targets),
    ):
        raise ValueError(
            "Las cantidades de billetes deben ser enteras."
        )

    if len(TARGET_COLUMNS) != len(DENOMINATIONS):
        raise ValueError(
            "Las denominaciones no coinciden con los objetivos."
        )

    # 5. Ordenar cronológicamente.
    df = df.sort_values(
        ["date", "atmId"]
    ).reset_index(drop=True)

    # 6. División temporal.
    test_date = df["date"].max()

    train_df = df[
        df["date"] < test_date
    ].copy()

    test_df = df[
        df["date"] == test_date
    ].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(
            "No hay suficientes fechas para entrenar y evaluar."
        )

    train_max_date = train_df["date"].max()
    test_min_date = test_df["date"].min()

    if train_max_date >= test_min_date:
        raise ValueError(
            "Se detectó superposición temporal."
        )

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[list(TARGET_COLUMNS)]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[list(TARGET_COLUMNS)]

    # 7. Entrenar Random Forest.
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    # 8. Predecir.
    predictions = model.predict(X_test)

    predictions = np.maximum(
        np.rint(predictions),
        0,
    ).astype(int)

    print("\n========== MODELO MULTISALIDA ==========")

    print(
        f"Última fecha entrenamiento: "
        f"{train_max_date.date()}"
    )

    print(
        f"Fecha evaluación: "
        f"{test_date.date()}"
    )

    print(
        f"Registros entrenamiento: "
        f"{len(train_df)}"
    )

    print(
        f"Registros evaluación: "
        f"{len(test_df)}"
    )

    # 9. Evaluar por denominación.
    results = {}

    for index, column in enumerate(TARGET_COLUMNS):

        mae = mean_absolute_error(
            y_test[column],
            predictions[:, index],
        )

        results[column] = float(mae)

        print(
            f"{column} - MAE: "
            f"{mae:.2f} billetes"
        )

    # 10. Evaluación monetaria sintética.
    actual_amount = (
        y_test.to_numpy(dtype=int)
        * DENOMINATIONS
    ).sum(axis=1)

    predicted_amount = (
        predictions * DENOMINATIONS
    ).sum(axis=1)

    amount_mae = mean_absolute_error(
        actual_amount,
        predicted_amount,
    )

    print(
        f"\nMAE monetario sintético: "
        f"{amount_mae:,.2f} COP"
    )

    # 11. Guardar modelo y metadatos.
    model_path = Path(model_path)

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        {
            "model": model,
            "features": FEATURE_COLUMNS,
            "targets": TARGET_COLUMNS,
            "train_max_date": str(
                train_max_date.date()
            ),
            "test_date": str(
                test_date.date()
            ),
            "denominations": DENOMINATIONS.tolist(),
        },
        model_path,
    )

    print(
        f"\nModelo guardado en: "
        f"{model_path}"
    )

    return model, results