from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)


FEATURE_COLUMNS = [
    "day_of_week",
    "day_of_month",
    "is_weekend",
    "outcome_lag_1",
    "outcome_lag_2",
    "outcome_rolling_mean_3",
]

TARGET_COLUMN = "daily_outcome"


def train_baseline(
    data_path: str | Path,
    model_path: str | Path,
):
    """
    Entrena y evalúa un Random Forest.

    Compara sus predicciones con un modelo ingenuo,
    utilizando una separación cronológica global.
    """

    df = pd.read_csv(
        data_path,
        parse_dates=["date"],
    )

    df = df.sort_values(
        ["date", "atmId"]
    ).reset_index(drop=True)

    # Separación temporal:
    # entrenamos con fechas anteriores y evaluamos
    # con la última fecha disponible.
    unique_dates = sorted(df["date"].unique())

    if len(unique_dates) < 2:
        raise ValueError(
            "Se necesitan al menos dos fechas "
            "para entrenamiento y evaluación."
        )

    test_date = unique_dates[-1]

    train_df = df[df["date"] < test_date].copy()
    test_df = df[df["date"] == test_date].copy()

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    # --------------------------------------------------------
    # MODELO 1: BASELINE INGENUO
    # --------------------------------------------------------

    naive_predictions = test_df[
        "outcome_lag_1"
    ].to_numpy()

    naive_mae = mean_absolute_error(
        y_test,
        naive_predictions,
    )

    naive_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            naive_predictions,
        )
    )

    # --------------------------------------------------------
    # MODELO 2: RANDOM FOREST
    # --------------------------------------------------------

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(X_test)

    model_mae = mean_absolute_error(
        y_test,
        predictions,
    )

    model_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions,
        )
    )

    # --------------------------------------------------------
    # GUARDAR MODELO
    # --------------------------------------------------------

    model_path = Path(model_path)

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        {
            "model": model,
            "features": FEATURE_COLUMNS,
            "target": TARGET_COLUMN,
        },
        model_path,
    )

    # --------------------------------------------------------
    # MOSTRAR RESULTADOS
    # --------------------------------------------------------

    print("\n========== RESULTADOS ==========")

    print(f"Fecha de evaluación: {pd.Timestamp(test_date).date()}")
    print(f"Registros de entrenamiento: {len(train_df)}")
    print(f"Registros de evaluación: {len(test_df)}")

    print("\nModelo ingenuo:")
    print(f"MAE: {naive_mae:.2f}")
    print(f"RMSE: {naive_rmse:.2f}")

    print("\nRandom Forest:")
    print(f"MAE: {model_mae:.2f}")
    print(f"RMSE: {model_rmse:.2f}")

    improvement = (
        (naive_mae - model_mae) / naive_mae * 100
        if naive_mae > 0
        else float("nan")
    )

    print(f"\nMejora relativa en MAE: {improvement:.2f}%")

    print(f"\nModelo guardado en: {model_path}")

    return model