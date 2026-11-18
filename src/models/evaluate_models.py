from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)

from src.models.multitask_network import MultiTaskATMNetwork
from src.simulation.synthetic_demand import TARGET_COLUMNS


DENOMINATIONS = np.array([
    10_000,
    20_000,
    50_000,
    100_000,
])


def evaluate_models(
    data_path: str | Path,
    rf_path: str | Path,
    nn_path: str | Path,
    output_path: str | Path,
):
    """
    Compara tres modelos sobre la misma fecha de evaluación.

    Modelos:
    - Baseline ingenuo.
    - Random Forest multisalida.
    - Red neuronal multitarea.
    """

    # ========================================================
    # 1. CARGAR DATOS
    # ========================================================

    df = pd.read_csv(
        data_path,
        parse_dates=["date"],
    )

    df = df.sort_values(
        ["atmId", "date"]
    ).reset_index(drop=True)

    test_date = df["date"].max()

    test_df = df[
        df["date"] == test_date
    ].copy()

    if test_df.empty:
        raise ValueError(
            "No existen registros para evaluación."
        )

    y_true = test_df[TARGET_COLUMNS].to_numpy()

    # ========================================================
    # 2. BASELINE INGENUO
    # ========================================================

    lagged = df.copy()

    for column in TARGET_COLUMNS:
        lagged[f"{column}_lag_1"] = (
            lagged.groupby("atmId")[column].shift(1)
        )

    naive_df = lagged[
        lagged["date"] == test_date
    ].copy()

    lag_columns = [
        f"{column}_lag_1"
        for column in TARGET_COLUMNS
    ]

    if naive_df[lag_columns].isna().any().any():
        raise ValueError(
            "No hay historial suficiente para el baseline."
        )

    naive_predictions = naive_df[
        lag_columns
    ].to_numpy()

    # ========================================================
    # 3. RANDOM FOREST
    # ========================================================

    rf_bundle = joblib.load(rf_path)

    rf_model = rf_bundle["model"]
    feature_columns = rf_bundle["features"]

    X_test = test_df[feature_columns]

    rf_predictions = rf_model.predict(X_test)

    rf_predictions = np.maximum(
        np.rint(rf_predictions),
        0,
    ).astype(int)

    # ========================================================
    # 4. RED NEURONAL MULTITAREA
    # ========================================================

    checkpoint = torch.load(
        nn_path,
        map_location="cpu",
        weights_only=True,
    )

    scaler_path = Path(nn_path).with_suffix(
        ".scalers.joblib"
    )

    scalers = joblib.load(scaler_path)

    x_scaler = scalers["x_scaler"]
    y_scaler = scalers["y_scaler"]

    nn_features = checkpoint["features"]

    X_nn = test_df[nn_features].to_numpy(
        dtype=np.float32
    )

    X_nn_scaled = x_scaler.transform(X_nn)

    model = MultiTaskATMNetwork(
        input_dim=checkpoint["input_dim"]
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    with torch.no_grad():

        nn_scaled_predictions = model(
            torch.tensor(
                X_nn_scaled,
                dtype=torch.float32,
            )
        ).numpy()

    nn_predictions = y_scaler.inverse_transform(
        nn_scaled_predictions
    )

    nn_predictions = np.maximum(
        np.rint(nn_predictions),
        0,
    ).astype(int)

    # ========================================================
    # 5. CALCULAR MÉTRICAS
    # ========================================================

    predictions = {
        "Naive": naive_predictions,
        "Random Forest": rf_predictions,
        "MultiTask Neural Network": nn_predictions,
    }

    results = []

    for model_name, y_pred in predictions.items():

        for index, target in enumerate(TARGET_COLUMNS):

            mae = mean_absolute_error(
                y_true[:, index],
                y_pred[:, index],
            )

            rmse = np.sqrt(
                mean_squared_error(
                    y_true[:, index],
                    y_pred[:, index],
                )
            )

            results.append({
                "model": model_name,
                "denomination": target,
                "MAE_notes": mae,
                "RMSE_notes": rmse,
            })

        actual_cash = (
            y_true * DENOMINATIONS
        ).sum(axis=1)

        predicted_cash = (
            y_pred * DENOMINATIONS
        ).sum(axis=1)

        cash_mae = mean_absolute_error(
            actual_cash,
            predicted_cash,
        )

        results.append({
            "model": model_name,
            "denomination": "TOTAL_CASH",
            "MAE_notes": np.nan,
            "RMSE_notes": np.nan,
            "MAE_COP": cash_mae,
        })

    results_df = pd.DataFrame(results)

    # ========================================================
    # 6. GUARDAR RESULTADOS
    # ========================================================

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print("\n========== COMPARACIÓN DE MODELOS ==========")

    print(
        results_df.to_string(
            index=False
        )
    )

    print(f"\nResultados guardados en: {output_path}")

    return results_df