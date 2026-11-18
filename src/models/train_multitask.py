from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from torch import nn

from src.models.multitask_network import MultiTaskATMNetwork
from src.models.train_multioutput import FEATURE_COLUMNS
from src.simulation.synthetic_demand import TARGET_COLUMNS


def train_multitask(
    data_path: str | Path,
    model_path: str | Path,
    epochs: int = 150,
):
    """
    Entrena una red neuronal multitarea.

    Utiliza una separación temporal para evaluar
    las predicciones sobre la última fecha disponible.
    """

    torch.manual_seed(42)
    np.random.seed(42)

    df = pd.read_csv(
        data_path,
        parse_dates=["date"],
    )

    df = df.sort_values(
        ["date", "atmId"]
    ).reset_index(drop=True)

    test_date = df["date"].max()

    train_df = df[df["date"] < test_date].copy()
    test_df = df[df["date"] == test_date].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(
            "No hay suficientes datos para entrenar."
        )

    # Variables predictoras.
    X_train = train_df[FEATURE_COLUMNS].to_numpy(
        dtype=np.float32
    )
    X_test = test_df[FEATURE_COLUMNS].to_numpy(
        dtype=np.float32
    )

    # Cuatro variables objetivo.
    y_train = train_df[TARGET_COLUMNS].to_numpy(
        dtype=np.float32
    )
    y_test = test_df[TARGET_COLUMNS].to_numpy(
        dtype=np.float32
    )

    # Normalizar sin utilizar datos de evaluación.
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)

    y_train_scaled = y_scaler.fit_transform(y_train)

    # Convertir a tensores.
    X_tensor = torch.tensor(
        X_train_scaled,
        dtype=torch.float32,
    )

    y_tensor = torch.tensor(
        y_train_scaled,
        dtype=torch.float32,
    )

    X_test_tensor = torch.tensor(
        X_test_scaled,
        dtype=torch.float32,
    )

    # Crear modelo.
    model = MultiTaskATMNetwork(
        input_dim=len(FEATURE_COLUMNS)
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    criterion = nn.MSELoss()

    # Entrenamiento.
    model.train()

    for epoch in range(epochs):

        optimizer.zero_grad()

        predictions = model(X_tensor)

        loss = criterion(
            predictions,
            y_tensor,
        )

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 25 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs} "
                f"- Loss: {loss.item():.6f}"
            )

    # Evaluación.
    model.eval()

    with torch.no_grad():
        predictions_scaled = model(
            X_test_tensor
        ).numpy()

    predictions = y_scaler.inverse_transform(
        predictions_scaled
    )

    predictions = np.maximum(
        np.rint(predictions),
        0,
    ).astype(int)

    # Métricas.
    print("\n========== RED MULTITAREA ==========")
    print(f"Fecha de evaluación: {test_date.date()}")
    print(f"Entrenamiento: {len(train_df)} registros")
    print(f"Evaluación: {len(test_df)} registros")

    results = {}

    for index, column in enumerate(TARGET_COLUMNS):

        mae = mean_absolute_error(
            y_test[:, index],
            predictions[:, index],
        )

        results[column] = mae

        print(
            f"{column}: MAE = {mae:.2f} billetes"
        )

    # Error monetario sintético.
    denominations = np.array([
        10_000,
        20_000,
        50_000,
        100_000,
    ])

    actual_amount = (
        y_test * denominations
    ).sum(axis=1)

    predicted_amount = (
        predictions * denominations
    ).sum(axis=1)

    amount_mae = mean_absolute_error(
        actual_amount,
        predicted_amount,
    )

    print(
        f"\nMAE monetario sintético: "
        f"{amount_mae:,.2f} COP"
    )

    # Guardar modelo y escaladores.
    model_path = Path(model_path)

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": len(FEATURE_COLUMNS),
            "features": FEATURE_COLUMNS,
            "targets": TARGET_COLUMNS,
        },
        model_path,
    )

    joblib.dump(
        {
            "x_scaler": x_scaler,
            "y_scaler": y_scaler,
        },
        model_path.with_suffix(".scalers.joblib"),
    )

    print(f"\nModelo guardado en: {model_path}")

    return model, results