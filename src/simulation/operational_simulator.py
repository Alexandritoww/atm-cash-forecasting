import numpy as np
import pandas as pd
import joblib

from src.models.train_multioutput import FEATURE_COLUMNS
from src.simulation.synthetic_demand import TARGET_COLUMNS
from src.simulation.replenishment_policy import ReplenishmentPolicy


DENOMINATIONS = (10_000, 20_000, 50_000, 100_000)


def simulate_replenishment(
    data_path,
    model_path,
    safety_stock=20,
    cassette_capacity=2000,
    initial_stock=100,
):
    """
    Compara políticas de abastecimiento por denominación.

    Simulación diaria agregada:
    - No modela retiros individuales.
    - No considera rutas ni costos de transporte.
    - Los valores monetarios son COP sintéticos.
    """

    df = pd.read_csv(data_path, parse_dates=["date"])
    df = df.sort_values(["atmId", "date"]).reset_index(drop=True)

    test_date = df["date"].max()
    test_df = df[df["date"] == test_date].copy()

    previous = df.groupby("atmId")[TARGET_COLUMNS].shift(1)
    previous.columns = [f"previous_{c}" for c in TARGET_COLUMNS]

    df = pd.concat([df, previous], axis=1)
    test_df = df[df["date"] == test_date].copy()

    if test_df.empty or test_df[previous.columns].isna().any().any():
        raise ValueError("No hay datos suficientes para la simulación.")

    bundle = joblib.load(model_path)
    model = bundle["model"]

    rf_predictions = np.maximum(
        np.rint(model.predict(test_df[bundle["features"]])),
        0,
    ).astype(int)

    actual = test_df[TARGET_COLUMNS].to_numpy(dtype=int)
    naive = test_df[list(previous.columns)].to_numpy(dtype=int)

    scenarios = {
        "Naive": naive,
        "Random Forest": rf_predictions,
        "Oracle": actual,
    }

    policy = ReplenishmentPolicy(
        safety_stock=safety_stock,
        cassette_capacity=cassette_capacity,
    )

    results = []

    for scenario, predicted_matrix in scenarios.items():
        for i, (_, record) in enumerate(test_df.iterrows()):

            inventory = {
                d: initial_stock for d in DENOMINATIONS
            }

            predicted_demand = {
                d: int(predicted_matrix[i, j])
                for j, d in enumerate(DENOMINATIONS)
            }

            actual_demand = {
                d: int(actual[i, j])
                for j, d in enumerate(DENOMINATIONS)
            }

            replenishment = policy.calculate_replenishment(
                inventory, predicted_demand
            )

            stock_after_replenishment = {
                d: inventory[d] + replenishment[d]
                for d in DENOMINATIONS
            }

            unmet = {
                d: max(0, actual_demand[d] - stock_after_replenishment[d])
                for d in DENOMINATIONS
            }

            remaining = {
                d: max(0, stock_after_replenishment[d] - actual_demand[d])
                for d in DENOMINATIONS
            }

            results.append({
                "atmId": record["atmId"],
                "date": record["date"],
                "scenario": scenario,
                "replenished_cop": sum(
                    d * replenishment[d] for d in DENOMINATIONS
                ),
                "unmet_cop": sum(
                    d * unmet[d] for d in DENOMINATIONS
                ),
                "remaining_cop": sum(
                    d * remaining[d] for d in DENOMINATIONS
                ),
                "unmet_notes": sum(unmet.values()),
                "has_shortage": int(any(unmet.values())),
            })

    return pd.DataFrame(results)