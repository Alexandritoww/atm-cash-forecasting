from pathlib import Path

import numpy as np
import pandas as pd


def audit_temporal_features(
    data_path: str | Path,
) -> dict:
    """
    Audita la consistencia temporal de las variables
    históricas utilizadas para forecasting.

    Verifica:
    - Fechas válidas.
    - Duplicados por ATM y fecha.
    - Continuidad temporal.
    - Corrección de rezagos.
    - Cálculo de la media móvil histórica.
    """

    df = pd.read_csv(data_path)

    required_columns = [
        "atmId",
        "date",
        "daily_outcome",
        "outcome_lag_1",
        "outcome_lag_2",
        "outcome_rolling_mean_3",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Faltan columnas: {missing}"
        )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    if df["date"].isna().any():
        raise ValueError("Se encontraron fechas inválidas.")

    df = df.sort_values(
        ["atmId", "date"]
    ).reset_index(drop=True)

    duplicated = df.duplicated(
        subset=["atmId", "date"]
    ).sum()

    # Revisar si existen saltos entre fechas.
    date_differences = (
        df.groupby("atmId")["date"]
        .diff()
    )

    gaps = (
        date_differences.notna()
        & (date_differences != pd.Timedelta(days=1))
    ).sum()

    # Los rezagos se calculan sobre las filas
    # disponibles en el dataset auditado.
    expected_lag_1 = (
        df.groupby("atmId")["daily_outcome"]
        .shift(1)
    )

    expected_lag_2 = (
        df.groupby("atmId")["daily_outcome"]
        .shift(2)
    )

    expected_rolling = (
        df.groupby("atmId")["daily_outcome"]
        .transform(
            lambda series: (
                series.shift(1)
                .rolling(window=3)
                .mean()
            )
        )
    )

    def count_mismatches(actual, expected):
        comparable = expected.notna()

        return int(
            (
                ~np.isclose(
                    actual[comparable],
                    expected[comparable],
                    rtol=1e-6,
                    atol=1e-6,
                    equal_nan=True,
                )
            ).sum()
        )

    results = {
        "total_records": len(df),
        "total_atms": df["atmId"].nunique(),
        "min_date": str(df["date"].min().date()),
        "max_date": str(df["date"].max().date()),
        "duplicated_atm_dates": int(duplicated),
        "temporal_gaps": int(gaps),
        "lag_1_mismatches": count_mismatches(
            df["outcome_lag_1"],
            expected_lag_1,
        ),
        "lag_2_mismatches": count_mismatches(
            df["outcome_lag_2"],
            expected_lag_2,
        ),
        "rolling_mean_mismatches": count_mismatches(
            df["outcome_rolling_mean_3"],
            expected_rolling,
        ),
    }

    return results