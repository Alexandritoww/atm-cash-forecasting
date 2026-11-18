import pandas as pd


def create_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construye variables temporales para predecir
    la demanda diaria de efectivo por ATM.

    Todas las variables de demanda histórica
    utilizan información anterior al día objetivo.
    """

    data = df.copy()

    data["date"] = pd.to_datetime(data["date"])

    data = data.sort_values(
        ["atmId", "date"]
    ).reset_index(drop=True)

    # Variables de calendario.
    data["day_of_week"] = data["date"].dt.dayofweek
    data["day_of_month"] = data["date"].dt.day
    data["is_weekend"] = (
        data["day_of_week"] >= 5
    ).astype(int)

    # Agrupar por ATM.
    grouped = data.groupby("atmId")["daily_outcome"]

    # Demanda del día anterior.
    data["outcome_lag_1"] = grouped.shift(1)

    # Demanda de dos días atrás.
    data["outcome_lag_2"] = grouped.shift(2)

    # Promedio de los últimos tres días anteriores.
    data["outcome_rolling_mean_3"] = (
        data.groupby("atmId")["daily_outcome"]
        .transform(
            lambda series: (
                series.shift(1)
                .rolling(window=3, min_periods=3)
                .mean()
            )
        )
    )

    # Conservar únicamente filas con historial suficiente.
    data = data.dropna(
        subset=[
            "outcome_lag_1",
            "outcome_lag_2",
            "outcome_rolling_mean_3",
        ]
    ).reset_index(drop=True)

    return data