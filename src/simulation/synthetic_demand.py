from dataclasses import dataclass
import hashlib

import numpy as np
import pandas as pd


DENOMINATIONS = (10_000, 20_000, 50_000, 100_000)

TARGET_COLUMNS = [
    "notes_10000",
    "notes_20000",
    "notes_50000",
    "notes_100000",
]


@dataclass(frozen=True)
class SyntheticDemandConfig:
    """
    Configuración del generador sintético.

    units_to_cop:
        Factor hipotético de conversión a COP.

    seed:
        Semilla para reproducibilidad.
    """

    units_to_cop: int = 1000
    seed: int = 42

    def __post_init__(self):

        if (
            isinstance(self.units_to_cop, bool)
            or not isinstance(self.units_to_cop, int)
            or self.units_to_cop <= 0
        ):
            raise ValueError(
                "units_to_cop debe ser un entero positivo."
            )

        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError(
                "seed debe ser un entero no negativo."
            )


def allocate_notes(
    amount_cop: int,
    rng: np.random.Generator,
) -> dict[int, int]:
    """
    Distribuye un monto sintético entre denominaciones.

    Garantiza la conservación monetaria.
    No representa restricciones reales de inventario.
    """

    if (
        isinstance(amount_cop, bool)
        or not isinstance(amount_cop, int)
        or amount_cop < 0
        or amount_cop % 10_000 != 0
    ):
        raise ValueError(
            "El monto debe ser no negativo y múltiplo de 10.000."
        )

    remaining = amount_cop

    allocation = {
        denomination: 0
        for denomination in DENOMINATIONS
    }

    for denomination in (
        100_000,
        50_000,
        20_000,
    ):

        max_notes = remaining // denomination

        quantity = (
            int(rng.integers(0, max_notes + 1))
            if max_notes > 0
            else 0
        )

        allocation[denomination] = quantity

        remaining -= quantity * denomination

    allocation[10_000] = remaining // 10_000

    return allocation


def _stable_seed(
    atm_id: str,
    date: str,
    seed: int,
) -> int:
    """
    Genera una semilla estable a partir de
    ATM, fecha y semilla global.

    No depende del orden de las filas.
    """

    key = f"{seed}|{atm_id}|{date}"

    digest = hashlib.sha256(
        key.encode("utf-8")
    ).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="big",
    )


def generate_synthetic_demand(
    df: pd.DataFrame,
    config: SyntheticDemandConfig | None = None,
) -> pd.DataFrame:
    """
    Genera demanda sintética por denominación
    de forma reproducible por ATM y fecha.
    """

    if config is None:
        config = SyntheticDemandConfig()

    data = df.copy()

    required_columns = [
        "atmId",
        "date",
        "daily_outcome",
    ]

    missing = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Faltan columnas: {missing}"
        )

    if data["atmId"].isna().any():
        raise ValueError(
            "Existen identificadores ATM vacíos."
        )

    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce",
    )

    if data["date"].isna().any():
        raise ValueError(
            "Existen fechas inválidas."
        )

    if data.duplicated(
        subset=["atmId", "date"]
    ).any():
        raise ValueError(
            "Existen registros duplicados por ATM y fecha."
        )

    values = pd.to_numeric(
        data["daily_outcome"],
        errors="coerce",
    )

    if (
        values.isna().any()
        or not np.isfinite(
            values.to_numpy(dtype=float)
        ).all()
        or (values < 0).any()
    ):
        raise ValueError(
            "daily_outcome contiene valores inválidos."
        )

    # Convertir a montos hipotéticos en COP.
    scaled_amounts = (
        values.to_numpy(dtype=float)
        * config.units_to_cop
        / 10_000
    )

    if (
        not np.isfinite(scaled_amounts).all()
        or (scaled_amounts > np.iinfo(np.int64).max / 10_000).any()
    ):
        raise ValueError(
            "Los montos sintéticos exceden el rango permitido."
        )

    data["synthetic_amount_cop"] = (
        np.rint(scaled_amounts).astype(np.int64)
        * 10_000
    )

    allocations = []

    for row in data.itertuples(index=False):

        atm_id = str(row.atmId)

        date = pd.Timestamp(
            row.date
        ).strftime("%Y-%m-%d")

        amount = int(
            row.synthetic_amount_cop
        )

        row_seed = _stable_seed(
            atm_id=atm_id,
            date=date,
            seed=config.seed,
        )

        rng = np.random.default_rng(
            row_seed
        )

        allocation = allocate_notes(
            amount_cop=amount,
            rng=rng,
        )

        allocations.append(allocation)

    for denomination in DENOMINATIONS:

        data[f"notes_{denomination}"] = [
            allocation[denomination]
            for allocation in allocations
        ]

    reconstructed = sum(
        data[f"notes_{denomination}"]
        * denomination
        for denomination in DENOMINATIONS
    )

    if not (
        reconstructed
        == data["synthetic_amount_cop"]
    ).all():
        raise RuntimeError(
            "La asignación no conserva el monto."
        )

    return data