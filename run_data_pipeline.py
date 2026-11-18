from pathlib import Path

from src.data.load_data import load_atm_data
from src.data.preprocess import preprocess_atm_data
from src.data.feature_engineering import create_features


# ============================================================
# 1. CONFIGURACIÓN DE RUTAS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

RAW_PATH = BASE_DIR / "data" / "raw" / "atm_transactions.csv"

OUTPUT_PATH = (
    BASE_DIR / "data" / "processed" / "atm_daily_demand.csv"
)


# ============================================================
# 2. PIPELINE PRINCIPAL
# ============================================================

def main():

    print("\n========== ATM CASH FORECASTING ==========")

    # PASO 1: Cargar datos originales
    print("\n1. Cargando dataset...")

    df = load_atm_data(RAW_PATH)

    print(f"Registros originales: {len(df)}")

    # PASO 2: Procesar y agregar datos diarios
    print("\n2. Procesando datos...")

    daily = preprocess_atm_data(df)

    print(f"Registros diarios: {len(daily)}")

    # PASO 3: Crear variables predictivas
    print("\n3. Creando variables predictivas...")

    features = create_features(daily)

    print(f"Registros con historial: {len(features)}")

    # PASO 4: Guardar dataset procesado
    print("\n4. Guardando dataset procesado...")

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    features.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(f"Archivo generado: {OUTPUT_PATH}")

    # PASO 5: Mostrar resultados
    print("\nPrimeros registros:")

    print(features.head())

    print("\nPipeline completado correctamente.")


if __name__ == "__main__":
    main()