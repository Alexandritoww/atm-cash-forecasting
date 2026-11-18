from pathlib import Path
import argparse
import subprocess
import sys
import time


BASE_DIR = Path(__file__).resolve().parent

# Orden provisional basado en las etapas del proyecto.
# Debe validarse con las dependencias de cada script.
PIPELINE = [
    ("Preparación de datos", "run_data_pipeline.py"),
    ("Entrenamiento baseline", "run_training.py"),
    ("Entrenamiento multisalida", "run_multioutput.py"),
    ("Entrenamiento multitarea", "run_multitask.py"),
    ("Auditoría temporal", "run_temporal_audit.py"),
    ("Validación de modelos", "run_model_validation.py"),
    ("Comparación de modelos", "run_evaluation.py"),
    ("Simulación operativa", "run_operational_simulation.py"),
    ("Optimización de stock", "run_safety_optimization.py"),
    ("Evaluación económica", "run_economic_evaluation.py"),
    ("Análisis de sensibilidad", "run_sensitivity_analysis.py"),
    ("Selección de políticas", "run_policy_selection.py"),
    ("Análisis de nivel de servicio", "run_service_level_analysis.py"),
]


def validate_scripts():
    """Comprueba que todos los scripts existan."""

    missing = [
        filename
        for _, filename in PIPELINE
        if not (BASE_DIR / filename).is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "Faltan los siguientes scripts:\n"
            + "\n".join(missing)
        )


def run_pipeline(dry_run=True):
    """Ejecuta las etapas en orden."""

    validate_scripts()

    print("\n========== ATM CASH FORECASTING ==========")

    if dry_run:
        print("MODO VALIDACIÓN: no se ejecutarán scripts.")
    else:
        print("MODO EJECUCIÓN COMPLETA")

    start = time.perf_counter()

    for number, (name, filename) in enumerate(
        PIPELINE, start=1
    ):
        print(
            f"\n[{number}/{len(PIPELINE)}] "
            f"{name} - {filename}",
            flush=True,
        )

        if dry_run:
            continue

        step_start = time.perf_counter()

        result = subprocess.run(
            [sys.executable, str(BASE_DIR / filename)],
            cwd=BASE_DIR,
            check=False,
        )

        elapsed = time.perf_counter() - step_start

        if result.returncode != 0:
            print(
                f"\nERROR en {filename}. "
                f"Código: {result.returncode}"
            )
            print(
                "Pipeline detenido para evitar "
                "continuar con resultados inválidos."
            )
            return result.returncode

        print(
            f"Etapa completada en {elapsed:.2f} segundos."
        )

    total_time = time.perf_counter() - start

    if dry_run:
        print("\nValidación completada.")
        print("No se modificaron archivos.")
    else:
        print("\nPIPELINE COMPLETADO CORRECTAMENTE.")
        print(f"Tiempo total: {total_time:.2f} segundos.")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline de ATM Cash Forecasting"
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help="Ejecuta los scripts. Sin esta opción solo valida.",
    )

    args = parser.parse_args()

    return run_pipeline(
        dry_run=not args.execute
    )


if __name__ == "__main__":
    sys.exit(main())