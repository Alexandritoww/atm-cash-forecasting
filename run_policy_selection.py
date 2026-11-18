from pathlib import Path

from src.simulation.policy_selection import (
    select_best_policy,
)


BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "economic_evaluation.csv"
)


def main():

    print("\n===== SELECCIÓN DE POLÍTICAS =====")

    # Evaluar diferentes objetivos de servicio.
    service_limits = [5, 10, 20, 30, 50]

    for limit in service_limits:

        print(
            f"\n--- Máximo permitido: {limit}% ---"
        )

        result = select_best_policy(
            data_path=DATA_PATH,
            max_shortage_pct=limit,
        )

        if result["feasible"]:

            best = result["best_policy"]

            print("Estado: VIABLE")
            print(f"Política: {best['scenario']}")
            print(
                f"Stock de seguridad: "
                f"{int(best['safety_stock'])}"
            )
            print(
                f"ATM con faltantes: "
                f"{best['shortage_rate_pct']:.2f}%"
            )
            print(
                f"Costo total: "
                f"{best['total_cost']:,.2f}"
            )
            print(
                f"Alternativas viables: "
                f"{result['feasible_count']}"
            )

        else:

            closest = result["closest_policy"]

            print("Estado: NO VIABLE")
            print(
                "Ninguna configuración cumple "
                "el objetivo solicitado."
            )

            print(
                f"Menor porcentaje encontrado: "
                f"{closest['shortage_rate_pct']:.2f}%"
            )

            print(
                f"Política más cercana: "
                f"{closest['scenario']}"
            )


if __name__ == "__main__":
    main()