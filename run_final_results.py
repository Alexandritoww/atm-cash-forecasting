from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
REPORT_DIR = BASE_DIR / "reports"

FILES = [
    "model_comparison.csv",
    "operational_simulation.csv",
    "safety_stock_optimization.csv",
    "service_level_analysis.csv",
]

def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    summary = []

    print("\n===== RESULTADOS FINALES ATM =====")

    for filename in FILES:
        path = DATA_DIR / filename

        if not path.exists():
            print(f"No disponible: {filename}")
            continue

        df = pd.read_csv(path)

        print(f"\nArchivo: {filename}")
        print(f"Registros: {len(df)}")
        print(df.head(5).to_string(index=False))

        summary.append({
            "archivo": filename,
            "registros": len(df),
            "columnas": len(df.columns),
        })

    if summary:
        output = REPORT_DIR / "final_results_summary.csv"
        pd.DataFrame(summary).to_csv(
            output,
            index=False,
        )
        print(f"\nResumen guardado en: {output}")

    print("\nProceso finalizado.")


if __name__ == "__main__":
    main()