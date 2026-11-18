from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
FIGURES_DIR = BASE_DIR / "reports" / "figures"


def save_figure(filename):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    path = FIGURES_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Gráfica guardada: {path}")


def plot_model_comparison():
    path = DATA_DIR / "model_comparison.csv"

    if not path.exists():
        print("No se encontró model_comparison.csv")
        return

    df = pd.read_csv(path)

    required = {"model", "denomination", "MAE_notes"}

    if not required.issubset(df.columns):
        print("Columnas insuficientes para comparar modelos.")
        return

    data = df.dropna(subset=["MAE_notes"])

    pivot = data.pivot_table(
        index="denomination",
        columns="model",
        values="MAE_notes",
        aggfunc="mean",
    )

    ax = pivot.plot(kind="bar", figsize=(10, 5))

    ax.set_title("Error absoluto medio por denominación")
    ax.set_xlabel("Denominación")
    ax.set_ylabel("MAE (billetes)")
    ax.legend(title="Modelo")

    save_figure("final_model_comparison.png")


def plot_shortage_rates():
    path = DATA_DIR / "safety_stock_optimization.csv"

    if not path.exists():
        print("No se encontró safety_stock_optimization.csv")
        return

    df = pd.read_csv(path)

    required = {
        "scenario",
        "safety_stock",
        "shortage_rate_pct",
    }

    if not required.issubset(df.columns):
        print("Columnas insuficientes para faltantes.")
        return

    pivot = df.pivot_table(
        index="safety_stock",
        columns="scenario",
        values="shortage_rate_pct",
        aggfunc="mean",
    )

    ax = pivot.plot(marker="o", figsize=(10, 5))

    ax.set_title("Faltantes según stock de seguridad")
    ax.set_xlabel("Stock de seguridad (billetes)")
    ax.set_ylabel("Tasa de faltantes (%)")
    ax.grid(alpha=0.3)

    save_figure("final_shortage_rates.png")


def plot_replenishment():
    path = DATA_DIR / "safety_stock_optimization.csv"

    if not path.exists():
        return

    df = pd.read_csv(path)

    required = {
        "scenario",
        "safety_stock",
        "total_replenished_cop",
    }

    if not required.issubset(df.columns):
        print("Columnas insuficientes para reposición.")
        return

    pivot = df.pivot_table(
        index="safety_stock",
        columns="scenario",
        values="total_replenished_cop",
        aggfunc="mean",
    )

    ax = (pivot / 1_000_000).plot(
        marker="o",
        figsize=(10, 5),
    )

    ax.set_title("Efectivo repuesto según stock de seguridad")
    ax.set_xlabel("Stock de seguridad (billetes)")
    ax.set_ylabel("Efectivo repuesto (millones de COP)")
    ax.grid(alpha=0.3)

    save_figure("final_replenishment.png")


def main():
    print("\nGenerando gráficas finales...")

    plot_model_comparison()
    plot_shortage_rates()
    plot_replenishment()

    print("\nProceso completado.")


if __name__ == "__main__":
    main()