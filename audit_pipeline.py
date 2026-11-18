from pathlib import Path
import ast
import re


BASE_DIR = Path(__file__).resolve().parent

SCRIPTS = sorted(BASE_DIR.glob("run_*.py"))

FILE_PATTERN = re.compile(
    r"""["']([^"']+\.(?:csv|joblib|pt|json|png))["']""",
    re.IGNORECASE,
)


def analyze_script(path: Path):
    """Extrae imports, referencias a archivos y operaciones."""

    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    imports = set()
    operations = set()

    for node in ast.walk(tree):

        if isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module)

        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)

        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                operations.add(node.func.attr)

    files = sorted(set(FILE_PATTERN.findall(source)))

    return {
        "imports": sorted(imports),
        "files": files,
        "operations": sorted(operations),
    }


def main():

    print("\n========== AUDITORÍA DEL PIPELINE ==========")

    for script in SCRIPTS:

        if script.name == "run_full_pipeline.py":
            continue

        print(f"\n{'=' * 60}")
        print(f"SCRIPT: {script.name}")
        print("=" * 60)

        try:
            result = analyze_script(script)

        except (SyntaxError, UnicodeError) as error:
            print(f"ERROR: {error}")
            continue

        print("\nMódulos del proyecto:")

        project_imports = [
            module
            for module in result["imports"]
            if module.startswith("src.")
        ]

        if project_imports:
            for module in project_imports:
                print(f"  - {module}")
        else:
            print("  No detectados")

        print("\nArchivos referenciados:")

        if result["files"]:
            for filename in result["files"]:
                print(f"  - {filename}")
        else:
            print("  No detectados directamente")

        relevant_operations = [
            operation
            for operation in result["operations"]
            if operation in {
                "read_csv",
                "to_csv",
                "dump",
                "load",
                "save",
                "savefig",
                "read_json",
                "to_json",
            }
        ]

        print("\nOperaciones de lectura/escritura:")

        if relevant_operations:
            for operation in relevant_operations:
                print(f"  - {operation}")
        else:
            print("  No detectadas directamente")

    print("\nAuditoría finalizada.")
    print(
        "Nota: este análisis es estático y puede no "
        "detectar rutas construidas dinámicamente "
        "o lecturas dentro de módulos importados."
    )


if __name__ == "__main__":
    main()