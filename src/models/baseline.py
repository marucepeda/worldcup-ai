"""
Baseline: siempre predice que gana el equipo local.

Sirve como piso de comparación: cualquier modelo entrenado debería
superar esta accuracy para justificar su complejidad.
"""

from src.models.common import load_dataset, report

HOME_WIN_CLASS = 2


def main() -> None:
    df = load_dataset()

    predictions = [HOME_WIN_CLASS] * len(df)

    report("Baseline (siempre gana el local)", df["result_class"], predictions)


if __name__ == "__main__":
    main()
