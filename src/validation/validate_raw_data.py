"""
Chequeo de calidad sobre el dataset crudo, antes de procesarlo.

No es un framework de validación (por ejemplo, Great Expectations
sería sobreingeniería para este proyecto). Es simplemente un script
que informa el estado de los datos y corta el pipeline si encuentra
un problema estructural grave (columnas faltantes, por ejemplo).

Nota: el dataset crudo tiene un porcentaje bajo (<1%) de filas con
valores faltantes -partidos aún no jugados, o filas mal parseadas por
una coma dentro del nombre de la ciudad-. Ambos casos se resuelven en
el siguiente paso del pipeline (build_matches_dataset.py), así que acá
solo se reportan, no se bloquea la ejecución por su presencia.
"""

import pandas as pd

from src.config import RAW_MATCHES_FILE

REQUIRED_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "neutral",
]

MAX_MISSING_RATIO = 0.01


def validate(df: pd.DataFrame) -> None:
    """Lanza un AssertionError si el dataset tiene problemas bloqueantes."""

    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)
    assert not missing_columns, f"Faltan columnas requeridas: {missing_columns}"

    missing_ratio = df["home_team"].isna().mean()
    assert missing_ratio <= MAX_MISSING_RATIO, (
        f"El {missing_ratio:.1%} de las filas tiene datos faltantes, "
        f"por encima del umbral tolerado ({MAX_MISSING_RATIO:.0%})"
    )


def main() -> None:
    df = pd.read_csv(RAW_MATCHES_FILE)

    print("Filas:", len(df))
    print("\nColumnas:", df.columns.tolist())
    print("\nValores faltantes por columna:")
    print(df.isnull().sum())
    print("\nPartidos duplicados:", df.duplicated().sum())

    validate(df)
    print("\nValidación superada: el dataset crudo está listo para procesarse.")


if __name__ == "__main__":
    main()
