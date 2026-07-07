"""
Transforma el dataset crudo de partidos internacionales en una tabla
"processed" con nombres de columnas estandarizados y un match_id único.

El archivo crudo tiene dos problemas de calidad de datos conocidos:

1. 77 filas están mal parseadas: algunas ciudades contienen una coma
   (por ejemplo "Washington, D.C."), y el exportador original envolvió
   la fila entera entre comillas en vez de solo ese campo. Pandas lee
   entonces la fila completa como si fuera un único valor en la columna
   "date", dejando el resto de las columnas en NaN. Como el contenido
   original sigue siendo un CSV válido, se puede reparar volviendo a
   parsear esa fila con el módulo csv de la librería estándar.

2. Un puñado de partidos todavía no se jugaron (fixtures futuros, con
   "NA" en el marcador). No tienen resultado, así que no sirven para
   entrenar ni evaluar un modelo y se descartan.

Input:  data/raw/international/international_results.csv
Output: data/processed/international/international_matches.csv
"""

import csv
import io

import pandas as pd

from src.config import PROCESSED_MATCHES_FILE, RAW_MATCHES_FILE

COLUMN_MAPPING = {
    "date": "date",
    "home_team": "home_team",
    "away_team": "away_team",
    "home_score": "home_goals",
    "away_score": "away_goals",
    "tournament": "competition",
    "neutral": "neutral_venue",
}


def _repair_malformed_rows(raw_matches: pd.DataFrame) -> pd.DataFrame:
    """Vuelve a parsear las filas donde una coma en 'city' rompió el CSV original."""

    raw_matches = raw_matches.astype(object)
    malformed = raw_matches["home_team"].isna() & raw_matches["date"].str.contains(",", na=False)

    for index in raw_matches[malformed].index:
        fields = next(csv.reader(io.StringIO(raw_matches.loc[index, "date"])))
        raw_matches.loc[index, raw_matches.columns] = fields

    return raw_matches


def build_processed_matches(raw_matches: pd.DataFrame) -> pd.DataFrame:
    """Repara filas mal parseadas, descarta partidos sin jugar y estandariza columnas."""

    raw_matches = _repair_malformed_rows(raw_matches)

    raw_matches["home_score"] = pd.to_numeric(raw_matches["home_score"], errors="coerce")
    raw_matches["away_score"] = pd.to_numeric(raw_matches["away_score"], errors="coerce")
    raw_matches = raw_matches.dropna(subset=["home_score", "away_score", "home_team", "away_team"])

    processed = raw_matches.rename(columns=COLUMN_MAPPING)[list(COLUMN_MAPPING.values())].copy()
    processed = processed.reset_index(drop=True)
    processed.insert(0, "match_id", range(1, len(processed) + 1))

    return processed


def main() -> None:
    raw_matches = pd.read_csv(RAW_MATCHES_FILE)

    processed = build_processed_matches(raw_matches)

    PROCESSED_MATCHES_FILE.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(PROCESSED_MATCHES_FILE, index=False)

    dropped = len(raw_matches) - len(processed)
    print(f"Dataset procesado creado: {PROCESSED_MATCHES_FILE}")
    print(f"Partidos descartados por falta de resultado: {dropped}")
    print(processed.head())


if __name__ == "__main__":
    main()

