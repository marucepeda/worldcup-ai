"""
Chequeo exploratorio rápido sobre el dataset final de Machine Learning.

Este script es exploratorio (se ejecuta a mano durante el desarrollo),
por eso vive fuera de src/ en vez de ser parte del pipeline.
"""

import pandas as pd

from src.config import MATCH_DATASET_FILE


def check_quality() -> None:
    df = pd.read_csv(MATCH_DATASET_FILE)

    print("\n===== DIMENSIONES =====")
    print(df.shape)

    print("\n===== COLUMNAS =====")
    print(df.columns.tolist())

    print("\n===== VALORES FALTANTES =====")
    print(df.isna().sum())

    print("\n===== DISTRIBUCIÓN DEL TARGET (result_class) =====")
    print(df["result_class"].value_counts())

    print("\n===== FECHAS =====")
    print(df["date"].min(), "-", df["date"].max())

    print("\n===== EQUIPOS =====")
    teams = pd.concat([df["home_team"], df["away_team"]])
    print("Cantidad de equipos:", teams.nunique())


if __name__ == "__main__":
    check_quality()
