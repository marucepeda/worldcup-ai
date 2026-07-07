"""
Utilidades compartidas por los scripts de entrenamiento.

Los cuatro modelos (baseline, regresión logística, regresión logística
balanceada y random forest) usan el mismo dataset, el mismo conjunto de
features, el mismo split temporal y la misma forma de reportar
resultados. Centralizarlo acá evita repetir el mismo bloque de código
cuatro veces.
"""

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from src.config import MATCH_DATASET_FILE

TARGET_COLUMN = "result_class"

FEATURE_COLUMNS = [
    "home_avg_goals_for", "home_avg_goals_against", "home_matches_played",
    "away_avg_goals_for", "away_avg_goals_against", "away_matches_played",
    "home_recent_goals_for", "home_recent_goals_against",
    "home_recent_wins", "home_recent_draws", "home_recent_losses",
    "away_recent_goals_for", "away_recent_goals_against",
    "away_recent_wins", "away_recent_draws", "away_recent_losses",
    "attack_difference", "defense_difference",
    "form_difference", "recent_attack_difference",
    "home_elo", "away_elo", "elo_difference",
]

TRAIN_FRACTION = 0.8


def load_dataset() -> pd.DataFrame:
    return pd.read_csv(MATCH_DATASET_FILE)


def time_based_split(df: pd.DataFrame):
    """
    Separa train/test respetando el orden cronológico: entrena con los
    partidos más antiguos y evalúa con los más recientes. Un split
    aleatorio "filtraría" información del futuro hacia el pasado.

    El dataset ya viene ordenado por fecha desde build_match_dataset.py.
    """

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    split_index = int(len(df) * TRAIN_FRACTION)

    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    return X_train, X_test, y_train, y_test


def report(model_name: str, y_test: pd.Series, predictions) -> None:
    accuracy = accuracy_score(y_test, predictions)

    print(f"{model_name} accuracy:")
    print(round(accuracy, 4))

    print("\nReporte:")
    print(classification_report(y_test, predictions))
