"""
Ensambla el dataset final de Machine Learning.

Combina, por match_id:
  - la tabla de partidos procesada (fecha, equipos, goles),
  - las estadísticas históricas por equipo (team_stats),
  - la forma reciente por equipo (recent_form),
  - el rating Elo (elo_rating),

y agrega encima las features comparativas (diferencias entre el equipo
local y el visitante) y la variable objetivo.

Input:
  data/processed/international/international_matches.csv
  data/features/international/team_stats.csv
  data/features/international/recent_form.csv
  data/features/international/elo_ratings.csv
Output:
  data/features/international/match_dataset.csv
"""

import pandas as pd

from src.config import (
    ELO_RATINGS_FILE,
    MATCH_DATASET_FILE,
    PROCESSED_MATCHES_FILE,
    RECENT_FORM_FILE,
    TEAM_STATS_FILE,
)


def _add_comparative_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["attack_difference"] = df["home_avg_goals_for"] - df["away_avg_goals_for"]
    df["defense_difference"] = df["away_avg_goals_against"] - df["home_avg_goals_against"]
    df["form_difference"] = df["home_recent_wins"] - df["away_recent_wins"]
    df["recent_attack_difference"] = df["home_recent_goals_for"] - df["away_recent_goals_for"]
    # elo_difference ya viene calculada en elo_ratings.csv

    return df


def _add_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    conditions = [
        df["home_goals"] > df["away_goals"],
        df["home_goals"] == df["away_goals"],
    ]
    # 2 = gana el local, 1 = empate, 0 = gana el visitante
    df["result_class"] = pd.Series(2, index=df.index)
    df.loc[conditions[1], "result_class"] = 1
    df.loc[~conditions[0] & ~conditions[1], "result_class"] = 0

    return df


def build_match_dataset(
    matches: pd.DataFrame,
    team_stats: pd.DataFrame,
    recent_form: pd.DataFrame,
    elo_ratings: pd.DataFrame,
) -> pd.DataFrame:
    df = matches.merge(team_stats, on="match_id", how="left")
    df = df.merge(recent_form, on="match_id", how="left")
    df = df.merge(elo_ratings, on="match_id", how="left")

    df = _add_comparative_features(df)
    df = _add_target(df)

    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "match_id"]).reset_index(drop=True)


def main() -> None:
    matches = pd.read_csv(PROCESSED_MATCHES_FILE)
    team_stats = pd.read_csv(TEAM_STATS_FILE)
    recent_form = pd.read_csv(RECENT_FORM_FILE)
    elo_ratings = pd.read_csv(ELO_RATINGS_FILE)

    dataset = build_match_dataset(matches, team_stats, recent_form, elo_ratings)

    MATCH_DATASET_FILE.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(MATCH_DATASET_FILE, index=False)

    print(f"Dataset de ML creado: {MATCH_DATASET_FILE}")
    print(dataset.shape)
    print("\nDistribución de la variable objetivo (result_class):")
    print(dataset["result_class"].value_counts())


if __name__ == "__main__":
    main()
