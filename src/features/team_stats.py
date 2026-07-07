"""
Estadísticas históricas por equipo: promedio de goles a favor, en contra
y cantidad de partidos jugados.

Importante: estas estadísticas se calculan "point-in-time", es decir,
usando únicamente los partidos ANTERIORES a la fecha del partido que
se está describiendo. Calcular el promedio histórico con el dataset
completo (incluyendo partidos futuros) sería data leakage: el modelo
"vería" información que todavía no existía en el momento del partido.

Input:  data/processed/international/international_matches.csv
Output: data/features/international/team_stats.csv (una fila por match_id)
"""

import pandas as pd

from src.config import PROCESSED_MATCHES_FILE, TEAM_STATS_FILE
from src.features.team_match_log import build_team_match_log


def _prior_matches_stats(log: pd.DataFrame) -> pd.DataFrame:
    """Agrega, para cada fila, las estadísticas de los partidos previos del equipo."""

    log = log.copy()
    grouped = log.groupby("team")

    # shift(1) excluye el partido actual antes de calcular el promedio acumulado
    log["avg_goals_for"] = grouped["goals_for"].transform(lambda s: s.shift(1).expanding().mean())
    log["avg_goals_against"] = grouped["goals_against"].transform(lambda s: s.shift(1).expanding().mean())
    log["matches_played"] = grouped.cumcount()

    # El primer partido de cada equipo no tiene historial: se completa con 0
    log[["avg_goals_for", "avg_goals_against"]] = log[["avg_goals_for", "avg_goals_against"]].fillna(0)

    return log


def build_team_stats(matches: pd.DataFrame) -> pd.DataFrame:
    log = build_team_match_log(matches)
    log = _prior_matches_stats(log)

    stats_columns = ["match_id", "avg_goals_for", "avg_goals_against", "matches_played"]

    home_stats = log[log["is_home"]][stats_columns].rename(columns={
        "avg_goals_for": "home_avg_goals_for",
        "avg_goals_against": "home_avg_goals_against",
        "matches_played": "home_matches_played",
    })

    away_stats = log[~log["is_home"]][stats_columns].rename(columns={
        "avg_goals_for": "away_avg_goals_for",
        "avg_goals_against": "away_avg_goals_against",
        "matches_played": "away_matches_played",
    })

    return home_stats.merge(away_stats, on="match_id").sort_values("match_id").reset_index(drop=True)


def main() -> None:
    matches = pd.read_csv(PROCESSED_MATCHES_FILE)

    features = build_team_stats(matches)

    TEAM_STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(TEAM_STATS_FILE, index=False)

    print(f"Team stats creadas: {TEAM_STATS_FILE}")
    print(features.head())


if __name__ == "__main__":
    main()
