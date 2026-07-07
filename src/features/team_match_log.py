"""
Helper compartido por team_stats.py y recent_form.py.

Convierte la tabla de partidos (una fila por partido, con equipo local
y visitante) en una tabla "long" con una fila por equipo y partido.
Esto es lo que permite calcular, para cada equipo, sus estadísticas
ordenadas cronológicamente sin importar si jugó de local o visitante.
"""

import pandas as pd

TEAM_LOG_COLUMNS = ["match_id", "date", "team", "goals_for", "goals_against", "is_home"]


def build_team_match_log(matches: pd.DataFrame) -> pd.DataFrame:
    """Devuelve una fila por (partido, equipo), ordenada por equipo y fecha."""

    home = matches[["match_id", "date", "home_team", "home_goals", "away_goals"]].copy()
    home.columns = ["match_id", "date", "team", "goals_for", "goals_against"]
    home["is_home"] = True

    away = matches[["match_id", "date", "away_team", "away_goals", "home_goals"]].copy()
    away.columns = ["match_id", "date", "team", "goals_for", "goals_against"]
    away["is_home"] = False

    log = pd.concat([home, away], ignore_index=True)[TEAM_LOG_COLUMNS]
    log["date"] = pd.to_datetime(log["date"])

    return log.sort_values(["team", "date", "match_id"]).reset_index(drop=True)
