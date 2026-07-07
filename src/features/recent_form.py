"""
Forma reciente por equipo: resultados y goles de los últimos 5 partidos.

Igual que en team_stats.py, la ventana de los últimos 5 partidos se
calcula excluyendo el partido actual (shift(1) antes del rolling), para
no filtrar información del futuro hacia atrás.

Input:  data/processed/international/international_matches.csv
Output: data/features/international/recent_form.csv (una fila por match_id)
"""

import pandas as pd

from src.config import PROCESSED_MATCHES_FILE, RECENT_FORM_FILE
from src.features.team_match_log import build_team_match_log

FORM_WINDOW = 5


def _prior_form(log: pd.DataFrame) -> pd.DataFrame:
    log = log.copy()

    log["win"] = (log["goals_for"] > log["goals_against"]).astype(int)
    log["draw"] = (log["goals_for"] == log["goals_against"]).astype(int)
    log["loss"] = (log["goals_for"] < log["goals_against"]).astype(int)

    grouped = log.groupby("team")

    def prior_rolling(column: str, agg: str) -> pd.Series:
        return grouped[column].transform(
            lambda s: s.shift(1).rolling(FORM_WINDOW, min_periods=1).agg(agg)
        )

    log["recent_goals_for"] = prior_rolling("goals_for", "mean")
    log["recent_goals_against"] = prior_rolling("goals_against", "mean")
    log["recent_wins"] = prior_rolling("win", "sum")
    log["recent_draws"] = prior_rolling("draw", "sum")
    log["recent_losses"] = prior_rolling("loss", "sum")

    recent_columns = [
        "recent_goals_for", "recent_goals_against",
        "recent_wins", "recent_draws", "recent_losses",
    ]
    log[recent_columns] = log[recent_columns].fillna(0)

    return log


def build_recent_form(matches: pd.DataFrame) -> pd.DataFrame:
    log = build_team_match_log(matches)
    log = _prior_form(log)

    form_columns = ["match_id", "recent_goals_for", "recent_goals_against",
                     "recent_wins", "recent_draws", "recent_losses"]

    home_form = log[log["is_home"]][form_columns].rename(columns={
        "recent_goals_for": "home_recent_goals_for",
        "recent_goals_against": "home_recent_goals_against",
        "recent_wins": "home_recent_wins",
        "recent_draws": "home_recent_draws",
        "recent_losses": "home_recent_losses",
    })

    away_form = log[~log["is_home"]][form_columns].rename(columns={
        "recent_goals_for": "away_recent_goals_for",
        "recent_goals_against": "away_recent_goals_against",
        "recent_wins": "away_recent_wins",
        "recent_draws": "away_recent_draws",
        "recent_losses": "away_recent_losses",
    })

    return home_form.merge(away_form, on="match_id").sort_values("match_id").reset_index(drop=True)


def main() -> None:
    matches = pd.read_csv(PROCESSED_MATCHES_FILE)

    features = build_recent_form(matches)

    RECENT_FORM_FILE.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(RECENT_FORM_FILE, index=False)

    print(f"Recent form creada: {RECENT_FORM_FILE}")
    print(features.head())


if __name__ == "__main__":
    main()
