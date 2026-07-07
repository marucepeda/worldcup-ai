"""
Sistema de puntuación Elo para selecciones nacionales.

Cada equipo arranca con un rating inicial y lo va actualizando partido
a partido, en orden cronológico, según el resultado real. El Elo de un
equipo antes de un partido queda registrado como feature de ese partido,
por lo que -al igual que team_stats y recent_form- no usa información
del futuro.

Input:  data/processed/international/international_matches.csv
Output: data/features/international/elo_ratings.csv (una fila por match_id)
"""

import pandas as pd

from src.config import ELO_RATINGS_FILE, PROCESSED_MATCHES_FILE

INITIAL_ELO = 1500
K_FACTOR = 20


def expected_score(rating_a: float, rating_b: float) -> float:
    """Probabilidad esperada de que el equipo A le gane al equipo B."""

    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def update_elo(rating_a: float, rating_b: float, result: float) -> tuple[float, float]:
    """
    Devuelve los nuevos ratings de A y B tras un partido.

    result: 1 si ganó A, 0.5 si empataron, 0 si ganó B.
    """

    expected_a = expected_score(rating_a, rating_b)

    new_a = rating_a + K_FACTOR * (result - expected_a)
    new_b = rating_b + K_FACTOR * ((1 - result) - (1 - expected_a))

    return new_a, new_b


def build_elo_ratings(matches: pd.DataFrame) -> pd.DataFrame:
    matches = matches.copy()
    matches["date"] = pd.to_datetime(matches["date"])
    matches = matches.sort_values(["date", "match_id"])

    ratings: dict[str, float] = {}
    home_elo, away_elo = [], []

    for row in matches.itertuples():
        home_rating = ratings.setdefault(row.home_team, INITIAL_ELO)
        away_rating = ratings.setdefault(row.away_team, INITIAL_ELO)

        home_elo.append(home_rating)
        away_elo.append(away_rating)

        if row.home_goals > row.away_goals:
            result = 1.0
        elif row.home_goals == row.away_goals:
            result = 0.5
        else:
            result = 0.0

        ratings[row.home_team], ratings[row.away_team] = update_elo(home_rating, away_rating, result)

    matches["home_elo"] = home_elo
    matches["away_elo"] = away_elo
    matches["elo_difference"] = matches["home_elo"] - matches["away_elo"]

    columns = ["match_id", "home_elo", "away_elo", "elo_difference"]
    return matches[columns].sort_values("match_id").reset_index(drop=True)


def main() -> None:
    matches = pd.read_csv(PROCESSED_MATCHES_FILE)

    features = build_elo_ratings(matches)

    ELO_RATINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(ELO_RATINGS_FILE, index=False)

    print(f"Elo ratings creados: {ELO_RATINGS_FILE}")
    print(features.head())


if __name__ == "__main__":
    main()
