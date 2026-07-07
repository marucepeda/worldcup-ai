import pandas as pd
import pytest


@pytest.fixture
def sample_matches() -> pd.DataFrame:
    """
    Un mini dataset de partidos, con el mismo esquema que
    data/processed/international/international_matches.csv, pensado
    para poder calcular a mano el resultado esperado de cada feature.

    Historial relevante para Argentina, en orden cronológico:
      match 1 (local vs Brasil):  gana 2-0
      match 3 (local vs Chile):   gana 3-1
      match 4 (visitante vs Chile): gana 2-0
    """

    return pd.DataFrame({
        "match_id": [1, 2, 3, 4, 5],
        "date": [
            "2020-01-01", "2020-01-05", "2020-01-10", "2020-01-15", "2020-01-20",
        ],
        "home_team": ["Argentina", "Brazil", "Argentina", "Chile", "Brazil"],
        "away_team": ["Brazil", "Argentina", "Chile", "Argentina", "Chile"],
        "home_goals": [2, 1, 3, 0, 2],
        "away_goals": [0, 1, 1, 2, 2],
    })
