from src.features.build_match_dataset import build_match_dataset
from src.features.elo_rating import build_elo_ratings
from src.features.recent_form import build_recent_form
from src.features.team_stats import build_team_stats


def _build_dataset(sample_matches):
    team_stats = build_team_stats(sample_matches)
    recent_form = build_recent_form(sample_matches)
    elo_ratings = build_elo_ratings(sample_matches)

    return build_match_dataset(sample_matches, team_stats, recent_form, elo_ratings)


def test_result_class_matches_the_scoreline(sample_matches):
    dataset = _build_dataset(sample_matches)

    result_by_match = dataset.set_index("match_id")["result_class"]

    assert result_by_match[1] == 2  # Argentina 2-0 Brazil: gana el local
    assert result_by_match[2] == 1  # Brazil 1-1 Argentina: empate
    assert result_by_match[4] == 0  # Chile 0-2 Argentina: gana el visitante


def test_no_missing_values_in_engineered_features(sample_matches):
    dataset = _build_dataset(sample_matches)

    engineered_columns = [
        "home_avg_goals_for", "away_avg_goals_for",
        "home_recent_wins", "away_recent_wins",
        "home_elo", "away_elo", "elo_difference",
    ]

    assert dataset[engineered_columns].isna().sum().sum() == 0


def test_dataset_is_sorted_chronologically(sample_matches):
    dataset = _build_dataset(sample_matches)

    assert list(dataset["match_id"]) == sorted(dataset["match_id"])
