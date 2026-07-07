from src.features.team_stats import build_team_stats


def test_first_match_has_no_history(sample_matches):
    features = build_team_stats(sample_matches)

    first_match = features[features["match_id"] == 1].iloc[0]

    assert first_match["home_matches_played"] == 0
    assert first_match["home_avg_goals_for"] == 0
    assert first_match["away_matches_played"] == 0
    assert first_match["away_avg_goals_for"] == 0


def test_stats_only_use_prior_matches(sample_matches):
    """
    Antes del match 4 (Chile vs Argentina), Argentina ya jugó los
    matches 1 (2 a favor, 0 en contra), 2 (1 a favor, 1 en contra, de
    visitante) y 3 (3 a favor, 1 en contra). El promedio point-in-time
    debe ser (2+1+3)/3 = 2.0 y (0+1+1)/3 ≈ 0.667, SIN incluir el
    resultado del propio match 4.
    """

    features = build_team_stats(sample_matches)

    match_4 = features[features["match_id"] == 4].iloc[0]

    assert match_4["away_matches_played"] == 3
    assert match_4["away_avg_goals_for"] == 2.0
    assert round(match_4["away_avg_goals_against"], 3) == 0.667
