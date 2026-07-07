from src.features.elo_rating import build_elo_ratings, expected_score, update_elo


def test_expected_score_is_fifty_fifty_for_equal_ratings():
    assert expected_score(1500, 1500) == 0.5


def test_expected_score_favors_higher_rating():
    assert expected_score(1600, 1400) > 0.5
    assert expected_score(1400, 1600) < 0.5


def test_winner_gains_rating_and_loser_loses_rating():
    new_winner, new_loser = update_elo(1500, 1500, result=1)

    assert new_winner > 1500
    assert new_loser < 1500


def test_first_match_uses_initial_rating(sample_matches):
    features = build_elo_ratings(sample_matches)

    first_match = features[features["match_id"] == 1].iloc[0]

    assert first_match["home_elo"] == 1500
    assert first_match["away_elo"] == 1500


def test_elo_updates_after_a_win(sample_matches):
    """Argentina ganó el match 1, así que en el match 3 debería tener más de 1500."""

    features = build_elo_ratings(sample_matches)

    match_3 = features[features["match_id"] == 3].iloc[0]

    assert match_3["home_elo"] > 1500
