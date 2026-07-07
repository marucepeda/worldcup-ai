from src.features.recent_form import build_recent_form


def test_first_match_has_no_form(sample_matches):
    features = build_recent_form(sample_matches)

    first_match = features[features["match_id"] == 1].iloc[0]

    assert first_match["home_recent_wins"] == 0
    assert first_match["home_recent_goals_for"] == 0


def test_form_counts_wins_before_the_match(sample_matches):
    """
    Antes del match 4, Argentina jugó 3 partidos: ganó el match 1,
    empató el match 2 y ganó el match 3. Por lo tanto recent_wins debe
    ser 2, recent_draws 1 y recent_losses 0, sin contar el resultado
    del propio match 4.
    """

    features = build_recent_form(sample_matches)

    match_4 = features[features["match_id"] == 4].iloc[0]

    assert match_4["away_recent_wins"] == 2
    assert match_4["away_recent_draws"] == 1
    assert match_4["away_recent_losses"] == 0
    assert match_4["away_recent_goals_for"] == 2.0
