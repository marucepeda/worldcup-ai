"""
Predice el resultado de un partido puntual (por ejemplo, uno que todavía
no se jugó o que está en curso) usando el mismo pipeline de features que
el resto del repo, sin modificar ningún archivo de datos existente.

Idea:
  1. Lee el CSV crudo tal cual está (ya incluye el fixture, con el
     marcador en NaN si todavía no terminó).
  2. Corre el mismo pipeline de features (team_stats, recent_form, elo)
     pero SIN descartar filas sin resultado, para que el partido que nos
     interesa quede incluido con sus features "point-in-time" (calculadas
     solo con partidos anteriores, como el resto del dataset).
  3. Entrena un modelo con todos los partidos que sí tienen resultado.
  4. Predice la fila del partido pedido.

Uso:
  python scripts/predict_match.py "Switzerland" "Colombia"
  python scripts/predict_match.py "Switzerland" "Colombia" --model rf
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.config import RAW_MATCHES_FILE  # noqa: E402
from src.features.build_match_dataset import _add_comparative_features  # noqa: E402
from src.features.elo_rating import build_elo_ratings  # noqa: E402
from src.features.recent_form import build_recent_form  # noqa: E402
from src.features.team_stats import build_team_stats  # noqa: E402
from src.models.common import FEATURE_COLUMNS  # noqa: E402
from src.processing.build_matches_dataset import _repair_malformed_rows  # noqa: E402

RESULT_LABELS = {2: "Gana el local", 1: "Empate", 0: "Gana el visitante"}


def build_processed_matches_keep_future(raw_matches: pd.DataFrame) -> pd.DataFrame:
    """Igual que build_processed_matches, pero SIN descartar fixtures sin jugar."""

    raw_matches = _repair_malformed_rows(raw_matches)

    raw_matches["home_score"] = pd.to_numeric(raw_matches["home_score"], errors="coerce")
    raw_matches["away_score"] = pd.to_numeric(raw_matches["away_score"], errors="coerce")
    raw_matches = raw_matches.dropna(subset=["home_team", "away_team"])

    column_mapping = {
        "date": "date", "home_team": "home_team", "away_team": "away_team",
        "home_score": "home_goals", "away_score": "away_goals",
        "tournament": "competition", "neutral": "neutral_venue",
    }
    processed = raw_matches.rename(columns=column_mapping)[list(column_mapping.values())].copy()
    processed = processed.reset_index(drop=True)
    processed.insert(0, "match_id", range(1, len(processed) + 1))
    return processed


def find_fixture_row(matches: pd.DataFrame, home: str, away: str) -> pd.Series:
    candidates = matches[
        (matches["home_team"].str.lower() == home.lower())
        & (matches["away_team"].str.lower() == away.lower())
        & matches["home_goals"].isna()
    ]
    if candidates.empty:
        raise SystemExit(
            f"No encontré un partido pendiente {home} vs {away} en el CSV crudo.\n"
            f"Revisá data/raw/international/international_results.csv, o probá invirtiendo local/visitante."
        )
    return candidates.iloc[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Predice un partido puntual.")
    parser.add_argument("home_team")
    parser.add_argument("away_team")
    parser.add_argument("--model", choices=["logreg", "rf"], default="logreg")
    parser.add_argument(
        "--neutral", action="store_true",
        help="Cancha de nadie: promedia la predicción con cada equipo como 'local' "
             "para anular el sesgo de localía (recomendado en mundiales).",
    )
    parser.add_argument(
        "--knockout", action="store_true",
        help="Fase eliminatoria: no hay empate posible, reparte esa probabilidad "
             "50/50 (penales) entre los dos equipos para estimar quién avanza.",
    )
    args = parser.parse_args()

    raw_matches = pd.read_csv(RAW_MATCHES_FILE)
    matches = build_processed_matches_keep_future(raw_matches)

    fixture = find_fixture_row(matches, args.home_team, args.away_team)
    team_a, team_b = fixture["home_team"], fixture["away_team"]
    print(f"Partido encontrado: {team_a} vs {team_b} ({fixture['date']})")
    if fixture["neutral_venue"]:
        print("(cancha neutral, según el dataset)")

    team_stats = build_team_stats(matches)
    recent_form = build_recent_form(matches)
    elo_ratings = build_elo_ratings(matches)

    dataset = matches.merge(team_stats, on="match_id", how="left")
    dataset = dataset.merge(recent_form, on="match_id", how="left")
    dataset = dataset.merge(elo_ratings, on="match_id", how="left")
    dataset = _add_comparative_features(dataset)

    # Entrenamos con todo lo que sí tiene resultado real (excluye el fixture)
    train = dataset.dropna(subset=["home_goals", "away_goals"]).copy()
    conditions_win = train["home_goals"] > train["away_goals"]
    conditions_draw = train["home_goals"] == train["away_goals"]
    train["result_class"] = 2
    train.loc[conditions_draw, "result_class"] = 1
    train.loc[~conditions_win & ~conditions_draw, "result_class"] = 0

    X_train = train[FEATURE_COLUMNS]
    y_train = train["result_class"]

    if args.model == "logreg":
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_scaled, y_train)
    else:
        scaler = None
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)

    def predict_row(match_id):
        row = dataset[dataset["match_id"] == match_id][FEATURE_COLUMNS]
        row_input = scaler.transform(row) if scaler is not None else row
        probs = model.predict_proba(row_input)[0]
        return {cls: probs[list(model.classes_).index(cls)] for cls in RESULT_LABELS if cls in model.classes_}

    fixture_id = fixture["match_id"]
    probs_a_home = predict_row(fixture_id)  # team_a de local, team_b de visitante

    if args.neutral:
        row = dataset[dataset["match_id"] == fixture_id][FEATURE_COLUMNS].iloc[0].copy()
        swapped = row.copy()
        # Intercambiamos cada par home_/away_ para simular a team_b como local
        for col in FEATURE_COLUMNS:
            if col.startswith("home_"):
                away_col = "away_" + col[len("home_"):]
                swapped[col], swapped[away_col] = row[away_col], row[col]
        swapped["elo_difference"] *= -1
        swapped["attack_difference"] *= -1
        swapped["defense_difference"] *= -1
        swapped["form_difference"] *= -1
        swapped["recent_attack_difference"] *= -1
        swapped_df = pd.DataFrame([swapped[FEATURE_COLUMNS]])
        swapped_input = scaler.transform(swapped_df) if scaler is not None else swapped_df
        probs_swapped = model.predict_proba(swapped_input)[0]
        probs_b_home = {cls: probs_swapped[list(model.classes_).index(cls)] for cls in RESULT_LABELS if cls in model.classes_}

        # probs_b_home está en términos de "team_b local"; lo volvemos a términos de team_a/team_b
        p_a_win = (probs_a_home.get(2, 0) + probs_b_home.get(0, 0)) / 2
        p_draw = (probs_a_home.get(1, 0) + probs_b_home.get(1, 0)) / 2
        p_b_win = (probs_a_home.get(0, 0) + probs_b_home.get(2, 0)) / 2
    else:
        p_a_win, p_draw, p_b_win = probs_a_home.get(2, 0), probs_a_home.get(1, 0), probs_a_home.get(0, 0)

    print(f"\nModelo: {args.model} | Entrenado con {len(train)} partidos históricos\n")
    print("Probabilidades en 90 minutos:")
    print(f"  Gana {team_a:<15} {p_a_win:.1%}")
    print(f"  Empate{'':<15}{p_draw:.1%}")
    print(f"  Gana {team_b:<15} {p_b_win:.1%}")

    if args.knockout:
        # Reparte el empate 50/50 (penales no modelados: es esencialmente una moneda)
        p_a_advances = p_a_win + p_draw / 2
        p_b_advances = p_b_win + p_draw / 2
        print("\nFase eliminatoria -> no hay empate posible (penales si hay igualdad):")
        print(f"  {team_a:<20} avanza: {p_a_advances:.1%}")
        print(f"  {team_b:<20} avanza: {p_b_advances:.1%}")
        print("  (el 50/50 del empate es una suposición nuestra, no algo que el modelo sepa predecir)")
    else:
        best = max([(p_a_win, team_a), (p_draw, "Empate"), (p_b_win, team_b)])
        print(f"\nPredicción: {best[1]}")


if __name__ == "__main__":
    main()
