"""
Punto de entrada único para correr todo el pipeline de punta a punta:

  1. Valida el dataset crudo.
  2. Construye el dataset procesado.
  3. Construye las features (estadísticas históricas, forma reciente, Elo).
  4. Ensambla el dataset final de Machine Learning.
  5. Entrena y evalúa los cuatro modelos.

Uso:
    python run_pipeline.py
"""

from src.features import build_match_dataset, elo_rating, recent_form, team_stats
from src.models import baseline, logistic_regression, logistic_regression_balanced, random_forest
from src.processing import build_matches_dataset
from src.validation import validate_raw_data


def main() -> None:
    steps = [
        ("Validación de datos crudos", validate_raw_data.main),
        ("Construcción del dataset procesado", build_matches_dataset.main),
        ("Estadísticas históricas por equipo", team_stats.main),
        ("Forma reciente por equipo", recent_form.main),
        ("Rating Elo", elo_rating.main),
        ("Dataset final de Machine Learning", build_match_dataset.main),
        ("Modelo: Baseline", baseline.main),
        ("Modelo: Logistic Regression", logistic_regression.main),
        ("Modelo: Logistic Regression (Balanced)", logistic_regression_balanced.main),
        ("Modelo: Random Forest", random_forest.main),
    ]

    for title, step in steps:
        print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")
        step()


if __name__ == "__main__":
    main()
