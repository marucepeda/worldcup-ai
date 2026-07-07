# Football Match Prediction Pipeline ⚽🤖

## Overview

Proyecto personal desarrollado para aprender y aplicar Ingeniería de
Datos y Machine Learning, construyendo un pipeline completo para la
predicción de resultados de partidos internacionales de fútbol.

El proyecto cubre todo el flujo de trabajo: ingesta y validación de
datos, procesamiento, feature engineering, generación de un dataset de
Machine Learning y entrenamiento/evaluación de modelos, siguiendo una
arquitectura modular.

No busca ser el mejor predictor de fútbol del mundo. Su objetivo es
demostrar la capacidad de construir un pipeline de datos reproducible
de punta a punta, con buenas prácticas de ingeniería.

---

## Objectives

* Construir un pipeline reproducible de procesamiento de datos.
* Aplicar Feature Engineering para mejorar el desempeño de los modelos.
* Comparar distintos algoritmos de Machine Learning.
* Incorporar métricas futbolísticas: estadísticas históricas, forma
  reciente y sistema de rating Elo.
* Evitar data leakage: todas las features se calculan point-in-time.

---

## Architecture

```text
worldcup-ai/
│
├── analysis/              # Scripts exploratorios (no forman parte del pipeline)
├── data/
│   ├── raw/                # Dataset original, sin modificar
│   ├── processed/          # Partidos limpios y estandarizados
│   └── features/           # Tablas de features + dataset final de ML
│
├── docs/                   # Catálogo de datasets, schema, fuentes
├── src/
│   ├── config.py           # Rutas de datos centralizadas
│   ├── processing/         # raw -> processed
│   ├── validation/         # Chequeos de calidad sobre el dataset crudo
│   ├── features/           # processed -> features -> dataset de ML
│   └── models/              # Entrenamiento y evaluación de modelos
│
├── tests/                  # Tests unitarios e de integración (pytest)
├── run_pipeline.py         # Corre todo el pipeline de punta a punta
└── requirements.txt
```

Cada etapa lee y escribe archivos CSV dentro de `data/`, lo que permite
inspeccionar el resultado de cada paso de forma independiente sin tener
que ejecutar todo el pipeline.

---

## Pipeline

1. **Validación de datos crudos** (`src/validation`) — chequea columnas
   requeridas y reporta problemas de calidad conocidos.
2. **Procesamiento** (`src/processing`) — limpia el dataset crudo:
   repara filas mal parseadas, descarta partidos sin marcador y
   estandariza nombres de columnas.
3. **Feature engineering** (`src/features`) — calcula, para cada
   partido, tres familias de features de forma independiente:
   * `team_stats.py`: promedio histórico de goles a favor/en contra.
   * `recent_form.py`: forma en los últimos 5 partidos.
   * `elo_rating.py`: rating Elo actualizado partido a partido.
4. **Dataset final** (`build_match_dataset.py`) — combina las tres
   tablas de features con los partidos, agrega las diferencias
   comparativas (local vs. visitante) y crea la variable objetivo.
5. **Entrenamiento y evaluación** (`src/models`) — entrena los cuatro
   modelos con un split cronológico (80% partidos más antiguos para
   entrenar, 20% más recientes para evaluar) y reporta accuracy y
   classification report.

Correr todo de punta a punta:

```bash
python run_pipeline.py
```

O ejecutar cada paso por separado (útil mientras se desarrolla o depura
una etapa puntual):

```bash
python -m src.validation.validate_raw_data
python -m src.processing.build_matches_dataset
python -m src.features.team_stats
python -m src.features.recent_form
python -m src.features.elo_rating
python -m src.features.build_match_dataset
python -m src.models.baseline
python -m src.models.logistic_regression
python -m src.models.logistic_regression_balanced
python -m src.models.random_forest
```

---

## Feature Engineering

El proyecto genera variables relacionadas con:

* Promedio histórico de goles a favor y en contra.
* Cantidad de partidos disputados.
* Rendimiento en los últimos 5 partidos (goles, victorias, empates, derrotas).
* Diferencias ofensivas y defensivas entre el equipo local y el visitante.
* Diferencia de forma reciente.
* Sistema de rating Elo, actualizado partido a partido.

**Importante — sin data leakage:** las estadísticas históricas, la
forma reciente y el rating Elo de un equipo se calculan usando
únicamente los partidos anteriores a la fecha del partido que se está
describiendo (usando `shift` + `expanding`/`rolling` sobre la serie
temporal de cada equipo). Un partido nunca "ve" información de
partidos futuros.

---

## Models

Se entrenan y comparan cuatro modelos:

* **Baseline** — siempre predice que gana el equipo local.
* **Logistic Regression** — con todas las features escaladas.
* **Logistic Regression (Balanced)** — igual, con `class_weight="balanced"`
  para compensar que el target está desbalanceado (el local gana con
  más frecuencia que empata o pierde).
* **Random Forest** — no lineal, no requiere escalado.

---

## Technologies

* Python
* Pandas / NumPy
* Scikit-learn
* Pytest
* Git

---

## Current Results

Split cronológico 80/20 sobre ~49.500 partidos internacionales (1872-2026):

| Modelo | Accuracy |
|---|---|
| Baseline (siempre gana el local) | 0.49 |
| Logistic Regression | 0.60 |
| Logistic Regression (Balanced) | 0.55 |
| Random Forest | 0.58 |

La regresión logística balanceada tiene menor accuracy global, pero
mucho mejor recall en empates y victorias visitantes (ver
`classification_report` al correr cada script) — un ejemplo concreto
de por qué accuracy sola no alcanza para elegir un modelo cuando las
clases están desbalanceadas.

El objetivo del proyecto no es maximizar la accuracy, sino construir un
pipeline robusto, reproducible y libre de data leakage sobre el cual
seguir iterando.

---

## Installation

```bash
git clone <repo-url>
cd worldcup-ai
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running tests

```bash
pytest
```

---

## Future Improvements

* Incorporar ranking FIFA como feature adicional.
* Optimizar el K-factor y el rating inicial del sistema Elo.
* Probar Gradient Boosting (XGBoost / LightGBM) como quinto modelo de comparación.
* Exportar el modelo entrenado (ej. con `joblib`) para reutilizarlo sin reentrenar.
* Validación cruzada temporal (walk-forward) en vez de un único split 80/20.
* API simple con FastAPI para exponer predicciones.

---

## Author

Desarrollado como proyecto de aprendizaje en Ingeniería de Datos y
Machine Learning, con foco en buenas prácticas de desarrollo,
arquitectura de datos y construcción de pipelines reproducibles.
