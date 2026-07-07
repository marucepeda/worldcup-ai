# Data Schema

## Tabla: international_matches (processed)

Tabla de partidos internacionales, ya limpia, usada como base para
todo el feature engineering.

Ubicación: `data/processed/international/international_matches.csv`

| Campo | Tipo | Descripción |
|---|---|---|
| match_id | integer | Identificador único del partido |
| date | date | Fecha del partido |
| home_team | string | Selección local |
| away_team | string | Selección visitante |
| home_goals | integer | Goles del equipo local |
| away_goals | integer | Goles del equipo visitante |
| competition | string | Competición del partido |
| neutral_venue | boolean | Indica si el partido fue en sede neutral |

### Reglas de calidad

- No hay partidos duplicados.
- Los goles son valores enteros no nulos (los partidos sin marcador se descartan).
- La fecha tiene formato válido.
- Los equipos no están vacíos.

---

## Tabla: match_dataset (features finales)

Dataset final que consumen los modelos. Se arma en
`src/features/build_match_dataset.py` combinando `international_matches`
con las tres tablas de features intermedias (`team_stats.csv`,
`recent_form.csv`, `elo_ratings.csv`).

Ubicación: `data/features/international/match_dataset.csv`

| Grupo | Campos |
|---|---|
| Identificación | match_id, date, home_team, away_team, competition, neutral_venue |
| Resultado | home_goals, away_goals, result_class |
| Estadísticas históricas | home_avg_goals_for, home_avg_goals_against, home_matches_played, away_avg_goals_for, away_avg_goals_against, away_matches_played |
| Forma reciente (últimos 5 partidos) | home_recent_goals_for, home_recent_goals_against, home_recent_wins, home_recent_draws, home_recent_losses, away_recent_* (ídem) |
| Elo | home_elo, away_elo, elo_difference |
| Comparativas | attack_difference, defense_difference, form_difference, recent_attack_difference |

`result_class` es la variable objetivo: `2` si gana el local, `1` si
empatan, `0` si gana el visitante.

Todas las features de historial, forma y Elo se calculan **point-in-time**:
usan únicamente partidos anteriores a la fecha del partido que describen,
para evitar data leakage (ver `docs/../REFORM_REPORT.md`).

---

## Flujo de datos

```
raw → processed → features (team_stats, recent_form, elo_ratings) → match_dataset
```
