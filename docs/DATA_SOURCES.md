# Data Sources

Este documento registra las fuentes de datos utilizadas por el proyecto
Football Match Prediction Pipeline.

---

## International Football Results (Kaggle)

**Estado:** Integrada. Es la única fuente utilizada actualmente.

**Autor:** Mart Jürisoo

**Tipo de datos:**
- Resultados históricos de selecciones nacionales
- Fecha del partido
- Equipo local y visitante
- Goles de cada equipo
- Competición
- Sede y condición de neutralidad

**Cobertura:** Partidos internacionales masculinos desde 1872 hasta la actualidad.

**Costo:** Gratuito.

**Uso:** Dataset base para entrenamiento del predictor de selecciones nacionales.

**Licencia:** Revisar los términos de uso oficiales de Kaggle antes de redistribuir el archivo.

**Pipeline:** raw → processed → features

**Validaciones aplicadas:**
- Columnas requeridas presentes (`src/validation/validate_raw_data.py`).
- Reparación de filas mal parseadas por comas en el campo `city`.
- Descarte de partidos sin marcador (fixtures futuros).

---

## Fuentes evaluadas y descartadas

**Football-Data.co.uk** se evaluó al comienzo del proyecto como fuente
alternativa de resultados de clubes, pero no se integró: el proyecto se
enfocó en selecciones nacionales, para las cuales el dataset de Kaggle
ya ofrece la cobertura histórica necesaria.
