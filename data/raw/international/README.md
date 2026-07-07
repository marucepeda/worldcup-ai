# Raw International Data

Esta carpeta contiene el archivo original descargado desde Kaggle
(ver `docs/DATA_SOURCES.md`).

## Reglas

- No modificar el archivo original.
- Mantener el nombre original.
- Registrar la fuente en `docs/DATASET_CATALOG.md`.
- Las transformaciones se realizan fuera de esta carpeta
  (`src/processing/build_matches_dataset.py`).

## Contenido

- `international_results.csv`: partidos internacionales de selecciones
  nacionales desde 1872 hasta la actualidad.

Flujo de datos:

raw → processed → features
