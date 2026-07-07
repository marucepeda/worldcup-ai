# Dataset Catalog

Inventario de datasets utilizados en Football Match Prediction Pipeline.

---

# Dataset 001

## Nombre

International Football Results

## Fuente

Mart Jürisoo - International Football Results (Kaggle)

## Estado

Integrado. Es el dataset base del proyecto.

## Tipo

Resultados históricos de selecciones nacionales.

## Cobertura

Partidos internacionales masculinos desde 1872 hasta la actualidad (~49.500 partidos).

## Uso

Dataset base para entrenamiento y evaluación del predictor.

## Ubicación RAW

`data/raw/international/international_results.csv`

## Tabla destino

`data/processed/international/international_matches.csv`

## Notas de calidad conocidas

- 77 filas del CSV original quedan mal parseadas porque el nombre de
  algunas ciudades contiene una coma (ej. "Washington, D.C.") y el
  exportador envolvió la fila entera entre comillas. Se reparan en
  `src/processing/build_matches_dataset.py` volviendo a parsear esa
  fila con el módulo `csv` de Python.
- Un partido aparece sin marcador porque todavía no se jugó (fixture
  futuro). Se descarta, ya que no tiene una variable objetivo válida.
