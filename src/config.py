"""
Rutas de datos centralizadas del proyecto.

Todos los scripts del pipeline importan sus rutas de entrada/salida
desde este módulo en lugar de repetir strings sueltos. Así, si la
estructura de carpetas cambia, solo hay que actualizar un lugar.
"""

from pathlib import Path

# Carpeta raíz del proyecto (dos niveles arriba de este archivo: src/config.py -> raíz)
ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw" / "international"
PROCESSED_DIR = DATA_DIR / "processed" / "international"
FEATURES_DIR = DATA_DIR / "features" / "international"

# Datos crudos
RAW_MATCHES_FILE = RAW_DIR / "international_results.csv"

# Datos procesados
PROCESSED_MATCHES_FILE = PROCESSED_DIR / "international_matches.csv"

# Features intermedias (una tabla por tipo de feature, indexada por match_id)
TEAM_STATS_FILE = FEATURES_DIR / "team_stats.csv"
RECENT_FORM_FILE = FEATURES_DIR / "recent_form.csv"
ELO_RATINGS_FILE = FEATURES_DIR / "elo_ratings.csv"

# Dataset final para entrenamiento y evaluación de modelos
MATCH_DATASET_FILE = FEATURES_DIR / "match_dataset.csv"
