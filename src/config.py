from pathlib import Path

# Base project directory (repo root)
BASE_DIR = Path(__file__).resolve().parents[1]

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw_data"
PROCESSED_DATA_DIR = DATA_DIR / "processed_data"

# Dataset-specific directories
C_MAPSS_DIR = RAW_DATA_DIR / "C_MAPSS"
IMS_DIR = RAW_DATA_DIR / "IMS"
NASA_BATTERY_DIR = RAW_DATA_DIR / "Nasa_Battery"
NASA_MILLING_DIR = RAW_DATA_DIR / "Nasa_Milling"
PRONOSTIA_DIR = RAW_DATA_DIR / "Pronostia"

# Outputs
RESULTS_DIR = BASE_DIR / "results"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# Auto-create dirs if missing
for d in [RESULTS_DIR, PROCESSED_DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

