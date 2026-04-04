"""
config.py — Project-wide path configuration
============================================
All scripts in src/ import paths from here so that the project
can be reorganised without hunting for hardcoded strings.
"""
from pathlib import Path

# Root of the repository (parent of src/)
ROOT = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR   = ROOT / "data"
WISDM_DIR  = DATA_DIR / "wisdm" / "WISDM_ar_v1.1"
WISDM_RAW  = WISDM_DIR / "WISDM_ar_v1.1_raw.txt"
UCI_DIR    = DATA_DIR / "ucihar" / "UCI HAR Dataset"

# Output directories
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"
