from pathlib import Path

# Project root
ROOT_DIR = Path(__file__).resolve().parent

# Data paths
RAW_DATA_PATH = (
    ROOT_DIR
    / "data"
    / "raw"
    / "scamsleuth_dataset_v1.2_final.csv"
)

SPLIT_DIR = ROOT_DIR / "data" / "splits"
ARTIFACT_DIR = ROOT_DIR / "artifacts"

# Reproducibility
RANDOM_SEED = 42