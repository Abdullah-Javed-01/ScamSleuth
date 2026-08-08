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

TRAIN_DATA_PATH = SPLIT_DIR / "train.csv"
VALIDATION_DATA_PATH = SPLIT_DIR / "validation.csv"
TEST_DATA_PATH = SPLIT_DIR / "test.csv"

FINAL_MODEL_PATH = (
    ARTIFACT_DIR / "scamsleuth_model.joblib"
)

FINAL_MODEL_METADATA_PATH = (
    ARTIFACT_DIR / "scamsleuth_model_metadata.json"
)

# Reproducibility
RANDOM_SEED = 42
DECISION_THRESHOLD = 0.31