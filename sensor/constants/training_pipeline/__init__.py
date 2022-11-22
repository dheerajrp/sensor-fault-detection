import os

# defining common constants for training pipeline
TARGET_COLUMN: str = "class"
PIPELINE_NAME: str = "sensor"
ARTIFACT_DIR: str = "artifact"
FILE_NAME: str = "sensor.csv"

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

PREPROCESSING_OBJECT_FILENAME: str = "preprocessing.pkl"
MODEL_FILE_NAME: str = "model.pkl"
SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")
SCHEMA_DROP_COLS: str = "drop_columns"

# data ingestion related constants
DATA_INGESTION_COLLECTION_NAME: str = "car"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO = 0.2

# data validation related constants
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "valid"
DATA_VALIDATION_INVALID_DIR: str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"
