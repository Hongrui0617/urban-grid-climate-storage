from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = PROJECT_ROOT / "data_raw"
DATA_PROCESSED = PROJECT_ROOT / "data_processed"
OUTPUTS = PROJECT_ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"
MODELS = OUTPUTS / "models"
LOGS = PROJECT_ROOT / "logs"

LOAD_FILE = DATA_RAW / "TEPCO_Load_2016_2022_hourly_with_units.csv"
WEATHER_FILE = DATA_RAW / "Tokyo_TempRH_2016_2022_hourly_wide_clean.csv"

MERGED_DATASET_CSV = DATA_PROCESSED / "merged_dataset_hourly.csv"
MERGED_DATASET_PARQUET = DATA_PROCESSED / "merged_dataset_hourly.parquet"

TARGET_COL = "load_mw"

FEATURE_COLS = [
    "temperature_c",
    "rh_percent",
    "hour",
    "dayofweek",
    "month",
    "is_weekend",
    "lag_1",
    "lag_24",
    "lag_168",
]

TRAIN_END = "2020-12-31 23:00:00"
VAL_END = "2021-12-31 23:00:00"
TEST_END = "2022-12-31 23:00:00"

LAGS = [1, 24, 168]

RANDOM_SEED = 42

# LSTM config
SEQ_LEN = 168   # one week
BATCH_SIZE = 64
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2
LEARNING_RATE = 1e-3
EPOCHS = 20
PATIENCE = 5
DEVICE = "cpu"   # change to "mps" on Apple Silicon if available
