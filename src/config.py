from pathlib import Path

PSP_COSTS = {
    "Moneycard": {"success": 5, "failure": 2},
    "Goldcard": {"success": 10, "failure": 5},
    "UK_Card": {"success": 3, "failure": 1},
    "Simplecard": {"success": 1, "failure": 0.5},
}
CAT_FEATURES = ["country", "card", "PSP"]

CYCLICAL_FEATURES = {"day": 31, "dow": 7, "hour": 24}

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed_data.csv"
RAW_DATA_PATH = BASE_DIR / "data" / "data.xlsx"
MODEL_PATH = BASE_DIR / "models" / "decision_tree_baseline.joblib"
OHC_PATH = BASE_DIR / "models" / "one_hot_encoder.joblib"
