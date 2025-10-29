from pathlib import Path

from scipy.stats import randint
from scipy.stats import uniform

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed_data.csv"
RAW_DATA_PATH = BASE_DIR / "data" / "data.xlsx"
DIAGRAMS_PATH = BASE_DIR / "diagrams"

DECISION_TREE_PATH = BASE_DIR / "models" / "decision_tree.joblib"
HGBOOST_PATH = BASE_DIR / "models" / "hgboost.joblib"
HGBOOST_OPTIMIZED_PATH = BASE_DIR / "models" / "hgboost_optimized.joblib"


CAT_FEATURES = [
    "PSP",
    "country",
    "card",
    "amount_bins",
    "interaction_psp_country",
    "interaction_psp_card",
    "interaction_psp_amount_bin",
    "interaction_psp_3D_secured",
]

CYCLICAL_FEATURES = {"day": 31, "dow": 7, "hour": 24}

PSP_COSTS = {
    "Moneycard": {"success": 5, "failure": 2},
    "Goldcard": {"success": 10, "failure": 5},
    "UK_Card": {"success": 3, "failure": 1},
    "Simplecard": {"success": 1, "failure": 0.5},
}

PARAM_DIST = {
    "learning_rate": uniform(0.01, 0.2),
    "max_iter": randint(100, 500),
    "max_depth": randint(3, 15),
    "l2_regularization": uniform(0, 1),
    "min_samples_leaf": randint(20, 100),
}
