COSTS = {
    "Moneycard": {"success": 5, "failure": 2},
    "Goldcard": {"success": 10, "failure": 5},
    "UK_Card": {"success": 3, "failure": 1},
    "Simplecard": {"success": 1, "failure": 0.5},
}

TIME_FEATURES = {"day": 31, "dow": 7, "hour": 24}

DATA_PATH = "data/processed_data.csv"
MODEL_PATH = "models/decision_tree_baseline.joblib"
