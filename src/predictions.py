from config import PSP_COSTS
import joblib
import pandas as pd

one_hot_encoder = joblib.load("models/one_hot_encoder.joblib")
columns = joblib.load("models/columns.joblib")


def find_best_psp(transaction: pd.Series, model) -> tuple[str, float]:
    expected_costs = {}

    features = transaction.copy()

    for psp in ["Moneycard", "Goldcard", "UK_Card", "Simplecard"]:
        features["cost_if_success"] = PSP_COSTS[psp]["success"]
        features["cost_if_failure"] = PSP_COSTS[psp]["failure"]

        # Create a copy of the transaction to modify PSP features
        features_for_prediction = transaction.copy().to_frame().T

        # Set all PSP one-hot encoded columns to 0
        for col in one_hot_encoder.get_feature_names_out(["country", "card", "PSP"]):
            if "PSP_" in col:
                features_for_prediction[col] = 0

        # Set the current PSP one-hot encoded column to 1
        psp_col_name = f"PSP_{psp}"
        if psp_col_name in features_for_prediction.columns:
            features_for_prediction[psp_col_name] = 1
        else:
            # If a PSP column is missing, add it and set to 1 (should not happen if encoder is consistent)
            features_for_prediction[psp_col_name] = 1

        # Reindex to ensure column order matches training data
        features_for_prediction = features_for_prediction.reindex(columns=columns, fill_value=0)

        prob_success = model.predict_proba(features_for_prediction)[0, 1]

        cost_success = PSP_COSTS[psp]["success"]
        cost_failure = PSP_COSTS[psp]["failure"]

        expected_cost = prob_success * cost_success + (1 - prob_success) * cost_failure
        expected_costs[psp] = expected_cost

    best_psp = min(expected_costs, key=expected_costs.get)

    return best_psp, expected_costs[best_psp]
