import pandas as pd
from config import CAT_FEATURES, PSP_COSTS, OHC_PATH
import joblib


def calculate_success_probability(model, features: pd.DataFrame) -> float:
    """Calculates the success probability for a given model and features."""
    return model.predict_proba(features)[:, 1]


def _prepare_features_for_psp(transactions: pd.DataFrame, psp: str) -> pd.DataFrame:
    """Prepares a feature set for a given PSP."""
    features = transactions.copy()

    # Set cost features to corresponding PSP costs
    features["cost_if_success"] = PSP_COSTS[psp]["success"]
    features["cost_if_failure"] = PSP_COSTS[psp]["failure"]

    one_hot_encoder = joblib.load(OHC_PATH)
    all_feature_names = one_hot_encoder.get_feature_names_out(CAT_FEATURES)
    psp_ohe_columns = [name for name in all_feature_names if name.startswith("PSP_")]

    # Simulate the choice of the PSP
    for col in psp_ohe_columns:
        if col in features.columns:
            features[col] = 0
    features[f"PSP_{psp}"] = 1

    # Re-align columns with the model's training data
    columns = transactions.columns.tolist()
    return features.reindex(columns=transactions.columns, fill_value=0)


def calculate_expected_costs(transactions: pd.DataFrame, model) -> pd.DataFrame:
    """Calculates the expected cost for each transaction for all possible PSPs."""
    expected_costs_df = pd.DataFrame(index=transactions.index)

    for psp in PSP_COSTS:
        # Prepare features for the current PSP
        features = _prepare_features_for_psp(transactions, psp)

        # Predict success probability
        prob_success = calculate_success_probability(model, features)

        expected_costs_df[psp] = (
            prob_success * features["cost_if_success"]
            + (1 - prob_success) * features["cost_if_failure"]
        )

    return expected_costs_df
