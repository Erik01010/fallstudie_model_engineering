import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from config import PSP_COSTS

# Load pre-fitted encoder and training columns
one_hot_encoder = joblib.load("models/one_hot_encoder.joblib")
columns = joblib.load("models/columns.joblib")


def _prepare_features_for_psp(transactions: pd.DataFrame, psp: str) -> pd.DataFrame:
    """Prepares a feature set for a given PSP."""
    features = transactions.copy()

    # Set cost features to corresponding PSP costs
    features["cost_if_success"] = PSP_COSTS[psp]["success"]
    features["cost_if_failure"] = PSP_COSTS[psp]["failure"]

    psp_ohe_columns = [f"PSP_{psp}" for psp in PSP_COSTS.keys()]

    # Simulate the choice of the PSP
    for col in psp_ohe_columns:
        if col in features.columns:
            features[col] = 0
    features[f"PSP_{psp}"] = 1

    # Re-align columns with the model's training data
    return features.reindex(columns=columns, fill_value=0)


def calculate_expected_costs(
    transactions: pd.DataFrame, model: DecisionTreeClassifier
) -> pd.DataFrame:
    """Calculates the expected cost for each transaction across all possible PSPs."""
    expected_costs_df = pd.DataFrame(index=transactions.index)

    for psp in PSP_COSTS:
        # Prepare features for the current PSP
        features = _prepare_features_for_psp(transactions, psp)

        # Predict success probability
        prob_success = model.predict_proba(features)[:, 1]

        expected_costs_df[psp] = (
            prob_success * features["cost_if_success"]
            + (1 - prob_success) * features["cost_if_failure"]
        )

    return expected_costs_df


def find_best_psp(
    transaction: pd.Series, model: DecisionTreeClassifier
) -> tuple[str, float]:
    """Finds the best PSP for a single transaction."""
    transaction_df = transaction.to_frame().T

    expected_costs = calculate_expected_costs(transaction_df, model)

    best_psp = expected_costs.idxmin(axis=1).iloc[0]
    min_cost = expected_costs.min(axis=1).iloc[0].astype(float)

    return best_psp, min_cost
