import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from config import PSP_COSTS

# Load pre-fitted encoder and training columns
one_hot_encoder = joblib.load("models/one_hot_encoder.joblib")
columns = joblib.load("models/columns.joblib")


def _prepare_features_for_psp(
    transactions: pd.DataFrame, psp: str, psp_ohe_columns: list[str]
) -> pd.DataFrame:
    """
    Prepares a feature set for a given PSP.

    Args:
        transactions: The input transactions.
        psp: The PSP to prepare features for.
        psp_ohe_columns: A list of one-hot encoded PSP column names.

    Returns:
        A DataFrame ready for prediction.
    """
    features = transactions.copy()

    # Set cost features for the given PSP
    features["cost_if_success"] = PSP_COSTS[psp]["success"]
    features["cost_if_failure"] = PSP_COSTS[psp]["failure"]

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
    """
    Calculates the expected cost for each transaction across all possible PSPs.

    Args:
        transactions: A DataFrame of transactions.
        model: The trained model for predicting success probabilities.

    Returns:
        A DataFrame of expected costs, with transactions as rows and PSPs as
        columns.
    """
    expected_costs_df = pd.DataFrame(index=transactions.index)
    psp_ohe_columns = [
        col
        for col in one_hot_encoder.get_feature_names_out(["country", "card", "PSP"])
        if "PSP_" in col
    ]

    for psp in PSP_COSTS:
        # Prepare features for the current PSP
        features = _prepare_features_for_psp(transactions, psp, psp_ohe_columns)

        # Predict success probability
        prob_success = model.predict_proba(features)[:, 1]

        # Calculate and store the expected cost
        cost_success = PSP_COSTS[psp]["success"]
        cost_failure = PSP_COSTS[psp]["failure"]
        expected_costs_df[psp] = (
            prob_success * cost_success + (1 - prob_success) * cost_failure
        )

    return expected_costs_df


def find_best_psp(
    transaction: pd.Series, model: DecisionTreeClassifier
) -> tuple[str, float]:
    """
    Finds the best PSP for a single transaction.

    Args:
        transaction: A Series representing a single transaction.
        model: The trained DecisionTreeClassifier model.

    Returns:
        A tuple with the best PSP's name and the minimum expected cost.
    """
    transaction_df = transaction.to_frame().T
    expected_costs = calculate_expected_costs(transaction_df, model)
    best_psp = expected_costs.idxmin(axis=1).iloc[0]
    min_cost = expected_costs.min(axis=1).iloc[0]
    return best_psp, min_cost