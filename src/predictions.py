import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from src.config import PSP_COSTS
from src.features import create_categorial_features
from src.features import engineer_features

ModelType = HistGradientBoostingClassifier | DecisionTreeClassifier


def _get_all_predictions(
    model: ModelType,
    x_test: pd.DataFrame,
    original_data: pd.DataFrame,
    encoder: OneHotEncoder,
) -> pd.DataFrame:
    """Predict values for each psp on the test set."""
    predictions_df = pd.DataFrame()
    predictions_df["original_psp"] = original_data.loc[x_test.index, "PSP"]
    for simulated_psp in PSP_COSTS:
        simulated_data = original_data.loc[x_test.index].copy()
        simulated_data = simulated_data.drop(columns=["success"], axis=1)
        simulated_data["PSP"] = simulated_psp
        simulated_data = create_categorial_features(data=simulated_data)
        simulated_features = engineer_features(data=simulated_data, encoder=encoder)
        predictions_psp = model.predict_proba(simulated_features)[:, 1]
        predictions_df[simulated_psp] = predictions_psp
    return predictions_df


def _calculate_strategy_kpis(
        choices: pd.Series,
        y_test: pd.Series,
        predictions_df: pd.DataFrame
) -> tuple[int, float, float]:
    """Calculate the total actual cost and success rate for a series of PSP choices."""
    total_cost = 0
    successful_transactions = 0
    expected_probabilities = []

    for index, psp_choice in choices.items():
        if y_test.loc[index] == 1:
            successful_transactions += 1
            total_cost += PSP_COSTS[psp_choice]["success"]
        else:
            total_cost += PSP_COSTS[psp_choice]["failure"]

        expected_prob = predictions_df.loc[index, psp_choice]
        expected_probabilities.append(expected_prob)

    success_rate = (successful_transactions / len(y_test)) * 100
    avg_expected_success_rate = (
        (sum(expected_probabilities) / len(expected_probabilities)) * 100
    )

    return total_cost, success_rate, avg_expected_success_rate


def min_expected_cost(predictions_df: pd.DataFrame) -> pd.Series:
    """Calculate the PSP with min expected Costs for each Transaction."""
    data = predictions_df.copy()
    psps = list(PSP_COSTS.keys())
    for column in psps:
        probs = data[column]
        cost_success = PSP_COSTS[column]["success"]
        cost_failure = PSP_COSTS[column]["failure"]

        data[column] = probs * cost_success + (1 - probs) * cost_failure
    return data[psps].idxmin(axis=1)


def max_success_prob(predictions_df: pd.DataFrame) -> pd.Series:
    """Calculate the PSP with max expected success rate for each Transaction."""
    psps = list(PSP_COSTS.keys())
    return predictions_df[psps].idxmax(axis=1)


def evaluate_business_strategies(
    model: ModelType,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    original_data: pd.DataFrame,
    encoder: OneHotEncoder,
) -> pd.DataFrame:
    """Simulate and evaluate different business strategies and compares to the legacy system."""
    predictions_df = _get_all_predictions(model, x_test, original_data, encoder)
    legacy_choices = original_data.loc[x_test.index, "PSP"]
    cost_optimized_choices = min_expected_cost(predictions_df)
    success_optimized_choices = max_success_prob(predictions_df)

    (
        legacy_cost,
        legacy_actual_sr,
        legacy_expected_sr,
    ) = _calculate_strategy_kpis(legacy_choices, y_test, predictions_df)
    (
        cost_opt_cost,
        cost_opt_actual_sr,
        cost_opt_expected_sr,
    ) = _calculate_strategy_kpis(cost_optimized_choices, y_test, predictions_df)
    (
        success_opt_cost,
        success_opt_actual_sr,
        success_opt_expected_sr,
    ) = _calculate_strategy_kpis(success_optimized_choices, y_test, predictions_df)
    results = {
        "Legacy System": {
            "Actual Success Rate": legacy_actual_sr,
            "Avg. Expected Success Rate": legacy_expected_sr,
            "Total Cost": legacy_cost,
        },
        "Cost-Optimized Model": {
            "Actual Success Rate": cost_opt_actual_sr,
            "Avg. Expected Success Rate": cost_opt_expected_sr,
            "Total Cost": cost_opt_cost,
        },
        "Success-Optimized Model": {
            "Actual Success Rate": success_opt_actual_sr,
            "Avg. Expected Success Rate": success_opt_expected_sr,
            "Total Cost": success_opt_cost,
        },
    }

    return pd.DataFrame(results).T
