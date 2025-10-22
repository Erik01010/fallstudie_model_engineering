import pandas as pd
from config import CAT_FEATURES, OHC_PATH, PSP_COSTS
from sklearn.metrics import roc_auc_score

from src.metrics import calculate_success_probability


def evaluate_technical_performance(
    model, x_test: pd.DataFrame, y_test: pd.DataFrame
) -> None:
    """Evaluates the model on the test set by calculating accuracy."""
    y_pred_proba = calculate_success_probability(model, x_test)
    score = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC AUC Score: {score:.4f}")


def _calculate_actual_costs(choices, y_true) -> float:
    """
    Helper function to calculate the total actual cost for a series of PSP choices
    based on the true transaction outcomes.
    """
    total_cost = 0

    for index, psp_choice in choices.items():
        if psp_choice in PSP_COSTS:
            cost_dict = PSP_COSTS[psp_choice]
            # Use the true outcome (y_true) to determine the actual cost
            actual_cost = (
                cost_dict["success"] if y_true.loc[index] else cost_dict["failure"]
            )
            total_cost += actual_cost
    return total_cost


def evaluate_business_impact(
    model, x_test: pd.DataFrame, y_test: pd.DataFrame, original_data: pd.DataFrame
) -> None:
    """Evaluates and compares the financial outcome of the model's routing strategy
    against the legacy system's strategy on the test set."""
    all_model_columns = x_test.columns.tolist()
    expected_costs_df = pd.DataFrame(index=x_test.index)

    for psp in PSP_COSTS:
        simulated_features = x_test.copy()

        for col in all_model_columns:
            if col.startswith("PSP_"):
                simulated_features[col] = 0
        simulated_features[f"PSP_{psp}"] = 1
        simulated_features = simulated_features.reindex(
            columns=all_model_columns, fill_value=0
        )
        prob_success = calculate_success_probability(model, simulated_features)

        expected_costs_df[psp] = (
            prob_success * PSP_COSTS[psp]["success"]
            + (1 - prob_success) * PSP_COSTS[psp]["failure"]
        )

    # Calculate Model Strategy Cost
    model_choices = expected_costs_df.idxmin(axis=1)
    total_cost_model = _calculate_actual_costs(model_choices, y_test)

    # Calculate Legacy System Cost
    legacy_choices = original_data.loc[x_test.index, "PSP"]
    total_cost_legacy = _calculate_actual_costs(legacy_choices, y_test)

    # Report the Financial Outcome
    savings = total_cost_legacy - total_cost_model
    savings_percent = (
        (savings / total_cost_legacy) * 100 if total_cost_legacy > 0 else 0
    )

    print(f"  Legacy System Cost: {total_cost_legacy:,.2f} €")
    print(f"  Model Strategy Cost: {total_cost_model:,.2f} €")
    print(f"  Savings: {savings:,.2f} € ({savings_percent:.2f}%)")

    def evaluate_business_strategies():
        """Calculate all scenarios."""
        pass

    def _get_all_predictions():
        pass
