import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from src.config import PSP_COSTS
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

from src.features import engineer_features, create_categorial_features
from typing import Union


ModelType = Union[HistGradientBoostingClassifier, DecisionTreeClassifier]


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


def calculate_strategy_kpis(choices: pd.Series, y_test: pd.Series) -> tuple[float, int]:
    """Calculate the total actual cost and success rate for a series of PSP choices."""
    total_cost = 0
    successful_transactions = 0

    for index, psp_choice in choices.items():
        if y_test.loc[index] == 1:
            successful_transactions += 1
            total_cost += PSP_COSTS[psp_choice]["success"]
        else:
            total_cost += PSP_COSTS[psp_choice]["failure"]

    success_rate = (successful_transactions / len(y_test)) * 100

    return success_rate, total_cost
