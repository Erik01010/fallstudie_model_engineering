import pandas as pd
from predictions import calculate_expected_costs


def calculate_total_cost(x_test: pd.DataFrame, model) -> None:
    """
    Evaluates the model on the test set by calculating the total cost.
    """
    # Calculate expected costs for all transactions in a vectorized way
    expected_costs = calculate_expected_costs(transactions=x_test, model=model)

    # Find the minimum cost for each transaction and sum them up
    total_cost = expected_costs.min(axis=1).sum()

    print(f"Total cost on test set: {total_cost:.2f}")
