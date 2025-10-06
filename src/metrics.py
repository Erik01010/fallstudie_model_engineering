import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from predictions import calculate_expected_costs


def calculate_total_cost(
    model: DecisionTreeClassifier, X_test: pd.DataFrame, y_test: pd.Series
) -> None:
    """
    Evaluates the model on the test set by calculating the total cost and
    accuracy.

    This function uses a vectorized approach to calculate the expected cost for
    each transaction across all PSPs. It then finds the minimum cost for each
    transaction and sums them to get the total cost. It also calculates and
    prints the model's accuracy on the test set.

    Args:
        model: The trained DecisionTreeClassifier model.
        X_test: The test set features.
        y_test: The true labels for the test set.
    """
    # Calculate expected costs for all transactions in a vectorized way
    expected_costs = calculate_expected_costs(X_test, model)

    # Find the minimum cost for each transaction and sum them up
    total_cost = expected_costs.min(axis=1).sum()

    # Calculate accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Model accuracy on test set: {accuracy:.4f}")
    print(f"Total cost on test set: {total_cost:.2f}")
