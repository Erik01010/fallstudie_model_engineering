import pandas as pd
from sklearn.model_selection import train_test_split
from predictions import find_best_psp


def calculate_total_cost(model, data: pd.DataFrame) -> None:
    """Evaluate the model on a test set."""
    y = data["success"]
    X = data.drop(columns=["success"], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    total_cost = 0
    for i in range(len(X_test)):
        transaction = X_test.iloc[i]
        best_psp, min_cost = find_best_psp(transaction, model)
        total_cost += min_cost

    print(f"Total cost on test set: {total_cost:.2f}")
