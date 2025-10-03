import pandas as pd
from feature_engineering import process_data
from sklearn import tree
from sklearn.model_selection import train_test_split


def calc_success_rate(df: pd.DataFrame) -> float:
    return round(df["success"].mean() * 100, 2)


def calc_avg_transaction_costs(df: pd.DataFrame) -> float:
    return round(df["cost"].mean(), 2)


data = process_data()
y = data["success", "PSP"]
X = data.drop(columns=["success", "PSP"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

decision_tree_model = tree.DecisionTreeClassifier(random_state=42)
decision_tree_model.fit(X_train, y_train)
preds = decision_tree_model.predict(X_test)


class BaseLineModel:
    """Class for Baseline model."""

    def __init__(self):
        self.model = tree.DecisionTreeClassifier(random_state=42)
        self.model.fit(X_train, y_train)
        self.model.predict(X_test)


if __name__ == "__main__":
    df = process_data()
    print(calc_success_rate(df=df))
    print(calc_avg_transaction_costs(df=df))
