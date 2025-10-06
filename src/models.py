import pandas as pd
from sklearn.metrics import accuracy_score

from sklearn import tree
from sklearn.model_selection import train_test_split
import joblib
from config import MODEL_PATH


def split_features_and_target(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits the dataset into features and target variable."""
    y = data["success"]
    X = data.drop(columns=["success"], axis=1)
    joblib.dump(X.columns, "models/columns.joblib")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def train_decision_tree(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tree.DecisionTreeClassifier:
    """Trains a Decision Tree Classifier and evaluates its accuracy."""
    baseline_model = tree.DecisionTreeClassifier(max_depth=5, random_state=42)
    baseline_model.fit(X_train, y_train)
    y_pred_test = baseline_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred_test)
    joblib.dump(baseline_model, MODEL_PATH)

    print(f"Model accuracy: {accuracy:.4f}")
    return baseline_model
