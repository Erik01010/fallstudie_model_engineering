import pandas as pd
from sklearn.metrics import accuracy_score

from sklearn import tree
from sklearn.model_selection import train_test_split
import joblib
from config import MODEL_PATH


def train_decision_tree(data: pd.DataFrame) -> tree.DecisionTreeClassifier:
    y = data["success"]
    X = data.drop(columns=["success", "country", "card"], axis=1)
    joblib.dump(X.columns, "models/columns.joblib")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    baseline_model = tree.DecisionTreeClassifier(random_state=42)
    baseline_model.fit(X_train, y_train)
    y_pred_test = baseline_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred_test)
    joblib.dump(baseline_model, MODEL_PATH)

    print(f"Model accuracy: {accuracy:.4f}")
    return baseline_model
