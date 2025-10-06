import joblib
import pandas as pd
from config import MODEL_PATH
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def train_ohc_encoder(data: pd.DataFrame) -> OneHotEncoder:
    """Trains and saves a OneHotEncoder for categorical features."""
    one_hot_encoder = OneHotEncoder(sparse_output=False)
    one_hot_encoder.fit_transform(data)
    joblib.dump(one_hot_encoder, "models/one_hot_encoder.joblib")
    return one_hot_encoder


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
    joblib.dump(baseline_model, MODEL_PATH)

    return baseline_model
