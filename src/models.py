import joblib
import pandas as pd
from config import OHC_PATH, DECISION_TREE_PATH, XGBOOST_PATH
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingClassifier


def train_ohc_encoder(data: pd.DataFrame) -> OneHotEncoder:
    """Trains and saves a OneHotEncoder for categorical features."""
    one_hot_encoder = OneHotEncoder(sparse_output=False)
    one_hot_encoder.fit_transform(data)
    joblib.dump(one_hot_encoder, OHC_PATH)
    return one_hot_encoder


def train_decision_tree(
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> DecisionTreeClassifier:
    """Trains a Decision Tree Classifier."""
    baseline_model = DecisionTreeClassifier(max_depth=5, random_state=42)
    baseline_model.fit(x_train, y_train)
    joblib.dump(baseline_model, DECISION_TREE_PATH)

    return baseline_model


def train_xgboost(
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> HistGradientBoostingClassifier:
    """Trains a XGBoost Classifier."""
    final_model = HistGradientBoostingClassifier(
        max_leaf_nodes=30, learning_rate=0.05, random_state=42
    )
    final_model.fit(x_train, y_train)
    joblib.dump(final_model, XGBOOST_PATH)

    return final_model
