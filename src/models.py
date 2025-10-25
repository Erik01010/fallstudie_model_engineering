import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from src.config import PARAM_DIST


def train_ohc_encoder(data: pd.DataFrame) -> OneHotEncoder:
    """Trains and saves a OneHotEncoder for categorical features. Only for training data!"""
    one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown="warn")
    one_hot_encoder.fit(data)
    return one_hot_encoder


def train_decision_tree(
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> DecisionTreeClassifier:
    """Trains a Decision Tree Classifier."""
    decision_tree_model = DecisionTreeClassifier(max_depth=5, random_state=42)
    decision_tree_model.fit(x_train, y_train)

    return decision_tree_model


def train_hgboost(
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> HistGradientBoostingClassifier:
    """Trains a HGBoost Classifier."""
    hgboost_model = HistGradientBoostingClassifier(
        random_state=42, class_weight="balanced", verbose=1
    )
    hgboost_model.fit(x_train, y_train)

    return hgboost_model


def tune_hyperparameters(
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> HistGradientBoostingClassifier:
    """Run randomized search to find the best hyperparameters."""
    hgboost_model = HistGradientBoostingClassifier(
        random_state=42, class_weight="balanced", verbose=1
    )
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    random_search = RandomizedSearchCV(
        estimator=hgboost_model,
        param_distributions=PARAM_DIST,
        n_iter=100,
        cv=cv_strategy,
        scoring="f1",
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )

    random_search.fit(x_train, y_train)

    print(f"Beste Parameter gefunden: {random_search.best_params_}")

    final_model = HistGradientBoostingClassifier(
        **random_search.best_params_, random_state=42, class_weight="balanced"
    )
    final_model.fit(x_train, y_train)

    return final_model
