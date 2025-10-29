import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from src.config import DECISION_TREE_PATH
from src.config import HGBOOST_OPTIMIZED_PATH
from src.config import HGBOOST_PATH
from src.config import PARAM_DIST


def train_ohc_encoder(data: pd.DataFrame) -> OneHotEncoder:
    """Trains and saves a OneHotEncoder for categorical features. Only for training data."""
    one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown="warn")
    one_hot_encoder.fit(data)
    return one_hot_encoder


def train_decision_tree(
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> DecisionTreeClassifier:
    """Train a Decision Tree Classifier or load if path exists."""
    if DECISION_TREE_PATH.exists():
        decision_tree_model = joblib.load(DECISION_TREE_PATH)
    else:
        decision_tree_model = DecisionTreeClassifier(max_depth=5, random_state=42)
        decision_tree_model.fit(x_train, y_train)
    joblib.dump(value=decision_tree_model, filename=DECISION_TREE_PATH)

    return decision_tree_model


def train_hgboost(
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> HistGradientBoostingClassifier:
    """Train HGBoost Classifier or load if path exists."""
    if HGBOOST_PATH.exists():
        hgboost_model = joblib.load(HGBOOST_PATH)
    else:
        hgboost_model = HistGradientBoostingClassifier(random_state=42, class_weight="balanced")
        hgboost_model.fit(x_train, y_train)
        joblib.dump(value=hgboost_model, filename=HGBOOST_PATH)

    return hgboost_model


def tune_hyperparameters(
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> HistGradientBoostingClassifier:
    """Run randomized search to find the best hyperparameters."""
    if HGBOOST_OPTIMIZED_PATH.exists():
        hgboost_optimized_model = joblib.load(HGBOOST_OPTIMIZED_PATH)
    else:
        hgboost_model = HistGradientBoostingClassifier(random_state=42, class_weight="balanced")
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

        hgboost_optimized_model = HistGradientBoostingClassifier(
            **random_search.best_params_, random_state=42, class_weight="balanced"
        )
        hgboost_optimized_model.fit(x_train, y_train)
        joblib.dump(value=hgboost_optimized_model, filename=HGBOOST_OPTIMIZED_PATH)

    return hgboost_optimized_model
