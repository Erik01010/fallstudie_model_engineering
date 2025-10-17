import pandas as pd
import joblib
from config import PARAM_DIST, FINAL_MODEL_PATH
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold


def find_best_hgb_model(
    x_train: pd.DataFrame, y_train: pd.DataFrame
) -> HistGradientBoostingClassifier:
    """Run randomized search to find the best hyperparameters."""
    hgboost_model = HistGradientBoostingClassifier(random_state=42)
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    random_search = RandomizedSearchCV(
        estimator=hgboost_model,
        param_distributions=PARAM_DIST,
        n_iter=1,
        cv=cv_strategy,
        scoring="roc_auc",
        n_jobs=-1,
        random_state=42,
        verbose=0,
    )

    random_search.fit(x_train, y_train)

    print(f"Beste Parameter gefunden: {random_search.best_params_}")
    print(f"Bester CV AUC-Score: {random_search.best_score_:.4f}")

    final_model = HistGradientBoostingClassifier(
        **random_search.best_params_, random_state=42
    )
    final_model.fit(x_train, y_train)
    joblib.dump(final_model, FINAL_MODEL_PATH)

    return final_model
