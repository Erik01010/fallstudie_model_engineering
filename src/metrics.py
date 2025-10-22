import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    PrecisionRecallDisplay,
)
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt


def calculate_success_probability(model, features: pd.DataFrame) -> float:
    """Calculates the success probability for a given model and features."""
    return model.predict_proba(features)[:, 1]


def get_scores(
    name: str,
    model: DecisionTreeClassifier | HistGradientBoostingClassifier,
    y_true: pd.Series,
    x_test: pd.DataFrame,
):
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]
    precision = precision_score(y_true=y_true, y_pred=y_pred)
    recall = recall_score(y_true=y_true, y_pred=y_pred)
    roc_auc = roc_auc_score(y_true=y_true, y_score=y_proba)
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)

    return precision, recall, roc_auc, cm


def get_feature_importance(
    models_to_evaluate: dict[
        str, DecisionTreeClassifier | HistGradientBoostingClassifier
    ],
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    # Feature Importance
    result_hgbm = permutation_importance(
        models_to_evaluate["hgboost_model"],
        x_test,
        y_test,
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
    )
    result_ohgbm = permutation_importance(
        models_to_evaluate["optimized_hgboost_model"],
        x_test,
        y_test,
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
    )
    dtm = models_to_evaluate["decision_tree_model"]
    perm_importance_df = pd.DataFrame(
        {
            "Feature": x_test.columns,
            "dtm": dtm.feature_importances_.round(4),
            "hgbm": result_hgbm.importances_mean.round(4),
            "ohgbm": result_ohgbm.importances_mean.round(4),
        }
    ).set_index("Feature")

    return perm_importance_df


def plot_confusion_matrix(
    x_test: pd.DataFrame,
    y_test: pd.Series,
    model: HistGradientBoostingClassifier | DecisionTreeClassifier,
) -> None:
    preds = model.predict(x_test)
    cm = confusion_matrix(y_test, preds)
    labels = [0, 1]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    plt.savefig("confusion_matrix.png")


def plot_precision_recall_curve(
    x_test: pd.DataFrame,
    y_test: pd.Series,
    model: HistGradientBoostingClassifier | DecisionTreeClassifier,
) -> None:
    preds = model.predict_proba(x_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, preds)
    disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    disp.plot()
    plt.savefig("precision_recall_curve.png")


def plot_multiple_precision_recall_curves(
    x_test: pd.DataFrame,
    y_test: pd.Series,
    models: dict[str, HistGradientBoostingClassifier | DecisionTreeClassifier],
) -> None:
    """Plot Precision-Recall curves for multiple models on the same plot."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, model in models.items():
        preds = model.predict_proba(x_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, preds)
        ax.plot(recall, precision, label=name, alpha=0.5)

    ax.set_title("Vergleich Precision-Recall-Kurven")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    ax.grid(True)

    plt.savefig("precision_recall_comparison.png")
    plt.show()