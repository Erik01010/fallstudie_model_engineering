import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier

from src.config import DIAGRAMS_PATH

ModelType = HistGradientBoostingClassifier | DecisionTreeClassifier


def get_scores(
    name: str,
    model: ModelType,
    y_true: pd.Series,
    x_test: pd.DataFrame,
    threshold: float = 0.5,
) -> tuple[float, float, float, float, float, any]:
    """Get scores Accuracy, Precision, Recall, F1, Roc-Auc."""
    y_pred = model.predict(x_test)
    if name == "optimized_hgboost_model":
        y_pred = model.predict_proba(x_test)[:, 1]
        y_pred = (y_pred >= threshold).astype(int)
    y_proba = model.predict_proba(x_test)[:, 1]
    precision = precision_score(y_true=y_true, y_pred=y_pred)
    recall = recall_score(y_true=y_true, y_pred=y_pred)
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    f1 = f1_score(y_true=y_true, y_pred=y_pred)
    roc_auc = roc_auc_score(y_true=y_true, y_score=y_proba)
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)

    return precision, recall, accuracy, f1, roc_auc, cm


def get_feature_importance(
    models_to_evaluate: dict[str, ModelType],
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Get feature importance for all models in models_to_evaluate."""
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
    dtc = models_to_evaluate["decision_tree_model"]
    return pd.DataFrame(
        {
            "Feature": x_test.columns,
            "dtc": dtc.feature_importances_.round(4),
            "hgbm": result_hgbm.importances_mean.round(4),
            "ohgbm": result_ohgbm.importances_mean.round(4),
        }
    ).set_index("Feature")


def plot_confusion_matrix(
    x_test: pd.DataFrame,
    y_test: pd.Series,
    models: dict[str, ModelType],
) -> None:
    """Plot confusion matrix."""
    num_models = len(models)
    fig, axes = plt.subplots(1, num_models, figsize=(5 * num_models, 5))
    if num_models == 1:
        axes = [axes]
    display_labels = [0, 1]
    for i, (name, model) in enumerate(models.items()):
        preds = model.predict(x_test)
        cm = confusion_matrix(y_test, preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
        disp.plot(ax=axes[i], cmap="Blues")
        disp.ax_.set_title(name)
        disp.ax_.set_ylabel("")
        disp.ax_.set_yticklabels([])
    plt.tight_layout()
    plt.savefig(DIAGRAMS_PATH / "confusion_matrix.png")
    plt.show()


def plot_multiple_precision_recall_curves(
    x_test: pd.DataFrame,
    y_test: pd.Series,
    models: dict[str, ModelType],
) -> None:
    """Plot Precision-Recall curves for multiple models on the same plot."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, model in models.items():
        preds = model.predict_proba(x_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, preds)
        ax.plot(recall, precision, label=name, alpha=0.7)

    ax.set_title("Vergleich Precision-Recall-Kurven")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    ax.grid()

    plt.savefig(DIAGRAMS_PATH / "precision_recall_comparison.png")
    plt.show()


def find_best_f1_threshold(
    model: HistGradientBoostingClassifier,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[float, float]:
    """Find the threshold that gives the best F1 score."""
    preds = model.predict_proba(x_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, preds)

    precision = precision[:-1]
    recall = recall[:-1]
    f1_scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )
    max_f1_index = f1_scores.argmax()
    best_threshold = thresholds[max_f1_index]
    best_f1_score = f1_scores[max_f1_index]

    plt.plot(recall, precision, label="Precision-Recall Kurve")
    plt.scatter(
        recall[max_f1_index],
        precision[max_f1_index],
        marker="o",
        color="red",
        label=f"Bester F1-Score={best_f1_score:.2f}\n(Threshold={best_threshold:.2f}\n"
        f"Precision={precision[max_f1_index]:.2f}\n"
        f"Recall={recall[max_f1_index]:.2f}\n",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Kurve mit optimalem F1-Punkt")
    plt.legend()
    plt.grid()
    plt.savefig(DIAGRAMS_PATH / "precision_recall_best_f1.png")

    plt.show()

    return best_f1_score, best_threshold
