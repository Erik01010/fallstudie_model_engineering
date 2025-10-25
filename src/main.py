import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import RAW_DATA_PATH, CAT_FEATURES
from src.features import engineer_features, create_categorial_features

from src.models import (
    train_ohc_encoder,
    train_decision_tree,
    train_hgboost,
    tune_hyperparameters,
)
from src.metrics import (
    get_scores,
    plot_confusion_matrix,
    plot_multiple_precision_recall_curves,
    find_best_f1_threshold,
)


def main() -> None:
    """Main function to run the pipeline."""
    # Load and process data
    raw_data = pd.read_excel(RAW_DATA_PATH, index_col=0)
    data = raw_data.drop_duplicates()
    # Split features and target
    y = data["success"]
    X = data.drop(columns=["success"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    # add categorial features
    X_train = create_categorial_features(data=X_train)
    X_test = create_categorial_features(data=X_test)

    # train OneHotEncoder
    ohc = train_ohc_encoder(data=X_train[CAT_FEATURES])

    # Engineer features using the encoder
    X_train = engineer_features(data=X_train, encoder=ohc)
    X_test = engineer_features(data=X_test, encoder=ohc)

    # Train models
    dtm = train_decision_tree(x_train=X_train, y_train=y_train)
    hgbm = train_hgboost(x_train=X_train, y_train=y_train)
    ohgbm = tune_hyperparameters(x_train=X_train, y_train=y_train)
    models_to_evaluate = {
        "decision_tree_model": dtm,
        "hgboost_model": hgbm,
        "optimized_hgboost_model": ohgbm,
    }

    # Evaluate models - technical score
    scores = []
    for name, model in models_to_evaluate.items():
        precision, recall, accuracy, f1, roc_auc, cm = get_scores(
            name=name, model=model, y_true=y_test, x_test=X_test, threshold=0.511
        )
        scores.append(
            {
                "model": name,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "accuracy": round(accuracy, 4),
                "f1": round(f1, 4),
                "roc-auc": round(roc_auc, 4),
            }
        )
    scores = pd.DataFrame(scores)
    print(scores)

    plot_confusion_matrix(x_test=X_test, y_test=y_test, models=models_to_evaluate)
    plot_multiple_precision_recall_curves(
        x_test=X_test, y_test=y_test, models=models_to_evaluate
    )
    find_best_f1_threshold(x_test=X_test, y_test=y_test, model=ohgbm)


if __name__ == "__main__":
    main()
