import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import CAT_FEATURES
from src.config import RAW_DATA_PATH
from src.features import create_categorial_features
from src.features import engineer_features
from src.metrics import get_scores
from src.models import train_decision_tree
from src.models import train_hgboost
from src.models import train_ohc_encoder
from src.models import tune_hyperparameters
from src.predictions import evaluate_business_strategies


def main() -> None:
    """Run the pipeline."""
    # Load and process data
    raw_data = pd.read_excel(RAW_DATA_PATH, index_col=0)
    processed_data = raw_data.drop_duplicates()
    # Split features and target
    y = processed_data["success"]
    X = processed_data.drop(columns=["success"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # add categorial features
    X_train = create_categorial_features(data=X_train)
    X_test = create_categorial_features(data=X_test)

    # train OneHotEncoder
    ohc = train_ohc_encoder(data=X_train[CAT_FEATURES])

    # Engineer features using the encoder
    X_train = engineer_features(data=X_train, encoder=ohc)
    X_test = engineer_features(data=X_test, encoder=ohc)

    # Train models
    decision_tree_model = train_decision_tree(x_train=X_train, y_train=y_train)
    hgboost_model = train_hgboost(x_train=X_train, y_train=y_train)
    hgboost_optimized_model = tune_hyperparameters(x_train=X_train, y_train=y_train)

    models_to_evaluate = {
        "decision_tree_model": decision_tree_model,
        "hgboost_model": hgboost_model,
        "optimized_hgboost_model": hgboost_optimized_model,
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

    results_df = evaluate_business_strategies(
        model=hgboost_optimized_model,
        x_test=X_test, y_test=y_test,
        original_data=processed_data,
        encoder=ohc
    )
    print("--- Strategy evaluation ---")
    column_order = [
        "Actual Success Rate",
        "Avg. Expected Success Rate",
        "Total Cost",
    ]
    print(
        results_df[column_order].to_string(
            formatters={
                "Actual Success Rate": "{:.2f}".format,
                "Avg. Expected Success Rate": "{:.2f}".format,
                "Total Cost": "{:,.2f}".format,
            }
        )
    )

    # plot_confusion_matrix(x_test=X_test, y_test=y_test, models=models_to_evaluate)
    # plot_multiple_precision_recall_curves(x_test=X_test, y_test=y_test, models=models_to_evaluate)
    # find_best_f1_threshold(x_test=X_test, y_test=y_test, model=hgboost_optimized_model)


if __name__ == "__main__":
    main()
