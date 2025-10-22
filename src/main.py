import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import RAW_DATA_PATH
from src.features import engineer_features
from src.predictions import evaluate_business_impact, evaluate_technical_performance

from src.models import (
    train_decision_tree,
    train_hgboost,
    tune_hyperparameters,
)
from src.metrics import (
    get_scores,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_multiple_precision_recall_curves,
)


def run_pipeline() -> None:
    """Main function to run the pipeline."""
    raw_data = pd.read_excel(RAW_DATA_PATH, index_col=0)
    processed_data, ohc = engineer_features(data=raw_data)
    processed_data.to_csv(
        path_or_buf="processed_data.csv", index=False, encoding="utf-8"
    )

    # Split features and target
    y = processed_data["success"]
    X = processed_data.drop(columns=["success"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train models
    models_to_evaluate = {
        "decision_tree_model": train_decision_tree(x_train=X_train, y_train=y_train),
        "hgboost_model": train_hgboost(x_train=X_train, y_train=y_train),
        "optimized_hgboost_model": tune_hyperparameters(
            x_train=X_train, y_train=y_train
        ),
    }

    # Evaluate models - technical score
    print("\n--- Model Evaluation ---")
    for name, model in models_to_evaluate.items():
        print(f"\nEvaluating {name}")
        evaluate_technical_performance(model=model, x_test=X_test, y_test=y_test)
        evaluate_business_impact(
            model=model, x_test=X_test, y_test=y_test, original_data=raw_data
        )
    print("\n--- Evaluation complete ---")


def main():
    """Main function to run the pipeline."""
    # Load and process data
    raw_data = pd.read_excel(RAW_DATA_PATH, index_col=0)
    processed_data, ohc = engineer_features(data=raw_data)

    # Split features and target
    y = processed_data["success"]
    X = processed_data.drop(columns=["success"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

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
    print("\n--- Model Evaluation Scores ---")
    for name, model in models_to_evaluate.items():
        precision, recall, roc_auc, cm = get_scores(
            name=name, model=model, y_true=y_test, x_test=X_test
        )
        print("Model Name:   ", name)
        print("Precision:    ", round(precision, 4))
        print("Recall:       ", round(recall, 4))
        print("ROC-AUC:      ", round(roc_auc, 4))
        print("\n")

    plot_confusion_matrix(x_test=X_test, y_test=y_test, model=ohgbm)
    plot_multiple_precision_recall_curves(x_test=X_test, y_test=y_test, models=models_to_evaluate)


if __name__ == "__main__":
    main()
    # run_pipeline()
