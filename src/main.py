from config import (
    CAT_FEATURES,
    RAW_DATA_PATH,
    DECISION_TREE_PATH,
    HGBOOST_PATH,
    FINAL_MODEL_PATH,
    DATA_PATH,
)
from features import engineer_features
from predictions import evaluate_business_impact, evaluate_technical_performance
from tuning import find_best_hgb_model
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

from models import (
    train_decision_tree,
    train_ohc_encoder,
    train_hgboost,
)  # type: ignore


def run_pipeline() -> None:
    """Main function to run the pipeline."""
    raw_data = pd.read_excel(RAW_DATA_PATH, index_col=0)
    ohc = train_ohc_encoder(data=raw_data[CAT_FEATURES])
    processed_data = engineer_features(data=raw_data, ohc=ohc)
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
        "optimized_hgboost_model": find_best_hgb_model(
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


def run_pre_loaded_model_evaluation():
    raw_data = pd.read_excel(RAW_DATA_PATH, index_col=0)
    processed_data = pd.read_csv(DATA_PATH, encoding="utf-8")

    # Split features and target
    y = processed_data["success"]
    X = processed_data.drop(columns=["success"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pretrained_models_to_evaluate = {
        "decision_tree_model": DECISION_TREE_PATH,
        "hgboost_model": HGBOOST_PATH,
        "optimized_hgboost_model": FINAL_MODEL_PATH,
    }

    print("\n--- Model Evaluation ---")
    for name, model_path in pretrained_models_to_evaluate.items():
        print(f"\nEvaluating {name}")
        print(model_path)

        with open(model_path, "rb") as f:
            model = joblib.load(f)
            evaluate_technical_performance(model=model, x_test=X_test, y_test=y_test)
            evaluate_business_impact(
                model=model, x_test=X_test, y_test=y_test, original_data=raw_data
            )
    print("\n--- Evaluation complete ---")


if __name__ == "__main__":
    # run_pre_loaded_model_evaluation()
    run_pipeline()
