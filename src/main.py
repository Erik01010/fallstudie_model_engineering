from config import CAT_FEATURES, RAW_DATA_PATH, OHC_PATH
from features import engineer_features
from metrics import calculate_total_cost
from sklearn.model_selection import train_test_split
import pandas as pd

from models import (
    train_decision_tree,
    train_ohc_encoder,
    train_xgboost,
)  # type: ignore


def run_pipeline() -> None:
    """Main function to run the pipeline."""
    # Load data
    raw_data = pd.read_excel(RAW_DATA_PATH, index_col=0)

    # Train and save OHE encoder
    ohc = train_ohc_encoder(data=raw_data[CAT_FEATURES])

    # Process data
    processed_data = engineer_features(data=raw_data, ohc=ohc)
    processed_data.to_csv(path_or_buf="processed_data.csv", index=False)

    # Split features and target
    y = processed_data["success"]
    X = processed_data.drop(columns=["success"], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    decision_tree_model = train_decision_tree(x_train=X_train, y_train=y_train)
    xgboost_model = train_xgboost(x_train=X_train, y_train=y_train)

    # Evaluate model
    print("Evaluating Decision Tree Model:")
    calculate_total_cost(x_test=X_test, model=decision_tree_model)

    print("Evaluating XGBoost Model:")
    calculate_total_cost(x_test=X_test, model=xgboost_model)


if __name__ == "__main__":
    run_pipeline()
    print(1)