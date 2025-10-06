from features import engineer_features
from models import train_decision_tree, split_features_and_target
from metrics import calculate_total_cost
from data_loader import load_dataset, save_dataset
from config import DATA_PATH, RAW_DATA_PATH


def run_pipeline() -> None:
    """Main function to run the pipeline."""
    # Load data
    raw_data = load_dataset(RAW_DATA_PATH)

    # Process data
    processed_data = engineer_features(raw_data)
    save_dataset(processed_data, DATA_PATH)

    # Split features and target
    X_train, X_test, y_train, y_test = split_features_and_target(processed_data)

    # Train model
    model = train_decision_tree(
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
    )

    # Evaluate model
    calculate_total_cost(model, X_test=X_test, y_test=y_test)


if __name__ == "__main__":
    run_pipeline()
