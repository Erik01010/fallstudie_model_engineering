from config import CAT_FEATURES, DATA_PATH, RAW_DATA_PATH
from data_loader import load_dataset, save_dataset
from features import engineer_features
from metrics import calculate_total_cost

from models import split_features_and_target, train_decision_tree, train_ohc_encoder


def run_pipeline() -> None:
    """Main function to run the pipeline."""
    # Load data
    raw_data = load_dataset(RAW_DATA_PATH)

    # Train and save OHE encoder
    ohc = train_ohc_encoder(raw_data[CAT_FEATURES])

    # Process data
    processed_data = engineer_features(raw_data, ohc=ohc)
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
