from features import engineer_features
from models import train_decision_tree
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

    # Train model
    model = train_decision_tree(processed_data)

    # Evaluate model
    calculate_total_cost(model, processed_data)


if __name__ == "__main__":
    run_pipeline()
