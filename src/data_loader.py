import pandas as pd


def load_dataset(path: str) -> pd.DataFrame:
    """Load data from a file."""
    if path.endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_excel(path)


def save_dataset(data: pd.DataFrame, path: str) -> None:
    """Save data to a file."""
    data.to_csv(path, index=False)
