from pathlib import Path

import pandas as pd


def load_dataset(path: Path) -> pd.DataFrame:
    """Load data from a file."""
    if path.suffix == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path, index_col=0)


def save_dataset(data: pd.DataFrame, path: Path) -> None:
    """Save data to a file."""
    data.to_csv(path, index=False)
