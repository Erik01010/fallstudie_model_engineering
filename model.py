import pandas as pd
from feature_engineering import process_data
from sklearn import tree


def calc_success_rate(df: pd.DataFrame) -> float:
    return round(df["success"].mean() * 100, 2)


def calc_avg_transaction_costs(df: pd.DataFrame) -> float:
    return round(df["cost"].mean(), 2)


class BaseLineModel:
    """Class for Baseline model."""




if __name__ == "__main__":
    df = process_data()
    print(calc_success_rate(df=df))
    print(calc_avg_transaction_costs(df=df))
