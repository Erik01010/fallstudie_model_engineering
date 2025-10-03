import pandas as pd
import numpy as np

COSTS = {
    "Moneycard": {"success": 5, "failed": 2},
    "Goldcard": {"success": 10, "failed": 5},
    "UK_Card": {"success": 3, "failed": 1},
    "Simplecard": {"success": 1, "failed": 0.5},
}
TIME_FEATURES = {"day": 31, "dow": 7, "hour": 24}

df = pd.read_excel("data.xlsx")
df = df.drop(columns=["Unnamed: 0"])


def process_data(processed: pd.DataFrame = df) -> pd.DataFrame:
    """Drop duplicates and generate Features."""
    processed = processed.drop_duplicates()
    processed = processed.copy()

    # Informationen aus Zeitstempel extrahieren
    processed["month"] = processed.loc[:, "tmsp"].dt.month.astype("int64")
    processed["week"] = processed.loc[:, "tmsp"].dt.isocalendar().week.astype("int64")
    processed["day"] = processed.loc[:, "tmsp"].dt.day.astype("int64")
    processed["dow"] = processed.loc[:, "tmsp"].dt.dayofweek.astype("int64")
    processed["hour"] = processed.loc[:, "tmsp"].dt.hour.astype("int64")
    processed["second"] = processed.loc[:, "tmsp"].dt.second.astype("int64")
    processed["is_weekend"] = np.where(processed["dow"] >= 5, True, False)
    processed["is_business_hours"] = np.where(
        (processed["hour"] >= 8) & (processed["hour"] < 20), True, False
    )

    # Zeit-Features zyklisch kodieren
    # week und month nicht zyklisch kodieren da kein Zyklusübergang
    for key, value in TIME_FEATURES.items():
        processed[f"{key}_sin"] = np.sin(2 * np.pi * processed[key] / value)
        processed[f"{key}_cos"] = np.cos(2 * np.pi * processed[key] / value)

    # Kosten
    processed["cost"] = processed.apply(
        lambda row: COSTS[row["PSP"]]["success"]
        if row["success"]
        else COSTS[row["PSP"]]["failed"],
        axis=1,
    )

    # Wiederholte Transaktionsversuche aufgrund fehlgeschlagener Transaktionen
    processed["timedelta"] = (
        processed["tmsp"].diff().dt.total_seconds().fillna(0).astype("int64")
    )
    cols_to_compare = ["country", "amount", "3D_secured", "card"]
    processed["is_retry"] = (
        processed[cols_to_compare] == processed[cols_to_compare].shift(1)
    ).all(axis=1)

    # Anzahl kontinuierlicher Retry Versuche
    retry_groups = (~processed["is_retry"]).cumsum()
    processed["retry_count"] = (
        processed.groupby(retry_groups)["is_retry"].cumsum().astype("int64")
    )

    # Wechsel PSP bei Retry
    processed["PSP_switch"] = False
    processed["PSP_switch"] = np.where(
        (processed["is_retry"]) & (processed["PSP"] != processed["PSP"].shift(1))
        | processed["is_retry"] & (processed["PSP_switch"].shift(1)),
        True,
        False,
    )
    # Anzahl aufeinanderfolgende failed unterschiedlicher Umsätze
    return processed


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    df = process_data()
    print(df)
