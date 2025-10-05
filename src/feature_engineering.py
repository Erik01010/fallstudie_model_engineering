import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from config import TIME_FEATURES, COSTS




def process_data(processed: pd.DataFrame) -> pd.DataFrame:
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
    processed['cost_if_success'] = processed['PSP'].map(lambda psp: COSTS[psp]['success'])
    processed['cost_if_failure'] = processed['PSP'].map(lambda psp: COSTS[psp]['failure'])

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

    # kategorische Merkmale encodieren
    cat_features = processed[["country", "card", "PSP"]]
    one_hot_encoder = OneHotEncoder(sparse_output=False)
    encoded_array = one_hot_encoder.fit_transform(cat_features)
    encoded_columns = one_hot_encoder.get_feature_names_out(cat_features.columns)
    encoded_df = pd.DataFrame(
        encoded_array, columns=encoded_columns, index=processed.index
    )
    processed = pd.concat([processed, encoded_df], axis=1)

    # Timestamp und nicht kategorische features entfernen
    processed = processed.drop(columns=["tmsp", "country", "card", "PSP"])

    return processed


if __name__ == "__main__":
    """Main function to process data and save to excel."""
    raw = pd.read_excel("../data/data.xlsx")
    raw = raw.drop(columns=["Unnamed: 0"])

    data = process_data(raw)
    data.to_csv("../data/processed_data.csv", index=False)
