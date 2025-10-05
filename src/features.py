import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from config import CYCLICAL_FEATURES, PSP_COSTS
import joblib


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates and generate Features."""
    data = data.drop_duplicates()
    data = data.copy()

    # Informationen aus Zeitstempel extrahieren
    data["month"] = data.loc[:, "tmsp"].dt.month.astype("int64")
    data["week"] = data.loc[:, "tmsp"].dt.isocalendar().week.astype("int64")
    data["day"] = data.loc[:, "tmsp"].dt.day.astype("int64")
    data["dow"] = data.loc[:, "tmsp"].dt.dayofweek.astype("int64")
    data["hour"] = data.loc[:, "tmsp"].dt.hour.astype("int64")
    data["second"] = data.loc[:, "tmsp"].dt.second.astype("int64")
    data["is_weekend"] = np.where(data["dow"] >= 5, True, False)
    data["is_business_hours"] = np.where(
        (data["hour"] >= 8) & (data["hour"] < 20), True, False
    )

    # Zeit-Features zyklisch kodieren
    # week und month nicht zyklisch kodieren da kein Zyklusübergang
    for key, value in CYCLICAL_FEATURES.items():
        data[f"{key}_sin"] = np.sin(2 * np.pi * data[key] / value)
        data[f"{key}_cos"] = np.cos(2 * np.pi * data[key] / value)

    # Kosten
    data["cost_if_success"] = data["PSP"].map(
        lambda psp: PSP_COSTS[psp]["success"]
    )
    data["cost_if_failure"] = data["PSP"].map(
        lambda psp: PSP_COSTS[psp]["failure"]
    )

    # Wiederholte Transaktionsversuche aufgrund fehlgeschlagener Transaktionen
    data["timedelta"] = (
        data["tmsp"].diff().dt.total_seconds().fillna(0).astype("int64")
    )
    cols_to_compare = ["country", "amount", "3D_secured", "card"]
    data["is_retry"] = (
        data[cols_to_compare] == data[cols_to_compare].shift(1)
    ).all(axis=1)

    # Anzahl kontinuierlicher Retry Versuche
    retry_groups = (~data["is_retry"]).cumsum()
    data["retry_count"] = (
        data.groupby(retry_groups)["is_retry"].cumsum().astype("int64")
    )

    # Wechsel PSP bei Retry
    data["PSP_switch"] = False
    data["PSP_switch"] = np.where(
        (data["is_retry"]) & (data["PSP"] != data["PSP"].shift(1))
        | data["is_retry"] & (data["PSP_switch"].shift(1)),
        True,
        False,
    )
    # Anzahl aufeinanderfolgende failed unterschiedlicher Umsätze

    # kategorische Merkmale encodieren
    cat_features = data[["country", "card", "PSP"]]
    one_hot_encoder = OneHotEncoder(sparse_output=False)
    encoded_array = one_hot_encoder.fit_transform(cat_features)
    joblib.dump(one_hot_encoder, "models/one_hot_encoder.joblib")
    encoded_columns = one_hot_encoder.get_feature_names_out(cat_features.columns)
    encoded_df = pd.DataFrame(
        encoded_array, columns=encoded_columns, index=data.index
    )
    data = pd.concat([data, encoded_df], axis=1)

    # Timestamp und nicht kategorische features entfernen
    data = data.drop(columns=["tmsp", "PSP"])

    return data
