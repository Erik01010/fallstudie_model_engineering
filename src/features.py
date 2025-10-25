import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from src.config import CAT_FEATURES, CYCLICAL_FEATURES, PSP_COSTS


def create_categorial_features(data: pd.DataFrame) -> pd.DataFrame:
    """Creates all raw categorical features, including interactions."""
    data = data.copy()

    # Temporären Amount-Bin erstellen
    data["amount_bins"] = pd.cut(
        data["amount"],
        bins=[0, 200, 400, float("inf")],
        labels=["amount_under_200", "amount_200_400", "amount_over_400"],
        right=False,
    )

    # Interaktions-Features erstellen
    data["interaction_psp_country"] = data["PSP"] + "_" + data["country"]
    data["interaction_psp_card"] = data["PSP"] + "_" + data["card"]
    data["interaction_psp_amount_bin"] = (
        data["PSP"] + "_" + data["amount_bins"].astype(str)
    )
    data["interaction_psp_3D_secured"] = (
        data["PSP"] + "_" + data["3D_secured"].astype(str)
    )

    return data


def engineer_features(data: pd.DataFrame, encoder: OneHotEncoder) -> pd.DataFrame:
    """Generate Features."""
    data = data.copy()

    # Informationen aus Zeitstempel extrahieren
    data["month"] = data.loc[:, "tmsp"].dt.month.astype("int64")
    data["week"] = data.loc[:, "tmsp"].dt.isocalendar().week.astype("int64")
    data["day"] = data.loc[:, "tmsp"].dt.day.astype("int64")
    data["dow"] = data.loc[:, "tmsp"].dt.dayofweek.astype("int64")
    data["hour"] = data.loc[:, "tmsp"].dt.hour.astype("int64")
    data["second"] = data.loc[:, "tmsp"].dt.second.astype("int64")
    data["is_weekend"] = data["dow"] >= 5
    data["is_business_hours"] = (data["hour"] >= 8) & (data["hour"] < 20)

    # Zeit-Features zyklisch kodieren
    # week und month nicht zyklisch kodieren da kein Zyklusübergang
    for key, value in CYCLICAL_FEATURES.items():
        data[f"{key}_sin"] = np.sin(2 * np.pi * data[key] / value)
        data[f"{key}_cos"] = np.cos(2 * np.pi * data[key] / value)

    # Kosten
    data["cost_if_success"] = data["PSP"].map(lambda psp: PSP_COSTS[psp]["success"])
    data["cost_if_failure"] = data["PSP"].map(lambda psp: PSP_COSTS[psp]["failure"])

    # Wiederholte Transaktionsversuche aufgrund fehlgeschlagener Transaktionen
    data["timedelta"] = data["tmsp"].diff().dt.total_seconds().fillna(0).astype("int64")
    cols_to_compare = ["country", "amount", "3D_secured", "card"]
    data["is_retry"] = (data[cols_to_compare] == data[cols_to_compare].shift(1)).all(
        axis=1
    )
    data["is_retry"] = data["is_retry"] & (data["timedelta"] <= 60)

    # Anzahl kontinuierlicher Retry Versuche
    retry_groups = (~data["is_retry"]).cumsum()
    data["retry_count"] = (
        data.groupby(retry_groups)["is_retry"].cumsum().astype("int64")
    )

    # Wechsel PSP bei Retry
    data["PSP_switch"] = data.groupby(retry_groups)["PSP"].transform(
        lambda x: (x != x.shift()).fillna(False).cumsum() > 0
    )

    # kategorische Merkmale encodieren
    encoded_array = encoder.transform(data[CAT_FEATURES])
    encoded_columns = encoder.get_feature_names_out(CAT_FEATURES)
    encoded_df = pd.DataFrame(encoded_array, columns=encoded_columns, index=data.index)
    data = pd.concat([data, encoded_df], axis=1)

    # Timestamp und nicht kategorische features entfernen
    cat_features = CAT_FEATURES + ["tmsp"]
    data = data.drop(columns=cat_features, axis=1)

    return data
