import joblib
from config import MODEL_PATH, COSTS

model = joblib.load(MODEL_PATH)


def select_best_psp(transaction_features, model):
    expected_costs = {}

    features = transaction_features.copy()

    for psp in ["Moneycard", "Goldcard", "UK_Card", "Simplecard"]:
        features["PSP"] = psp
        features['cost_if_success'] = COSTS[psp]['success']
        features['cost_if_failure'] = COSTS[psp]['failure']

        prob_success = model.predict_proba(features.to_frame().T)[0, 1]

        cost_success = COSTS[psp]['success']
        cost_failure = COSTS[psp]['failure']

        expected_cost = prob_success * cost_success + (1 - prob_success) * cost_failure
        expected_costs[psp] = expected_cost

    best_psp = min(expected_costs, key=expected_costs.get)

    return best_psp, expected_costs[best_psp]

