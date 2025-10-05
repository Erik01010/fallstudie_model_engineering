import pandas as pd
from sklearn.metrics import accuracy_score

from sklearn import tree
from sklearn.model_selection import train_test_split
import joblib
from config import DATA_PATH, MODEL_PATH
from evaluation import select_best_psp


df = pd.read_csv(filepath_or_buffer=DATA_PATH)
y = df["success"]
X = df.drop(columns=["success"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

baseline_model = tree.DecisionTreeClassifier(random_state=42)
baseline_model.fit(X_train, y_train)
y_pred_test = baseline_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred_test)
joblib.dump(baseline_model, MODEL_PATH)

print(f"Model accuracy: {accuracy:.4f}")

sample_transaction = X_test.iloc[0]
best_psp, min_cost = select_best_psp(sample_transaction, baseline_model)
print(f"Best PSP for the sample transaction: {best_psp} with expected cost:{min_cost:.2f}")