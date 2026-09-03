import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score


FEATURES = [
    "new_payee",
    "amount_ratio",
    "off_hours",
    "urgency_language_flag",
    "behavioral_anomaly_flag",
    "velocity_last_hour",
]


# Load dataset
df = pd.read_csv("transactions.csv")

X = df[FEATURES]
y = df["is_fraud"]


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Create model
model = LogisticRegression(
    max_iter=1000
)


# Train model
model.fit(
    X_train,
    y_train
)


# Evaluate
accuracy = accuracy_score(
    y_test,
    model.predict(X_test)
)

auc = roc_auc_score(
    y_test,
    model.predict_proba(X_test)[:, 1]
)


print(f"Test accuracy: {accuracy:.3f}")
print(f"Test ROC-AUC:  {auc:.3f}")

print("\nModel coefficients:")

for feature, coefficient in zip(
    FEATURES,
    model.coef_[0]
):
    print(
        f"{feature:30s} "
        f"{coefficient:+.3f}"
    )

print(
    f"{'intercept':30s} "
    f"{model.intercept_[0]:+.3f}"
)


# Save model and feature list
joblib.dump(
    {
        "model": model,
        "features": FEATURES
    },
    "fraud_model.joblib"
)

print("\nSaved model to fraud_model.joblib")