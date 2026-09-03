import numpy as np
import pandas as pd

np.random.seed(42)

N = 8000

# 1. New payee
new_payee = np.random.binomial(1, 0.30, N)

# 2. Transaction amount
amount = np.round(
    np.random.lognormal(mean=7.5, sigma=1.3, size=N), 2
)
amount = np.clip(amount, 50, 500000)

# 3. Customer's typical transaction amount
typical_amount = np.round(
    np.random.lognormal(mean=7.3, sigma=0.9, size=N), 2
)

amount_ratio = amount / typical_amount

# 4. Hour of transaction
hour = np.random.randint(0, 24, N)

# Off-hours
off_hours = ((hour < 6) | (hour >= 22)).astype(int)

# 5. Urgency language
urgency_rate = 0.06 + 0.35 * new_payee
urgency_language = np.random.binomial(
    1,
    urgency_rate,
    N
)

# 6. Behavioral anomaly
anomaly_rate = 0.08 + 0.25 * new_payee
behavioral_anomaly = np.random.binomial(
    1,
    anomaly_rate,
    N
)

# 7. Transaction velocity
velocity = np.random.poisson(0.4, N)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Fraud probability formula
z = (
    -3.4
    + 1.9 * new_payee
    + 0.9 * np.clip(np.log1p(amount_ratio), 0, 3)
    + 1.1 * off_hours
    + 2.6 * urgency_language
    + 2.0 * behavioral_anomaly
    + 0.6 * np.clip(velocity, 0, 5)
    + np.random.normal(0, 0.6, N)
)

fraud_prob = sigmoid(z)

# Sample fraud labels
is_fraud = np.random.binomial(
    1,
    fraud_prob
)


df = pd.DataFrame({
    "new_payee": new_payee,
    "amount": amount,
    "typical_amount": typical_amount,
    "amount_ratio": np.round(amount_ratio, 3),
    "hour": hour,
    "off_hours": off_hours,
    "urgency_language_flag": urgency_language,
    "behavioral_anomaly_flag": behavioral_anomaly,
    "velocity_last_hour": velocity,
    "is_fraud": is_fraud
})

df.to_csv(
    "transactions.csv",
    index=False
)

print(f"Wrote {len(df)} rows to transactions.csv")
print(f"Fraud rate: {df['is_fraud'].mean():.2%}")

print()
print(df.head(8).to_string(index=False))