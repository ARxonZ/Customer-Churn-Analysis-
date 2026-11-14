"""Generate a synthetic subscription-customer dataset (5,000 rows)."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 5000

tenure = rng.integers(1, 61, n)                         # months subscribed
logins = np.clip(rng.normal(14, 7, n), 0, None).round() # logins per month
days_since_login = np.clip(rng.exponential(8, n), 0, 60).round()
support = rng.poisson(1.5, n)                           # support contacts
feature_errors = rng.poisson(1.0, n)                    # "specific feature breaks"
plan = rng.choice(["Basic", "Standard", "Premium"], n, p=[0.45, 0.35, 0.20])
monthly_fee = pd.Series(plan).map({"Basic": 9, "Standard": 19, "Premium": 39}).values

# churn probability driven by the same signals shown in the task brief
z = (-1.2
     - 0.04 * tenure
     - 0.06 * logins
     + 0.07 * days_since_login
     + 0.30 * support
     + 0.45 * feature_errors
     + (days_since_login >= 14) * 1.2
     + (plan == "Basic") * 0.4)
prob = 1 / (1 + np.exp(-z))
churn = (rng.random(n) < prob).astype(int)

df = pd.DataFrame({
    "customer_id": [f"C{i:05d}" for i in range(1, n + 1)],
    "tenure_months": tenure,
    "plan": plan,
    "monthly_fee": monthly_fee,
    "logins_per_month": logins.astype(int),
    "days_since_last_login": days_since_login.astype(int),
    "support_contacts": support,
    "feature_errors": feature_errors,
    "churned": churn,
})
df.to_csv("data/customers.csv", index=False)
print(f"Saved data/customers.csv  rows={len(df)}  churn_rate={df.churned.mean():.1%}")
