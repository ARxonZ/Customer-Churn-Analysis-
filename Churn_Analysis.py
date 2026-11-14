"""Customer Churn Analysis: EDA, key churn factors, model, retention ideas."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

df = pd.read_csv("data/customers.csv")
out = []
def log(s=""):
    print(s); out.append(str(s))

# ---------- 1. Overview ----------
log("=== 1. DATASET OVERVIEW ===")
log(f"Rows: {len(df)} | Columns: {df.shape[1]}")
log(f"Missing values: {int(df.isna().sum().sum())}")
log(f"Overall churn rate: {df.churned.mean():.1%}")

# ---------- 2. Churn by segments ----------
log("\n=== 2. CHURN BY SEGMENT ===")
df["tenure_band"] = pd.cut(df.tenure_months, [0, 6, 12, 24, 60],
                           labels=["0-6m", "7-12m", "13-24m", "25-60m"])
df["login_band"] = pd.cut(df.logins_per_month, [-1, 5, 10, 20, 100],
                          labels=["0-5", "6-10", "11-20", "20+"])
df["inactive_14d"] = np.where(df.days_since_last_login >= 14, "14+ days", "<14 days")
df["errors_band"] = df.feature_errors.clip(upper=3).astype(str).replace({"3": "3+"})
df["support_band"] = df.support_contacts.clip(upper=4).astype(str).replace({"4": "4+"})
segs = ["tenure_band", "login_band", "inactive_14d", "errors_band", "support_band", "plan"]
seg_tables = {}
for s in segs:
    t = df.groupby(s, observed=True).churned.agg(["mean", "count"])
    t["mean"] = (t["mean"] * 100).round(1)
    seg_tables[s] = t
    log(f"\nChurn % by {s}:\n{t.rename(columns={'mean':'churn_%','count':'customers'}).to_string()}")

# ---------- 3. Model & key factors ----------
log("\n=== 3. MODEL & KEY CHURN FACTORS ===")
feats = ["tenure_months", "monthly_fee", "logins_per_month",
         "days_since_last_login", "support_contacts", "feature_errors"]
X, y = df[feats], df.churned
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

rf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
rf.fit(Xtr, ytr)
pred, proba = rf.predict(Xte), rf.predict_proba(Xte)[:, 1]
log(f"Random Forest accuracy: {accuracy_score(yte, pred):.3f} | ROC-AUC: {roc_auc_score(yte, proba):.3f}")
log("\nClassification report:\n" + classification_report(yte, pred))
cm = confusion_matrix(yte, pred)

lr = LogisticRegression(max_iter=1000).fit((Xtr - Xtr.mean()) / Xtr.std(), ytr)
imp = pd.DataFrame({"feature": feats,
                    "rf_importance": rf.feature_importances_,
                    "logreg_coef_std": lr.coef_[0]}).sort_values("rf_importance", ascending=False)
imp.to_csv("outputs/feature_importance.csv", index=False)
log("Feature importance:\n" + imp.round(3).to_string(index=False))

# ---------- 4. Charts (proof) ----------
plt.style.use("seaborn-v0_8-whitegrid")

fig, ax = plt.subplots(figsize=(6, 4))
ax.pie(df.churned.value_counts().sort_index(), labels=["Active", "Churned"],
       autopct="%1.1f%%", colors=["#4CAF50", "#E53935"], startangle=90)
ax.set_title("Overall Churn Rate"); fig.savefig("outputs/01_churn_rate.png", dpi=150, bbox_inches="tight"); plt.close()

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, s in zip(axes.ravel(), segs):
    t = seg_tables[s]
    ax.bar(t.index.astype(str), t["mean"], color="#1E88E5")
    ax.set_title(f"Churn % by {s}"); ax.set_ylabel("Churn %")
    for i, v in enumerate(t["mean"]): ax.text(i, v + 0.5, f"{v}%", ha="center", fontsize=9)
fig.tight_layout(); fig.savefig("outputs/02_churn_by_segment.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(7, 4))
imp.sort_values("rf_importance").plot.barh(x="feature", y="rf_importance", ax=ax, color="#FB8C00", legend=False)
ax.set_title("Key Churn Factors (Random Forest importance)")
fig.savefig("outputs/03_feature_importance.png", dpi=150, bbox_inches="tight"); plt.close()

fig, ax = plt.subplots(figsize=(4.5, 4))
ax.imshow(cm, cmap="Blues")
for (i, j), v in np.ndenumerate(cm): ax.text(j, i, v, ha="center", va="center", fontsize=14)
ax.set_xticks([0, 1], ["Active", "Churned"]); ax.set_yticks([0, 1], ["Active", "Churned"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion Matrix")
fig.savefig("outputs/04_confusion_matrix.png", dpi=150, bbox_inches="tight"); plt.close()

# ---------- 5. Insights ----------
inact = seg_tables["inactive_14d"]["mean"]
err = seg_tables["errors_band"]["mean"]
log("\n=== 4. KEY INSIGHTS ===")
log(f"- Customers inactive 14+ days churn at {inact['14+ days']}% vs {inact['<14 days']}% for active ones.")
log(f"- Customers hitting 3+ feature errors churn at {err['3+']}% vs {err['0']}% with none.")
log(f"- New customers (0-6m) churn at {seg_tables['tenure_band'].loc['0-6m','mean']}% vs {seg_tables['tenure_band'].loc['25-60m','mean']}% for 25m+.")
log(f"- Top predictors: {', '.join(imp.feature.head(3))}.")
log("\n=== 5. RETENTION RECOMMENDATIONS ===")
for r in ["Trigger a win-back email/notification when a user has no login for 7 days (before the 14-day cliff).",
          "Prioritise bug fixes for features with repeated errors; auto-alert support after 2 failures.",
          "Run a 90-day onboarding programme to reduce early-tenure churn.",
          "Proactively reach out to users with 3+ support contacts; resolve root cause.",
          "Offer discounts/upgrade trials to Basic-plan users flagged as high-risk by the model."]:
    log(f"* {r}")

open("outputs/analysis_summary.txt", "w").write("\n".join(out))
print("\nSaved outputs/ (charts, feature_importance.csv, analysis_summary.txt)")
