# ==========================================================
# VISUALISASI KINERJA MODEL
# ==========================================================

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix

# ----------------------------------------------------------
# 1. Generate predictions & metrics for visualization
# ----------------------------------------------------------
models = {
    "Random Forest": (rf_model, rf_t),
    "XGBoost": (xgb_model, xgb_t),
    "Logistic Regression": (lr_model, lr_t)
}

metrics = {"Model": [], "Precision": [], "Recall": [], "F1": []}

plt.figure(figsize=(7, 6))
for name, (model, thr) in models.items():
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= thr).astype(int)

    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    metrics["Model"].append(name)
    metrics["Precision"].append(precision)
    metrics["Recall"].append(recall)
    metrics["F1"].append(f1)

# Convert to DataFrame
df_metrics = pd.DataFrame(metrics)
display(df_metrics)

# ----------------------------------------------------------
# 2. BAR PLOT: Precision, Recall, F1
# ----------------------------------------------------------
df_metrics.set_index("Model").plot(kind="bar", figsize=(10,6))
plt.title("Perbandingan Precision, Recall, dan F1 Score")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.show()

# ----------------------------------------------------------
# 3. ROC CURVE
# ----------------------------------------------------------
plt.figure(figsize=(8, 6))

for name, (model, thr) in models.items():
    probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")

plt.plot([0, 1], [0, 1], "k--")
plt.title("ROC Curve - Perbandingan Model")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# ----------------------------------------------------------
# 4. CONFUSION MATRIX
# ----------------------------------------------------------
for name, (model, thr) in models.items():
    preds = (model.predict_proba(X_test)[:, 1] >= thr).astype(int)
    cm = confusion_matrix(y_test, preds)

    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# ----------------------------------------------------------
# 5. FEATURE IMPORTANCE (RF & XGB)
# ----------------------------------------------------------
# Random Forest
plt.figure(figsize=(8, 6))
importances = rf_model.named_steps["model"].feature_importances_
indices = np.argsort(importances)[-15:]  # 15 fitur paling penting
plt.barh(range(len(indices)), importances[indices])
plt.yticks(range(len(indices)), X.columns[indices])
plt.title("Top 15 Feature Importance - Random Forest")
plt.xlabel("Importance Score")
plt.show()

# XGBoost
plt.figure(figsize=(8, 6))
importances = xgb_model.named_steps["model"].feature_importances_
indices = np.argsort(importances)[-15:]
plt.barh(range(len(indices)), importances[indices])
plt.yticks(range(len(indices)), X.columns[indices])
plt.title("Top 15 Feature Importance - XGBoost")
plt.xlabel("Importance Score")
plt.show()
