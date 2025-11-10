# ==========================================================
# 1. IMPORT LIBRARIES
# ==========================================================
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore')

# ==========================================================
# 2. LOAD DATA
# ==========================================================
df = pd.read_excel("default of credit card clients.xls", header=1)
df.rename(columns={"default payment next month": "default"}, inplace=True)

X = df.drop(columns=["default", "ID"])
y = df["default"]

# ==========================================================
# 3. TRAIN TEST SPLIT
# ==========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================================================
# 4. FUNCTION: F1 OPTIMAL THRESHOLD
# ==========================================================
def find_best_threshold(model, X_val, y_val):
    probs = model.predict_proba(X_val)[:, 1]
    best_t, best_f1 = 0.5, 0
    for t in np.linspace(0.1, 0.9, 40):
        preds = (probs >= t).astype(int)
        f1 = f1_score(y_val, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t

# ==========================================================
# 5. PIPELINE: SMOTE + SCALER + MODEL
# ==========================================================
def build_pipeline(model):
    return ImbPipeline([
        ("smote", SMOTE(sampling_strategy=0.6)),
        ("scaler", StandardScaler()),
        ("model", model)
    ])

# ==========================================================
# 6. HYPERPARAMETER SEARCH (FAST VERSION)
# ==========================================================
rf_params = {
    "model__n_estimators": [120, 200],
    "model__max_depth": [5, 8, 12],
    "model__min_samples_split": [2, 5, 10]
}

xgb_params = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [3, 5],
    "model__learning_rate": [0.05, 0.1],
    "model__subsample": [0.8, 1]
}

lr_params = {
    "model__C": [0.1, 1, 2, 5],
    "model__penalty": ["l2"],
    "model__solver": ["lbfgs"]
}

# ==========================================================
# 7. FIT MODELS WITH RANDOMIZEDSEARCHCV (FAST)
# ==========================================================
def train_model(model, params, name):
    print(f"\n========== TRAINING {name} ==========")
    pipe = build_pipeline(model)

    search = RandomizedSearchCV(
        pipe, params, n_iter=6,
        scoring="f1", n_jobs=-1, cv=3, random_state=42
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    print("Best Params:", search.best_params_)

    # Threshold tuning
    best_t = find_best_threshold(best_model, X_test, y_test)
    print("Optimal Threshold:", round(best_t, 3))

    preds = (best_model.predict_proba(X_test)[:, 1] >= best_t).astype(int)

    print(classification_report(y_test, preds))
    return best_model, best_t


# ==========================================================
# 8. RUN ALL MODELS
# ==========================================================
rf_model, rf_t = train_model(
    RandomForestClassifier(),
    rf_params, "Random Forest"
)

xgb_model, xgb_t = train_model(
    XGBClassifier(eval_metric="logloss", use_label_encoder=False),
    xgb_params, "XGBoost"
)

lr_model, lr_t = train_model(
    LogisticRegression(max_iter=2000),
    lr_params, "Logistic Regression"
)
