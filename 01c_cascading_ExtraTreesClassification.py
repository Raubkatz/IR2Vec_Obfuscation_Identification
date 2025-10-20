"""
ExtraTrees training for cascaded experiments (0..5).

Usage:
- Set exp_nr = 0..5 to select the experiment.
- The script loads:
    ./Experiments_Casc/experiment_{exp_nr}/DATA/train.csv
    ./Experiments_Casc/experiment_{exp_nr}/DATA/test.csv

Targets by experiment (from cascaded builder):
    0 -> "Obfuscation"        (multiclass; all samples)
    1 -> "ObfBinary"          (binary; Ident removed)
    2 -> "SingleVsLayer"      (binary; only obfuscated, Ident & Non_Obf removed)
    3 -> "SingleMethod"       (multiclass; only single obfuscations)
    4 -> "LayeredLabel"       (multiclass; only layered obfuscations)
    5 -> "OLevel_or_NoO"      (multiclass; O-level or No_O)

Features:
    If columns "0".."299" exist, use them.
    Else, use all numeric columns except the chosen target.

Output:
    ./Results_ExtraTrees_Classifier/experiment_{exp_nr}/models/*.pkl (model, scaler)
    ./Results_ExtraTrees_Classifier/experiment_{exp_nr}/result_txts/*_results.txt
"""

# ----------------------------------------------------
# Config
# ----------------------------------------------------
exp_nr = 5                   # <-- set 0..5
random_state = 42
N_ITER = 50

# ----------------------------------------------------
# Imports
# ----------------------------------------------------
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesClassifier
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical

# ----------------------------------------------------
# Paths and experiment metadata
# ----------------------------------------------------
EXPERIMENT_ROOT = f"./Experiments_Casc/experiment_{exp_nr}/DATA"
TRAIN_PATH = os.path.join(EXPERIMENT_ROOT, "train.csv")
TEST_PATH  = os.path.join(EXPERIMENT_ROOT, "test.csv")

TARGET_BY_EXPERIMENT = {
    0: "Obfuscation",
    1: "ObfBinary",
    2: "SingleVsLayer",
    3: "SingleMethod",
    4: "LayeredLabel",
    5: "OLevel_or_NoO",
}
target_col = TARGET_BY_EXPERIMENT.get(exp_nr)
if target_col is None:
    raise ValueError(f"Unsupported exp_nr={exp_nr}. Expected 0..5.")

RESULTS_ROOT = f"./Results_ExtraTrees_Classifier/experiment_{exp_nr}"
MODELS_DIR = os.path.join(RESULTS_ROOT, "models")
TXT_DIR    = os.path.join(RESULTS_ROOT, "result_txts")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(TXT_DIR, exist_ok=True)

# ----------------------------------------------------
# Run stamp
# ----------------------------------------------------
timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
run = f"Exp{exp_nr}_{timestamp}"
start = datetime.now()

print("--------------START--------------")
print(f"Experiment: {exp_nr}")
print(f"Target: {target_col}")
print(f"Timestamp: {timestamp}")
print(f"Run ID: {run}")

# ----------------------------------------------------
# 1) Load data
# ----------------------------------------------------
if not os.path.isfile(TRAIN_PATH) or not os.path.isfile(TEST_PATH):
    raise FileNotFoundError(
        f"Train/Test not found for experiment {exp_nr}.\n"
        f"Expected:\n  {TRAIN_PATH}\n  {TEST_PATH}\n"
        f"Generate them with the cascaded experiment split script first."
    )

train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

if target_col not in train_df.columns or target_col not in test_df.columns:
    raise KeyError(f"Target column '{target_col}' missing in train/test splits.")

# ----------------------------------------------------
# 2) Select features
# ----------------------------------------------------
def infer_feature_columns(df: pd.DataFrame, target: str):
    """Prefer string '0'..'299' if present; else use numeric columns except target."""
    pref = [str(i) for i in range(300)]
    if all(col in df.columns for col in pref):
        return pref
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target in numeric_cols:
        numeric_cols.remove(target)
    if not numeric_cols:
        raise ValueError("No numeric feature columns found.")
    return numeric_cols

feature_cols = infer_feature_columns(train_df, target_col)

X_train = train_df[feature_cols].copy()
y_train = train_df[target_col].astype("category").copy()

X_test  = test_df[feature_cols].copy()
y_test  = test_df[target_col].astype("category").copy()

print(f"Features: {len(feature_cols)} columns")
print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")

# ----------------------------------------------------
# 3) Scale features
# ----------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ----------------------------------------------------
# 4) Train models (OOTB + Bayesian)
# ----------------------------------------------------
best_acc = -1.0
best_params = ""
best_model = None

# ---- Approach #1: Out-of-the-Box
print("\n[INFO] Approach #1: Out-of-the-Box")
ootb_model = ExtraTreesClassifier(random_state=random_state)
ootb_model.fit(X_train_scaled, y_train)
y_pred_ootb = ootb_model.predict(X_test_scaled)
acc_ootb = accuracy_score(y_test, y_pred_ootb)
print(f"[INFO] OOTB Test Accuracy: {acc_ootb:.4f}")

if acc_ootb > best_acc:
    best_acc = acc_ootb
    best_params = "OOTB"
    best_model = ootb_model

# Save after OOTB in case Bayesian fails
if best_model is not None:
    model_path = os.path.join(MODELS_DIR, f"BestModel_{run}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    scaler_path = os.path.join(MODELS_DIR, f"BestScaler_{run}.pkl")
    with open(scaler_path, "wb") as sf:
        pickle.dump(scaler, sf)

# ---- Approach #2: Bayesian Optimization
print("\n[INFO] Approach #2: Bayesian Optimization")

bayesian_search_spaces = {
    "n_estimators": Integer(80, 150),
    "max_depth": Integer(5, 30),
    "min_samples_split": Integer(2, 8),
    "min_samples_leaf": Integer(1, 5),
    "criterion": Categorical(["gini", "entropy"]),
    "max_features": Categorical(["log2", "sqrt", None]),
}

extra_trees_for_search = ExtraTreesClassifier(random_state=random_state)

bayes_search = BayesSearchCV(
    estimator=extra_trees_for_search,
    search_spaces=bayesian_search_spaces,
    n_iter=N_ITER,                     # keep light unless you want a deeper search
    cv=3,
    scoring="accuracy",
    random_state=random_state,
    verbose=100,
)

bayes_search.fit(X_train_scaled, y_train)
best_bayes_model = bayes_search.best_estimator_

y_pred_bayes = best_bayes_model.predict(X_test_scaled)
acc_bayes = accuracy_score(y_test, y_pred_bayes)
print(f"[INFO] Bayesian Test Accuracy: {acc_bayes:.4f}")
print(f"[INFO] Best Bayes Params: {bayes_search.best_params_}")

if acc_bayes > best_acc:
    best_acc = acc_bayes
    best_params = str(bayes_search.best_params_)
    best_model = best_bayes_model

# Save final best
if best_model is not None:
    model_path = os.path.join(MODELS_DIR, f"BestModel_{run}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    scaler_path = os.path.join(MODELS_DIR, f"BestScaler_{run}.pkl")
    with open(scaler_path, "wb") as sf:
        pickle.dump(scaler, sf)

# ----------------------------------------------------
# 5) Write results
# ----------------------------------------------------
results_filename = os.path.join(TXT_DIR, f"{run}_results.txt")
end = datetime.now()
runtime = end - start

with open(results_filename, "w") as f:
    f.write(f"Experiment: {exp_nr}\n")
    f.write(f"Target: {target_col}\n")
    f.write(f"Run: {run}\nStart: {start}\nEnd: {end}\nTotal Runtime: {runtime}\n")
    f.write(f"Features used: {len(feature_cols)}\n")
    f.write(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}\n\n")
    f.write(f"Best Test Accuracy: {best_acc:.4f}\n")
    f.write(f"Best Params: {best_params}\n\n")
    f.write("OOTB Results:\n")
    f.write(f"- Test Accuracy: {acc_ootb:.4f}\n\n")
    f.write("Bayesian Results:\n")
    f.write(f"- Test Accuracy: {acc_bayes:.4f}\n")
    f.write(f"- Best Bayes Params: {bayes_search.best_params_}\n")

print("--------------END--------------")
