import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from catboost import CatBoostClassifier
from skopt import BayesSearchCV
from skopt.space import Integer

# ----------------------------------------------------
# Create Folders
# ----------------------------------------------------
os.makedirs("./Results_CatBoost_Classifier", exist_ok=True)
os.makedirs("./Results_CatBoost_Classifier/result_txts", exist_ok=True)
os.makedirs("./Results_CatBoost_Classifier/models", exist_ok=True)

# ----------------------------------------------------
# Setup
# ----------------------------------------------------
timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
run = f"Obfus_{timestamp}"
start = datetime.now()

print("--------------START--------------")
print(f"Timestamp: {timestamp}")
print(f"Current Run: {run}")

# ----------------------------------------------------
# 1. Data Loading
# ----------------------------------------------------
#df = pd.read_csv("./DATA/obfuscation_data.csv")
#df = pd.read_csv("./DATA/train_obfuscation_data.csv")
#df = pd.read_csv("./DATA/train_obfuscation_data_functions.csv")
#df = pd.read_csv("./DATA/train_obfuscation_data_programs.csv")
df = pd.read_csv("./DATA/train_obfuscation_data_programs_cleaned_06032025.csv")



# “Obufscation” is our target, treat as categorical
y = df["Obfuscation"].astype("category")

# Features are columns 0..299 in string form
feature_cols = [str(i) for i in range(300)]
X = df[feature_cols]

# Single Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# Scale numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

best_acc = 0.0
best_params = ""
best_model = None

# ----------------------------------------------------
# Approach #1: Out-of-the-Box
# ----------------------------------------------------
print("\n[INFO] Approach #1: Out-of-the-Box")
ootb_model = CatBoostClassifier(
    task_type="GPU",
    devices="0",
    verbose=100
)
ootb_model.fit(X_train_scaled, y_train)

y_pred_ootb = ootb_model.predict(X_test_scaled)
acc_ootb = accuracy_score(y_test, y_pred_ootb)
print(f"[INFO] OOTB Test Accuracy: {acc_ootb:.4f}")

if acc_ootb > best_acc:
    best_acc = acc_ootb
    best_params = "OOTB"
    best_model = ootb_model

# ----------------------------------------------------
# Save Best Model and Scaler
# ----------------------------------------------------
if best_model:
    model_path = f"./Results_CatBoost_Classifier/models/BestModel_{run}.cbm"
    best_model.save_model(model_path)
    with open(model_path + ".pkl", "wb") as f:
        pickle.dump(best_model, f)

    scaler_path = f"./Results_CatBoost_Classifier/models/BestScaler_{run}.pkl"
    with open(scaler_path, "wb") as sf:
        pickle.dump(scaler, sf)

# ----------------------------------------------------
# Approach #2: Bayesian Optimization
# ----------------------------------------------------
print("\n[INFO] Approach #2: Bayesian Optimization")

# Define the parameter search space
bayesian_search_spaces_old = {
    "depth": Integer(4, 8),
    "iterations": Integer(500, 5000),
    "l2_leaf_reg": Integer(1, 10),
    "border_count": Integer(32, 255)
}

# Define the parameter search space
bayesian_search_spaces = {
    "depth": Integer(4, 8),
    "iterations": Integer(500, 3000),
    "l2_leaf_reg": Integer(1, 10),
    "border_count": Integer(32, 128)
}

catboost_for_search = CatBoostClassifier(
    task_type="GPU",
    devices="0",
    verbose=0  # silent while searching
)

bayes_search = BayesSearchCV(
    estimator=catboost_for_search,
    search_spaces=bayesian_search_spaces,
    n_iter=50,    # number of search iterations
    cv=3,         # cross-validation folds
    scoring="accuracy",
    random_state=42,
    verbose=100
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

# ----------------------------------------------------
# Save Best Model and Scaler
# ----------------------------------------------------
if best_model:
    model_path = f"./Results_CatBoost_Classifier/models/BestModel_{run}.cbm"
    best_model.save_model(model_path)
    with open(model_path + ".pkl", "wb") as f:
        pickle.dump(best_model, f)

    scaler_path = f"./Results_CatBoost_Classifier/models/BestScaler_{run}.pkl"
    with open(scaler_path, "wb") as sf:
        pickle.dump(scaler, sf)

# ----------------------------------------------------
# Write Out Simple Results
# ----------------------------------------------------
results_filename = f"./Results_CatBoost_Classifier/result_txts/{run}_results.txt"
end = datetime.now()
runtime = end - start

with open(results_filename, "w") as f:
    f.write(f"Run: {run}\nStart: {start}\nEnd: {end}\nTotal Runtime: {runtime}\n")
    f.write(f"Best Test Accuracy: {best_acc:.4f}\n")
    f.write(f"Best Params: {best_params}\n\n")

    f.write("OOTB Results:\n")
    f.write(f"- Test Accuracy: {acc_ootb:.4f}\n\n")

    f.write("Bayesian Results:\n")
    f.write(f"- Test Accuracy: {acc_bayes:.4f}\n")
    f.write(f"- Best Bayes Params: {bayes_search.best_params_}\n")

print("--------------END--------------")
