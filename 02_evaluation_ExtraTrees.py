import os
import pickle
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# -------------------------------------
# User-Defined Filenames/Paths
# -------------------------------------
MODEL_NAME = "BestModel_Obfus_20250306160903.pkl"  # Example: rename as needed
SCALER_NAME = "BestScaler_Obfus_20250306160903.pkl"  # Example: rename as needed
TEST_DATA_PATH = "./DATA/test_obfuscation_data_programs_cleaned_06032025.csv"  # Path to your test data

# Columns
TARGET_COLUMN = "Obfuscation"  # Same target column used during training
FEATURE_COLS = [str(i) for i in range(300)]  # Same feature columns (0..299)

# Class chunking size (for sub-confusion matrices)
CLASS_CHUNK_SIZE = 7

# Output directories for results
CONF_MATRIX_DIR = "./Results_ExtraTrees_Classifier/confusion_matrices"
RESULT_TXT_DIR = "./Results_ExtraTrees_Classifier/result_txts"
os.makedirs(CONF_MATRIX_DIR, exist_ok=True)
os.makedirs(RESULT_TXT_DIR, exist_ok=True)

# -------------------------------------
# Custom Plotting Palette
# -------------------------------------
custom_palette = ["#188FA7", "#769FB6", "#9DBBAE", "#D5D6AA", "#E2DBBE"]

# -------------------------------------
# 1. Load Model and Scaler
# -------------------------------------
model_path = f"./Results_ExtraTrees_Classifier/models/{MODEL_NAME}"
scaler_path = f"./Results_ExtraTrees_Classifier/models/{SCALER_NAME}"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file '{MODEL_NAME}' not found at {model_path}")
if not os.path.exists(scaler_path):
    raise FileNotFoundError(f"Scaler file '{SCALER_NAME}' not found at {scaler_path}")

print(f"[INFO] Loading model: {MODEL_NAME}")
with open(model_path, "rb") as f:
    model = pickle.load(f)

print(f"[INFO] Loading scaler: {SCALER_NAME}")
with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

# Ensure we have an ExtraTreesClassifier model
if not isinstance(model, ExtraTreesClassifier):
    raise TypeError("Loaded model is not an ExtraTreesClassifier.")

# -------------------------------------
# 2. Load and Preprocess Test Data
# -------------------------------------
print(f"[INFO] Reading test data from: {TEST_DATA_PATH}")
test_data = pd.read_csv(TEST_DATA_PATH)

X_test = test_data[FEATURE_COLS].copy()
y_test = test_data[TARGET_COLUMN].astype("category")

# Scale numeric features with the previously fitted scaler
X_test_scaled = scaler.transform(X_test)

# -------------------------------------
# 3. Single-Run Prediction (All Classes)
# -------------------------------------
y_pred = model.predict(X_test_scaled)

print("\n[INFO] Single-Run Classification Report (All Classes):")
print(classification_report(y_test, y_pred))

conf_matrix_single = confusion_matrix(y_test, y_pred)
print("[INFO] Single-Run Confusion Matrix (All Classes):")
print(conf_matrix_single)

# Feature Importances
feature_importances = model.feature_importances_
importance_tuples = list(zip(FEATURE_COLS, feature_importances))
importance_sorted = sorted(importance_tuples, key=lambda x: x[1], reverse=True)

print("\n[INFO] Feature Importances (sorted by importance):")
for feature, importance in importance_sorted:
    print(f"{feature}: {importance:.4f}")

# -------------------------------------
# 4. Plot & Save Full Confusion Matrix (Counts)
# -------------------------------------
full_classes = sorted(y_test.unique())

plt.figure(figsize=(40, 30), dpi=300)
ax = sns.heatmap(
    conf_matrix_single,
    annot=True,
    fmt="d",
    cmap=custom_palette,
    xticklabels=full_classes,
    yticklabels=full_classes,
    annot_kws={"fontsize": 12},
    linewidths=1,
    linecolor="black"
)
ax.set_title("Single-Run Confusion Matrix (All Classes)", fontsize=28)
ax.set_xlabel("Predicted", fontsize=24)
ax.set_ylabel("True", fontsize=24)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
plt.tight_layout()

conf_matrix_single_png = os.path.join(
    CONF_MATRIX_DIR, f"{MODEL_NAME}_single_conf_matrix.png"
)
conf_matrix_single_eps = os.path.join(
    CONF_MATRIX_DIR, f"{MODEL_NAME}_single_conf_matrix.eps"
)
plt.savefig(conf_matrix_single_png)
plt.savefig(conf_matrix_single_eps, format="eps")
plt.clf()
plt.close()
print(f"[INFO] Single-run confusion matrix saved to:\n{conf_matrix_single_png}\n{conf_matrix_single_eps}")

# -------------------------------------
# 5. Sub-Confusion Matrices (Class Chunks) - Counts
# -------------------------------------
unique_classes = sorted(y_test.unique())
num_classes = len(unique_classes)
num_class_chunks = math.ceil(num_classes / CLASS_CHUNK_SIZE)

print(f"\n[INFO] Creating sub-confusion matrices for class-chunks of size {CLASS_CHUNK_SIZE}.")

# We'll store each chunk's confusion matrix for the relative step
chunk_conf_matrices = []  # Will hold tuples of (chunk_classes, conf_matrix)

for i in range(num_class_chunks):
    start_idx = i * CLASS_CHUNK_SIZE
    end_idx = min(start_idx + CLASS_CHUNK_SIZE, num_classes)
    chunk_classes = unique_classes[start_idx:end_idx]

    # Filter test set to ONLY rows whose true label is in chunk_classes
    chunk_mask = y_test.isin(chunk_classes)
    X_chunk = X_test_scaled[chunk_mask]
    y_chunk = y_test[chunk_mask]

    if X_chunk.shape[0] == 0:
        print(f"\n[INFO] Chunk {i + 1}: No samples for classes {chunk_classes}. Skipping.")
        continue

    # Predict on this subset
    y_pred_chunk = model.predict(X_chunk)

    # Ignore predictions NOT in chunk_classes
    keep_mask = [pred in chunk_classes for pred in y_pred_chunk]
    X_chunk_filtered = X_chunk[keep_mask]
    y_chunk_filtered = y_chunk[keep_mask]
    y_pred_chunk_filtered = y_pred_chunk[keep_mask]

    # If everything got filtered out, skip
    if len(y_chunk_filtered) == 0:
        print(f"[INFO] Chunk {i + 1}: All predictions fell outside chunk classes {chunk_classes}. Skipping.")
        continue

    # Create confusion matrix for these classes only
    conf_matrix_chunk = confusion_matrix(
        y_chunk_filtered,
        y_pred_chunk_filtered,
        labels=chunk_classes
    )

    print(f"\n[INFO] Chunk {i + 1}/{num_class_chunks} | Classes: {chunk_classes}")
    print(f"Samples in chunk: {len(y_chunk)}; kept {len(y_chunk_filtered)} after ignoring out-of-chunk preds.")
    print("Chunk Confusion Matrix (Counts):")
    print(conf_matrix_chunk)

    # Save for the relative step
    chunk_conf_matrices.append((chunk_classes, conf_matrix_chunk))

    # Plot the sub-confusion matrix (Counts)
    plt.figure(figsize=(10, 8), dpi=150)
    ax = sns.heatmap(
        conf_matrix_chunk,
        annot=True,
        fmt="d",
        cmap=custom_palette,
        xticklabels=chunk_classes,
        yticklabels=chunk_classes,
        annot_kws={"fontsize": 14},
        linewidths=1,
        linecolor="black"
    )
    #ax.set_title(f"Chunk {i + 1} Confusion Matrix (Classes {start_idx}..{end_idx - 1})", fontsize=14)
    ax.set_xlabel("Predicted", fontsize=16)
    ax.set_ylabel("True", fontsize=16)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()

    chunk_file_png = os.path.join(
        CONF_MATRIX_DIR,
        f"{MODEL_NAME}_classes_{start_idx}_{end_idx - 1}_chunk_{i + 1}.png"
    )
    chunk_file_eps = os.path.join(
        CONF_MATRIX_DIR,
        f"{MODEL_NAME}_classes_{start_idx}_{end_idx - 1}_chunk_{i + 1}.eps"
    )
    plt.savefig(chunk_file_png)
    plt.savefig(chunk_file_eps, format="eps")
    plt.clf()
    plt.close()

    print(f"[INFO] Chunk {i + 1} confusion matrix saved to:\n  {chunk_file_png}\n  {chunk_file_eps}")

# -------------------------------------
# 6. Relative Confusion Matrix (All Classes)
# -------------------------------------
row_sums = conf_matrix_single.sum(axis=1, keepdims=True)
relative_conf_matrix_single = conf_matrix_single / np.maximum(row_sums, 1e-9)

print("\n[INFO] Relative (Row-Normalized) Confusion Matrix (All Classes):")
print(relative_conf_matrix_single)

plt.figure(figsize=(40, 30), dpi=300)
ax = sns.heatmap(
    relative_conf_matrix_single,
    annot=True,
    fmt=".2f",
    cmap=custom_palette,
    xticklabels=full_classes,
    yticklabels=full_classes,
    annot_kws={"fontsize": 12},
    linewidths=1,
    linecolor="black"
)
ax.set_title("Relative Confusion Matrix (All Classes)", fontsize=28)
ax.set_xlabel("Predicted", fontsize=24)
ax.set_ylabel("True", fontsize=24)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
plt.tight_layout()

rel_conf_matrix_single_png = os.path.join(
    CONF_MATRIX_DIR, f"{MODEL_NAME}_single_conf_matrix_REL.png"
)
rel_conf_matrix_single_eps = os.path.join(
    CONF_MATRIX_DIR, f"{MODEL_NAME}_single_conf_matrix_REL.eps"
)
plt.savefig(rel_conf_matrix_single_png)
plt.savefig(rel_conf_matrix_single_eps, format="eps")
plt.clf()
plt.close()
print(
    f"[INFO] Relative single-run confusion matrix saved to:\n{rel_conf_matrix_single_png}\n{rel_conf_matrix_single_eps}")

# -------------------------------------
# 7. Relative Sub-Confusion Matrices (Class Chunks)
# -------------------------------------
for i, (chunk_classes, cm_chunk) in enumerate(chunk_conf_matrices, start=1):
    row_sums_chunk = cm_chunk.sum(axis=1, keepdims=True)
    rel_cm_chunk = cm_chunk / np.maximum(row_sums_chunk, 1e-9)

    print(f"\n[INFO] Chunk {i} Relative Confusion Matrix:")
    print(rel_cm_chunk)

    plt.figure(figsize=(10, 8), dpi=150)
    ax = sns.heatmap(
        rel_cm_chunk,
        annot=True,
        fmt=".2f",
        cmap=custom_palette,
        xticklabels=chunk_classes,
        yticklabels=chunk_classes,
        annot_kws={"fontsize": 16},
        linewidths=1,
        linecolor="black"
    )
    #ax.set_title(f"Chunk {i} Relative Confusion Matrix", fontsize=14)
    ax.set_xlabel("Predicted", fontsize=16)
    ax.set_ylabel("True", fontsize=16)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=14)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=14)
    plt.tight_layout()

    chunk_file_png = os.path.join(
        CONF_MATRIX_DIR,
        f"{MODEL_NAME}_chunk_{i}_REL.png"
    )
    chunk_file_eps = os.path.join(
        CONF_MATRIX_DIR,
        f"{MODEL_NAME}_chunk_{i}_REL.eps"
    )
    plt.savefig(chunk_file_png)
    plt.savefig(chunk_file_eps, format="eps")
    plt.clf()
    plt.close()

    print(f"[INFO] Chunk {i} relative confusion matrix saved to:\n  {chunk_file_png}\n  {chunk_file_eps}")

# -------------------------------------
# 8. Write Classification Metrics to a LaTeX-Ready .txt
# -------------------------------------

# Convert the classification report into a dictionary
report_dict = classification_report(y_test, y_pred, output_dict=True)
accuracy_val = accuracy_score(y_test, y_pred)

latex_lines = []
latex_lines.append(r"\begin{table}[h!]")
latex_lines.append(r"\centering")
latex_lines.append(r"\begin{tabular}{lcccc}")
latex_lines.append(r"\hline")
latex_lines.append(r" & Precision & Recall & F1-score & Support \\")
latex_lines.append(r"\hline")

# Collect per-class metrics
for key, val in report_dict.items():
    # Skip 'accuracy' because it is a single float, will handle separately
    if key not in ["accuracy", "macro avg", "weighted avg"]:
        class_name = str(key)
        precision = f"{val['precision']:.4f}"
        recall = f"{val['recall']:.4f}"
        f1 = f"{val['f1-score']:.4f}"
        support = f"{int(val['support'])}"
        latex_lines.append(f"{class_name} & {precision} & {recall} & {f1} & {support}\\\\")

# Macro avg
macro = report_dict["macro avg"]
latex_lines.append(r"\hline")
latex_lines.append(
    f"Macro Avg & {macro['precision']:.4f} & {macro['recall']:.4f} & {macro['f1-score']:.4f} & {int(macro['support'])}\\\\"
)

# Weighted avg
weighted = report_dict["weighted avg"]
latex_lines.append(
    f"Weighted Avg & {weighted['precision']:.4f} & {weighted['recall']:.4f} & {weighted['f1-score']:.4f} & {int(weighted['support'])}\\\\"
)

latex_lines.append(r"\hline")
latex_lines.append(r"\end{tabular}")
latex_lines.append(r"\caption{Classification Report with Accuracy = " + f"{accuracy_val:.4f}" + r"}")
latex_lines.append(r"\label{tab:classification_report}")
latex_lines.append(r"\end{table}")

# Write to .txt file
latex_output_file = os.path.join(RESULT_TXT_DIR, f"{MODEL_NAME}_latex_classification_report.txt")
with open(latex_output_file, "w", encoding="utf-8") as f:
    for line in latex_lines:
        f.write(line + "\n")

print(f"\n[INFO] LaTeX classification report saved to: {latex_output_file}")
print("[INFO] You can copy the table directly from this file into your LaTeX document.")
