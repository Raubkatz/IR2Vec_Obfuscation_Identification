"""
ExtraTrees evaluation for cascaded experiments (0..5).

Usage:
- Set exp_nr = 0..5 below.
- By default, the script auto-detects the newest saved model/scaler in:
    ./Results_ExtraTrees_Classifier/experiment_{exp_nr}/models/
  You can override by setting MODEL_NAME / SCALER_NAME explicitly.

Inputs:
    ./Experiments_Casc/experiment_{exp_nr}/DATA/test.csv

Targets by experiment (cascaded builder):
    0 -> "Obfuscation"
    1 -> "ObfBinary"
    2 -> "SingleVsLayer"
    3 -> "SingleMethod"
    4 -> "LayeredLabel"
    5 -> "OLevel_or_NoO"

Features:
    Prefer columns "0".."299" if present; else all numeric columns except the target.

Outputs:
    Confusion matrices (PNG/EPS) and LaTeX report under:
    ./Results_ExtraTrees_Classifier/experiment_{exp_nr}/confusion_matrices
    ./Results_ExtraTrees_Classifier/experiment_{exp_nr}/result_txts
"""

# ----------------------------------------------------
# Config
# ----------------------------------------------------
exp_nr = 0                  # <-- set 0..5
CLASS_CHUNK_SIZE = 7        # classes per sub-matrix
MODEL_NAME = None           # e.g., "BestModel_Exp0_20250306154011.pkl" (optional override)
SCALER_NAME = None          # e.g., "BestScaler_Exp0_20250306154011.pkl" (optional override)

# Font scaling (1.0 keeps current defaults; >1 larger, <1 smaller)
#FONT_SCALE = 1.0

FONT_SCALE = 1.2


# ----------------------------------------------------
# Imports
# ----------------------------------------------------
import os
import glob
import pickle
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import ExtraTreesClassifier

# ----------------------------------------------------
# Paths and experiment metadata
# ----------------------------------------------------
EXPERIMENT_ROOT = f"./Experiments_Casc/experiment_{exp_nr}/DATA"
TEST_DATA_PATH  = os.path.join(EXPERIMENT_ROOT, "test.csv")

TARGET_BY_EXPERIMENT = {
    0: "Obfuscation",
    1: "ObfBinary",
    2: "SingleVsLayer",
    3: "SingleMethod",
    4: "LayeredLabel",
    5: "OLevel_or_NoO",
}
TARGET_COLUMN = TARGET_BY_EXPERIMENT.get(exp_nr)
if TARGET_COLUMN is None:
    raise ValueError(f"Unsupported exp_nr={exp_nr}. Expected 0..5.")

RESULTS_ROOT = f"./Results_ExtraTrees_Classifier/experiment_{exp_nr}"
MODELS_DIR   = os.path.join(RESULTS_ROOT, "models")
CONF_DIR     = os.path.join(RESULTS_ROOT, "confusion_matrices")
TXT_DIR      = os.path.join(RESULTS_ROOT, "result_txts")
os.makedirs(CONF_DIR, exist_ok=True)
os.makedirs(TXT_DIR, exist_ok=True)

print("--------------EVAL START--------------")
print(f"Experiment: {exp_nr} | Target: {TARGET_COLUMN}")
print(f"Test CSV:   {TEST_DATA_PATH}")
print(f"Models dir: {MODELS_DIR}")

# ----------------------------------------------------
# Fixed plotting palette (as requested)
# ----------------------------------------------------
custom_palette = ["#188FA7", "#769FB6", "#9DBBAE", "#D5D6AA", "#E2DBBE"]

# Centralized font sizes (scaled)
FULL_TITLE_FS   = 28 * FONT_SCALE
FULL_LABEL_FS   = 24 * FONT_SCALE
FULL_ANNOT_FS   = 12 * FONT_SCALE

CHUNK_TITLE_FS  = 14 * FONT_SCALE
CHUNK_LABEL_FS  = 12 * FONT_SCALE
CHUNK_ANNOT_FS  = 10 * FONT_SCALE

REL_TITLE_FS    = 28 * FONT_SCALE
REL_LABEL_FS    = 24 * FONT_SCALE
REL_ANNOT_FS    = 12 * FONT_SCALE

# ----------------------------------------------------
# Helper: feature inference
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

# ----------------------------------------------------
# Helper: pick the newest model/scaler if names not provided
# ----------------------------------------------------
def newest_file(pattern: str):
    candidates = glob.glob(pattern)
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]

def resolve_artifacts(models_dir, model_name, scaler_name):
    """
    Resolve model (.pkl) and scaler (.pkl).
    If names are None, pick newest matching files.
    """
    # Model
    if model_name is None:
        mdl = newest_file(os.path.join(models_dir, "BestModel_*.pkl"))
        if mdl is None:
            raise FileNotFoundError(f"No model found in {models_dir}")
        model_path = mdl
    else:
        model_path = os.path.join(models_dir, model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Specified model not found: {model_path}")

    # Scaler
    if scaler_name is None:
        scaler_path = newest_file(os.path.join(models_dir, "BestScaler_*.pkl"))
        if scaler_path is None:
            raise FileNotFoundError(f"No scaler found in {models_dir}")
    else:
        scaler_path = os.path.join(models_dir, scaler_name)
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Specified scaler not found: {scaler_path}")

    return model_path, scaler_path

# ----------------------------------------------------
# 1) Load model and scaler
# ----------------------------------------------------
model_path, scaler_path = resolve_artifacts(MODELS_DIR, MODEL_NAME, SCALER_NAME)
print(f"[INFO] Using model:  {os.path.basename(model_path)}")
print(f"[INFO] Using scaler: {os.path.basename(scaler_path)}")

# Load scaler
with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)
if not isinstance(scaler, StandardScaler):
    raise TypeError("Loaded scaler is not a sklearn StandardScaler.")

# Load model
with open(model_path, "rb") as f:
    model = pickle.load(f)
if not isinstance(model, ExtraTreesClassifier):
    raise TypeError("Loaded model is not an ExtraTreesClassifier.")

# ----------------------------------------------------
# 2) Load and preprocess test data
# ----------------------------------------------------
if not os.path.isfile(TEST_DATA_PATH):
    raise FileNotFoundError(f"Test CSV not found: {TEST_DATA_PATH}")

print(f"[INFO] Reading test data from: {TEST_DATA_PATH}")
test_df = pd.read_csv(TEST_DATA_PATH)

if TARGET_COLUMN not in test_df.columns:
    raise KeyError(f"Target column '{TARGET_COLUMN}' not present in test.csv")

FEATURE_COLS = infer_feature_columns(test_df, TARGET_COLUMN)
X_test = test_df[FEATURE_COLS].copy()
y_test = test_df[TARGET_COLUMN].astype("category")

# Scale
X_test_scaled = scaler.transform(X_test)

# ----------------------------------------------------
# 3) Predictions and overall metrics
# ----------------------------------------------------
y_pred = model.predict(X_test_scaled)
y_pred = np.array(y_pred).ravel()

report_single_run = classification_report(y_test, y_pred, digits=4)
acc_val = accuracy_score(y_test, y_pred)

print("\n[INFO] Classification Report (All Classes):")
print(report_single_run)
print(f"[INFO] Accuracy: {acc_val:.4f}")

# ----------------------------------------------------
# 4) Full confusion matrix (counts)
# ----------------------------------------------------
full_classes = sorted(pd.unique(y_test))
cm_full = confusion_matrix(y_test, y_pred, labels=full_classes)

plt.figure(figsize=(10, 8), dpi=150)
ax = sns.heatmap(
    cm_full,
    annot=True,
    fmt="d",
    cmap=custom_palette,
    xticklabels=full_classes,
    yticklabels=full_classes,
    annot_kws={"fontsize": CHUNK_ANNOT_FS},
    linewidths=1,
    linecolor="black",
)
#ax.set_title("Confusion Matrix (All Classes) — Counts", fontsize=FULL_TITLE_FS)
ax.set_xlabel("Predicted", fontsize=CHUNK_ANNOT_FS)
ax.set_ylabel("True", fontsize=CHUNK_ANNOT_FS)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=CHUNK_ANNOT_FS,  rotation=45, ha="right")
ax.set_yticklabels(ax.get_yticklabels(), fontsize=CHUNK_ANNOT_FS,  rotation=0)
plt.tight_layout()

model_stub = os.path.splitext(os.path.basename(model_path))[0]
full_png = os.path.join(CONF_DIR, f"{model_stub}_full_counts.png")
full_eps = os.path.join(CONF_DIR, f"{model_stub}_full_counts.eps")
plt.savefig(full_png)
plt.savefig(full_eps, format="eps")
plt.clf()
plt.close()
print(f"[INFO] Full confusion matrix saved:\n  {full_png}\n  {full_eps}")

# ----------------------------------------------------
# 5) Chunked confusion matrices (counts)
# ----------------------------------------------------
unique_classes = full_classes
num_classes = len(unique_classes)
num_chunks = math.ceil(num_classes / CLASS_CHUNK_SIZE)

print(f"\n[INFO] Creating chunked matrices with chunk size = {CLASS_CHUNK_SIZE} "
      f"({num_chunks} chunks, total {num_classes} classes).")

chunk_conf_matrices = []  # list of (chunk_classes, cm_chunk)

for i in range(num_chunks):
    start_idx = i * CLASS_CHUNK_SIZE
    end_idx = min(start_idx + CLASS_CHUNK_SIZE, num_classes)
    chunk_classes = unique_classes[start_idx:end_idx]

    # Filter rows whose true label is in chunk
    mask = y_test.isin(chunk_classes)
    X_chunk = X_test_scaled[mask]
    y_chunk = y_test[mask]

    if X_chunk.shape[0] == 0:
        print(f"[INFO] Chunk {i+1}: no samples for classes {chunk_classes}. Skipping.")
        continue

    y_pred_chunk = model.predict(X_chunk)
    y_pred_chunk = np.array(y_pred_chunk).ravel()

    # Keep only predictions inside the chunk
    keep = np.array([p in chunk_classes for p in y_pred_chunk])
    X_chunk = X_chunk[keep]
    y_chunk = y_chunk[keep]
    y_pred_chunk = y_pred_chunk[keep]

    if len(y_chunk) == 0:
        print(f"[INFO] Chunk {i+1}: all predictions outside chunk. Skipping.")
        continue

    cm_chunk = confusion_matrix(y_chunk, y_pred_chunk, labels=chunk_classes)
    chunk_conf_matrices.append((chunk_classes, cm_chunk))

    print(f"\n[INFO] Chunk {i+1}/{num_chunks} | Classes: {chunk_classes}")
    print(f"  Samples in chunk: {mask.sum()} | Kept after filtering: {len(y_chunk)}")
    print("  Confusion matrix (counts):")
    print(cm_chunk)

    plt.figure(figsize=(10, 8), dpi=150)
    ax = sns.heatmap(
        cm_chunk,
        annot=True,
        fmt="d",
        cmap=custom_palette,
        xticklabels=chunk_classes,
        yticklabels=chunk_classes,
        annot_kws={"fontsize": CHUNK_ANNOT_FS},
        linewidths=1,
        linecolor="black",
    )
    #ax.set_title(f"Chunk {i + 1} — Counts", fontsize=CHUNK_TITLE_FS)
    ax.set_xlabel("Predicted", fontsize=CHUNK_LABEL_FS)
    ax.set_ylabel("True", fontsize=CHUNK_LABEL_FS)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=FULL_ANNOT_FS,  rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=FULL_ANNOT_FS,  rotation=0)
    plt.tight_layout()

    chunk_png = os.path.join(CONF_DIR, f"{model_stub}_chunk_{i+1}_counts.png")
    chunk_eps = os.path.join(CONF_DIR, f"{model_stub}_chunk_{i+1}_counts.eps")
    plt.savefig(chunk_png)
    plt.savefig(chunk_eps, format="eps")
    plt.clf()
    plt.close()
    print(f"[INFO] Saved:\n  {chunk_png}\n  {chunk_eps}")

# ----------------------------------------------------
# 6) Full confusion matrix (relative by true row)
# ----------------------------------------------------
row_sums = cm_full.sum(axis=1, keepdims=True)
cm_full_rel = cm_full / np.maximum(row_sums, 1e-9)

print("\n[INFO] Relative confusion matrix (all classes):")
print(cm_full_rel)

plt.figure(figsize=(40, 30), dpi=300)
ax = sns.heatmap(
    cm_full_rel,
    annot=True,
    fmt=".2f",
    cmap=custom_palette,
    xticklabels=full_classes,
    yticklabels=full_classes,
    annot_kws={"fontsize": REL_LABEL_FS},
    linewidths=1,
    linecolor="black",
)
#ax.set_title("Confusion Matrix (All Classes) — Relative", fontsize=REL_TITLE_FS)
ax.set_xlabel("Predicted", fontsize=REL_LABEL_FS)
ax.set_ylabel("True", fontsize=REL_LABEL_FS)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=FULL_ANNOT_FS,  rotation=45, ha="right")
ax.set_yticklabels(ax.get_yticklabels(), fontsize=FULL_ANNOT_FS,  rotation=0)
plt.tight_layout()

full_rel_png = os.path.join(CONF_DIR, f"{model_stub}_full_relative.png")
full_rel_eps = os.path.join(CONF_DIR, f"{model_stub}_full_relative.eps")
plt.savefig(full_rel_png)
plt.savefig(full_rel_eps, format="eps")
plt.clf()
plt.close()
print(f"[INFO] Full relative matrix saved:\n  {full_rel_png}\n  {full_rel_eps}")

# ----------------------------------------------------
# 7) Chunked confusion matrices (relative)
# ----------------------------------------------------
for i, (chunk_classes, cm_chunk) in enumerate(chunk_conf_matrices, start=1):
    row_sums_chunk = cm_chunk.sum(axis=1, keepdims=True)
    cm_chunk_rel = cm_chunk / np.maximum(row_sums_chunk, 1e-9)

    print(f"\n[INFO] Chunk {i} relative confusion matrix:")
    print(cm_chunk_rel)

    plt.figure(figsize=(10, 8), dpi=150)
    ax = sns.heatmap(
        cm_chunk_rel,
        annot=True,
        fmt=".2f",
        cmap=custom_palette,
        xticklabels=chunk_classes,
        yticklabels=chunk_classes,
        annot_kws={"fontsize": REL_ANNOT_FS},
        linewidths=1,
        linecolor="black",
    )
    #ax.set_title(f"Chunk {i} — Relative", fontsize=CHUNK_TITLE_FS)
    ax.set_xlabel("Predicted", fontsize=CHUNK_LABEL_FS)
    ax.set_ylabel("True", fontsize=CHUNK_LABEL_FS)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=FULL_ANNOT_FS,  rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=FULL_ANNOT_FS,  rotation=0)
    plt.tight_layout()

    chunk_rel_png = os.path.join(CONF_DIR, f"{model_stub}_chunk_{i}_relative.png")
    chunk_rel_eps = os.path.join(CONF_DIR, f"{model_stub}_chunk_{i}_relative.eps")
    plt.savefig(chunk_rel_png)
    plt.savefig(chunk_rel_eps, format="eps")
    plt.clf()
    plt.close()
    print(f"[INFO] Saved:\n  {chunk_rel_png}\n  {chunk_rel_eps}")

# ----------------------------------------------------
# 8) LaTeX-ready classification report
# ----------------------------------------------------
report_dict = classification_report(y_test, y_pred, output_dict=True)
accuracy_val = accuracy_score(y_test, y_pred)

latex_lines = []
latex_lines.append(r"\begin{table}[h!]")
latex_lines.append(r"\centering")
latex_lines.append(r"\begin{tabular}{lcccc}")
latex_lines.append(r"\hline")
latex_lines.append(r" & Precision & Recall & F1-score & Support \\")
latex_lines.append(r"\hline")

for key, val in report_dict.items():
    if key not in ["accuracy", "macro avg", "weighted avg"]:
        class_name = str(key)
        precision = f"{val['precision']:.4f}"
        recall = f"{val['recall']:.4f}"
        f1 = f"{val['f1-score']:.4f}"
        support = f"{int(val['support'])}"
        latex_lines.append(f"{class_name} & {precision} & {recall} & {f1} & {support}\\\\")
latex_lines.append(r"\hline")

macro = report_dict["macro avg"]
latex_lines.append(
    f"Macro Avg & {macro['precision']:.4f} & {macro['recall']:.4f} & {macro['f1-score']:.4f} & {int(macro['support'])}\\\\"
)

weighted = report_dict["weighted avg"]
latex_lines.append(
    f"Weighted Avg & {weighted['precision']:.4f} & {weighted['recall']:.4f} & {weighted['f1-score']:.4f} & {int(weighted['support'])}\\\\"
)

latex_lines.append(r"\hline")
latex_lines.append(r"\end{tabular}")
latex_lines.append(r"\caption{Classification Report with Accuracy = " + f"{accuracy_val:.4f}" + r"}")
latex_lines.append(r"\label{tab:classification_report}")
latex_lines.append(r"\end{table}")

latex_output_file = os.path.join(TXT_DIR, f"{model_stub}_latex_classification_report.txt")
with open(latex_output_file, "w", encoding="utf-8") as f:
    f.write("\n".join(latex_lines) + "\n")

print(f"\n[INFO] LaTeX classification report saved to: {latex_output_file}")
print("--------------EVAL END--------------")
