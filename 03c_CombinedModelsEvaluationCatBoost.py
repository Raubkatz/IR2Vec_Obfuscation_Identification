"""
CatBoost FINAL Cascading Evaluation (Exp0 test → Exp1→Exp2→(Exp3|Exp4)).

Flow:
  - Load Exp0 test set as the evaluation pool.
  - Stage 1: Exp1 model predicts ObfBinary ∈ {Non_Obfuscated, Obfuscated}.
      * If Non_Obfuscated → final label = the exact non-obf token present in Exp0 truth.
      * If Obfuscated → proceed to Stage 2.
  - Stage 2: Exp2 model predicts SingleVsLayer ∈ {Single_Obf, Layered_Obf}.
      * If Single_Obf  → Stage 3 (Exp3) predicts SingleMethod (canonical single names).
      * If Layered_Obf → Stage 4 (Exp4) predicts LayeredLabel (original layered strings).
  - Build final predictions for all rows and evaluate vs. Exp0 'Obfuscation'.

Inputs:
  ./Experiments_Casc/experiment_0/DATA/test.csv

Models (auto-picked unless overridden):
  ./Results_CatBoost_Classifier/experiment_1/models/BestModel_*.cbm[.pkl] + BestScaler_*.pkl
  ./Results_CatBoost_Classifier/experiment_2/models/BestModel_*.cbm[.pkl] + BestScaler_*.pkl
  ./Results_CatBoost_Classifier/experiment_3/models/BestModel_*.cbm[.pkl] + BestScaler_*.pkl
  ./Results_CatBoost_Classifier/experiment_4/models/BestModel_*.cbm[.pkl] + BestScaler_*.pkl

Features:
  Prefer "0".."299"; else use all numeric cols except the target.

Outputs:
  ./Results_CatBoost_Classifier/experiment_CASCADING/confusion_matrices/*.png|*.eps
  ./Results_CatBoost_Classifier/experiment_CASCADING/result_txts/*_classification_report.txt
  ./Results_CatBoost_Classifier/experiment_CASCADING/result_txts/*_latex_classification_report.txt
  ./Results_CatBoost_Classifier/experiment_CASCADING/result_txts/*_run_info.txt
"""

# ----------------------------------------------------
# Config
# ----------------------------------------------------
CLASS_CHUNK_SIZE = 7        # classes per sub-matrix
FONT_SCALE = 1.05            # 1.0 = default; >1 larger; <1 smaller

# ----------------------------------------------------
# Imports
# ----------------------------------------------------
import os
import glob
import pickle
import math
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from catboost import CatBoostClassifier

# ----------------------------------------------------
# Paths
# ----------------------------------------------------
EXP0_TEST_PATH = "./Experiments_Casc/experiment_0/DATA/test.csv"

RESULTS_ROOT = "./Results_CatBoost_Classifier/experiment_CASCADING"
MODELS_ROOT  = "./Results_CatBoost_Classifier"
CONF_DIR     = os.path.join(RESULTS_ROOT, "confusion_matrices")
TXT_DIR      = os.path.join(RESULTS_ROOT, "result_txts")
os.makedirs(CONF_DIR, exist_ok=True)
os.makedirs(TXT_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
run_stub  = f"CASC_{timestamp}"

print("--------------CASCADING EVAL START (CatBoost)--------------")
print(f"Exp0 test input: {EXP0_TEST_PATH}")

# ----------------------------------------------------
# Palette & Font Sizes
# ----------------------------------------------------
custom_palette = ["#188FA7", "#769FB6", "#9DBBAE", "#D5D6AA", "#E2DBBE"]

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
# Helpers
# ----------------------------------------------------
def newest_file(pattern: str):
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]

def load_model_and_scaler_for_experiment(exp_nr: int):
    """Auto-pick newest CatBoost BestModel (.cbm preferred, else .cbm.pkl/.pkl) + BestScaler for an experiment."""
    mdir = os.path.join(MODELS_ROOT, f"experiment_{exp_nr}", "models")
    mdl = (
        newest_file(os.path.join(mdir, "BestModel_*.cbm"))
        or newest_file(os.path.join(mdir, "BestModel_*.cbm.pkl"))
        or newest_file(os.path.join(mdir, "BestModel_*.pkl"))
    )
    sclr = newest_file(os.path.join(mdir, "BestScaler_*.pkl"))

    if mdl is None or sclr is None:
        raise FileNotFoundError(f"[Exp {exp_nr}] Could not locate model/scaler in {mdir}")

    # Load scaler
    with open(sclr, "rb") as f:
        scaler = pickle.load(f)
    if not isinstance(scaler, StandardScaler):
        raise TypeError(f"[Exp {exp_nr}] Scaler is not a sklearn StandardScaler.")

    # Load model (prefer native .cbm)
    if mdl.endswith(".cbm"):
        model = CatBoostClassifier()
        model.load_model(mdl)
    else:
        with open(mdl, "rb") as f:
            model = pickle.load(f)
    if not isinstance(model, CatBoostClassifier):
        raise TypeError(f"[Exp {exp_nr}] Model is not a CatBoostClassifier.")

    print(f"[INFO] Loaded Exp{exp_nr} model:  {os.path.basename(mdl)}")
    print(f"[INFO] Loaded Exp{exp_nr} scaler: {os.path.basename(sclr)}")
    return model, scaler, os.path.basename(mdl)

def infer_feature_columns(df: pd.DataFrame, target: str):
    pref = [str(i) for i in range(300)]
    if all(col in df.columns for col in pref):
        return pref
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target in num_cols:
        num_cols.remove(target)
    if not num_cols:
        raise ValueError("No numeric feature columns found.")
    return num_cols

def find_truth_non_obf_token(y_true: pd.Series) -> str:
    """Pick the exact 'non-obfuscated' token present in the truth (fallback to 'non-obfuscated')."""
    tokens = {str(s).strip() for s in y_true.unique()}
    for cand in tokens:
        if cand.lower() in {"non-obfuscated", "noobf", "no_obf", "no-o", "no_o", "none", "original"}:
            return cand
    return "non-obfuscated"

# ----------------------------------------------------
# Load Exp0 test set & features
# ----------------------------------------------------
if not os.path.isfile(EXP0_TEST_PATH):
    raise FileNotFoundError(f"Exp0 test.csv not found: {EXP0_TEST_PATH}")

test_df = pd.read_csv(EXP0_TEST_PATH)
if "Obfuscation" not in test_df.columns:
    raise KeyError("Exp0 test.csv must contain 'Obfuscation' column.")

FEATURE_COLS = infer_feature_columns(test_df, "Obfuscation")
X_all = test_df[FEATURE_COLS].copy()
y_true = test_df["Obfuscation"].astype("category")

non_obf_truth_token = find_truth_non_obf_token(y_true)

print(f"[INFO] Samples in Exp0 test: {len(test_df)}")
print(f"[INFO] Feature columns used: {len(FEATURE_COLS)}")
print(f"[INFO] Non-obf token in truth: '{non_obf_truth_token}'")

# ----------------------------------------------------
# Load cascade models (Exp1..Exp4)
# ----------------------------------------------------
# Exp1: ObfBinary
model1, scaler1, mdl1_name = load_model_and_scaler_for_experiment(1)
# Exp2: SingleVsLayer
model2, scaler2, mdl2_name = load_model_and_scaler_for_experiment(2)
# Exp3: SingleMethod
model3, scaler3, mdl3_name = load_model_and_scaler_for_experiment(3)
# Exp4: LayeredLabel
model4, scaler4, mdl4_name = load_model_and_scaler_for_experiment(4)

# ----------------------------------------------------
# Stage 1: Exp1 (ObfBinary)
# ----------------------------------------------------
X1 = scaler1.transform(X_all)
pred1 = np.array(model1.predict(X1)).ravel()  # 'Non_Obfuscated' | 'Obfuscated'

is_non_obf = (pred1 == "Non_Obfuscated")
is_obf     = ~is_non_obf

# Initialize final predictions
final_pred = np.full(shape=(len(test_df),), fill_value="UNKNOWN", dtype=object)
final_pred[is_non_obf] = non_obf_truth_token

print(f"[STAGE 1] Non_Obfuscated: {is_non_obf.sum()} | Obfuscated: {is_obf.sum()}")

# ----------------------------------------------------
# Stage 2: Exp2 (SingleVsLayer) → only for 'Obfuscated'
# ----------------------------------------------------
idx_obf = np.where(is_obf)[0]
if idx_obf.size > 0:
    X2 = scaler2.transform(X_all.iloc[idx_obf])
    pred2 = np.array(model2.predict(X2)).ravel()  # 'Single_Obf' | 'Layered_Obf'

    is_single = (pred2 == "Single_Obf")
    is_layer  = (pred2 == "Layered_Obf")
    print(f"[STAGE 2] Single_Obf: {is_single.sum()} | Layered_Obf: {is_layer.sum()}")

    # ------------------------------------------------
    # Stage 3: Exp3 (SingleMethod) for Single_Obf
    # ------------------------------------------------
    idx_single = idx_obf[is_single]
    if idx_single.size > 0:
        X3 = scaler3.transform(X_all.iloc[idx_single])
        pred3 = np.array(model3.predict(X3)).ravel()  # e.g., 'Flatten', 'EncodeLiterals', ...
        final_pred[idx_single] = pred3

    # ------------------------------------------------
    # Stage 4: Exp4 (LayeredLabel) for Layered_Obf
    # ------------------------------------------------
    idx_layer = idx_obf[is_layer]
    if idx_layer.size > 0:
        X4 = scaler4.transform(X_all.iloc[idx_layer])
        pred4 = np.array(model4.predict(X4)).ravel()  # original layered strings
        final_pred[idx_layer] = pred4

# Safety: stringify
final_pred = np.array([str(x) for x in final_pred])

# ----------------------------------------------------
# Evaluation vs. Exp0 truth
# ----------------------------------------------------
report = classification_report(y_true, final_pred, digits=4)
acc    = accuracy_score(y_true, final_pred)

print("\n[INFO] FINAL Cascading Classification Report (CatBoost):")
print(report)
print(f"[INFO] FINAL Accuracy: {acc:.4f}")

# ----------------------------------------------------
# Confusion matrices (counts + relative)
# ----------------------------------------------------
# Use the truth's label set for plotting consistency
full_classes = sorted(pd.unique(y_true))
cm_full = confusion_matrix(y_true, final_pred, labels=full_classes)

# Counts
plt.figure(figsize=(40, 30), dpi=300)
ax = sns.heatmap(
    cm_full, annot=True, fmt="d", cmap=custom_palette,
    xticklabels=full_classes, yticklabels=full_classes,
    annot_kws={"fontsize": FULL_ANNOT_FS},
    linewidths=1, linecolor="black"
)
#ax.set_title("CASCADING — Confusion Matrix (Counts)", fontsize=FULL_TITLE_FS)
ax.set_xlabel("Predicted", fontsize=FULL_LABEL_FS)
ax.set_ylabel("True", fontsize=FULL_LABEL_FS)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
plt.tight_layout()

full_counts_png = os.path.join(CONF_DIR, f"{run_stub}_full_counts.png")
full_counts_eps = os.path.join(CONF_DIR, f"{run_stub}_full_counts.eps")
plt.savefig(full_counts_png)
plt.savefig(full_counts_eps, format="eps")
plt.clf(); plt.close()
print(f"[INFO] Saved counts matrix:\n  {full_counts_png}\n  {full_counts_eps}")

# Chunked (counts)
unique_classes = full_classes
num_classes = len(unique_classes)
num_chunks = math.ceil(num_classes / CLASS_CHUNK_SIZE)
chunk_conf_matrices = []

print(f"\n[INFO] Creating chunked matrices (size={CLASS_CHUNK_SIZE}) "
      f"→ {num_chunks} chunks across {num_classes} classes.")

for i in range(num_chunks):
    start_idx = i * CLASS_CHUNK_SIZE
    end_idx   = min(start_idx + CLASS_CHUNK_SIZE, num_classes)
    chunk_cls = unique_classes[start_idx:end_idx]

    # Subset truth/preds to rows whose TRUE label is in this chunk
    mask = y_true.isin(chunk_cls)
    y_true_chunk = y_true[mask]
    y_pred_chunk = final_pred[mask]

    if y_true_chunk.shape[0] == 0:
        print(f"[INFO] Chunk {i+1}: no samples for classes {chunk_cls}. Skipping.")
        continue

    cm_chunk = confusion_matrix(y_true_chunk, y_pred_chunk, labels=chunk_cls)
    chunk_conf_matrices.append((chunk_cls, cm_chunk))

    print(f"\n[INFO] Chunk {i+1}/{num_chunks} | Classes: {chunk_cls}")
    print(f"  Samples: {y_true_chunk.shape[0]}")
    print("  Confusion matrix (counts):")
    print(cm_chunk)

    plt.figure(figsize=(10, 8), dpi=150)
    ax = sns.heatmap(
        cm_chunk, annot=True, fmt="d", cmap=custom_palette,
        xticklabels=chunk_cls, yticklabels=chunk_cls,
        annot_kws={"fontsize": CHUNK_ANNOT_FS},
        linewidths=1, linecolor="black"
    )
    #ax.set_title(f"CASC Chunk {i+1} — Counts", fontsize=CHUNK_TITLE_FS)
    ax.set_xlabel("Predicted", fontsize=CHUNK_LABEL_FS)
    ax.set_ylabel("True", fontsize=CHUNK_LABEL_FS)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()

    chunk_png = os.path.join(CONF_DIR, f"{run_stub}_chunk_{i+1}_counts.png")
    chunk_eps = os.path.join(CONF_DIR, f"{run_stub}_chunk_{i+1}_counts.eps")
    plt.savefig(chunk_png)
    plt.savefig(chunk_eps, format="eps")
    plt.clf(); plt.close()
    print(f"[INFO] Saved:\n  {chunk_png}\n  {chunk_eps}")

# Relative (full)
row_sums = cm_full.sum(axis=1, keepdims=True)
cm_full_rel = cm_full / np.maximum(row_sums, 1e-9)

print("\n[INFO] FULL relative confusion matrix:")
print(cm_full_rel)

plt.figure(figsize=(40, 30), dpi=300)
ax = sns.heatmap(
    cm_full_rel, annot=True, fmt=".2f", cmap=custom_palette,
    xticklabels=full_classes, yticklabels=full_classes,
    annot_kws={"fontsize": REL_ANNOT_FS},
    linewidths=1, linecolor="black"
)
#ax.set_title("CASCADING — Confusion Matrix (Relative by True Row)", fontsize=REL_TITLE_FS)
ax.set_xlabel("Predicted", fontsize=REL_LABEL_FS)
ax.set_ylabel("True", fontsize=REL_LABEL_FS)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
plt.tight_layout()

full_rel_png = os.path.join(CONF_DIR, f"{run_stub}_full_relative.png")
full_rel_eps = os.path.join(CONF_DIR, f"{run_stub}_full_relative.eps")
plt.savefig(full_rel_png)
plt.savefig(full_rel_eps, format="eps")
plt.clf(); plt.close()
print(f"[INFO] Saved relative matrix:\n  {full_rel_png}\n  {full_rel_eps}")

# Relative (chunks)
for i, (chunk_cls, cm_chunk) in enumerate(chunk_conf_matrices, start=1):
    row_sums_chunk = cm_chunk.sum(axis=1, keepdims=True)
    cm_chunk_rel = cm_chunk / np.maximum(row_sums_chunk, 1e-9)

    print(f"\n[INFO] Chunk {i} relative confusion matrix:")
    print(cm_chunk_rel)

    plt.figure(figsize=(10, 8), dpi=150)
    ax = sns.heatmap(
        cm_chunk_rel, annot=True, fmt=".2f", cmap=custom_palette,
        xticklabels=chunk_cls, yticklabels=chunk_cls,
        annot_kws={"fontsize": CHUNK_LABEL_FS + 2},
        linewidths=1, linecolor="black"
    )
    #ax.set_title(f"CASC Chunk {i} — Relative", fontsize=CHUNK_TITLE_FS)
    ax.set_xlabel("Predicted", fontsize=CHUNK_LABEL_FS + 2)
    ax.set_ylabel("True", fontsize=CHUNK_LABEL_FS + 2)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=CHUNK_LABEL_FS + 2)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=CHUNK_LABEL_FS + 2)
    plt.tight_layout()

    chunk_rel_png = os.path.join(CONF_DIR, f"{run_stub}_chunk_{i}_relative.png")
    chunk_rel_eps = os.path.join(CONF_DIR, f"{run_stub}_chunk_{i}_relative.eps")
    plt.savefig(chunk_rel_png)
    plt.savefig(chunk_rel_eps, format="eps")
    plt.clf(); plt.close()
    print(f"[INFO] Saved:\n  {chunk_rel_png}\n  {chunk_rel_eps}")

# ----------------------------------------------------
# Write reports
# ----------------------------------------------------
# Human-readable
clsrep_path = os.path.join(TXT_DIR, f"{run_stub}_classification_report.txt")
with open(clsrep_path, "w", encoding="utf-8") as f:
    f.write("FINAL Cascading Classification Report (CatBoost)\n")
    f.write(report + "\n")
    f.write(f"FINAL Accuracy: {acc:.4f}\n")
print(f"[INFO] Saved report: {clsrep_path}")

# LaTeX-ready
report_dict = classification_report(y_true, final_pred, output_dict=True)
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
        recall    = f"{val['recall']:.4f}"
        f1        = f"{val['f1-score']:.4f}"
        support   = f"{int(val['support'])}"
        latex_lines.append(f"{class_name} & {precision} & {recall} & {f1} & {support}\\\\")
latex_lines.append(r"\hline")
macro = report_dict["macro avg"]
latex_lines.append(
    f"Macro Avg & {macro['precision']:.4f} & {macro['recall']:.4f} "
    f"& {macro['f1-score']:.4f} & {int(macro['support'])}\\\\"
)
weighted = report_dict["weighted avg"]
latex_lines.append(
    f"Weighted Avg & {weighted['precision']:.4f} & {weighted['recall']:.4f} "
    f"& {weighted['f1-score']:.4f} & {int(weighted['support'])}\\\\"
)
latex_lines.append(r"\hline")
latex_lines.append(r"\end{tabular}")
latex_lines.append(r"\caption{Cascading Evaluation (CatBoost): Accuracy = " + f"{acc:.4f}" + r"}")
latex_lines.append(r"\label{tab:cascading_report_catboost}")
latex_lines.append(r"\end{table}")

latex_path = os.path.join(TXT_DIR, f"{run_stub}_latex_classification_report.txt")
with open(latex_path, "w", encoding="utf-8") as f:
    f.write("\n".join(latex_lines) + "\n")
print(f"[INFO] Saved LaTeX: {latex_path}")

# Run info (which models used, etc.)
runinfo_path = os.path.join(TXT_DIR, f"{run_stub}_run_info.txt")
with open(runinfo_path, "w", encoding="utf-8") as f:
    f.write(f"Run: {run_stub}\n")
    f.write(f"Exp0 test: {EXP0_TEST_PATH}\n")
    f.write(f"Models used:\n")
    f.write(f"  Exp1: {mdl1_name}\n")
    f.write(f"  Exp2: {mdl2_name}\n")
    f.write(f"  Exp3: {mdl3_name}\n")
    f.write(f"  Exp4: {mdl4_name}\n")
    f.write(f"Feature columns: {len(FEATURE_COLS)}\n")
print(f"[INFO] Saved run info: {runinfo_path}")

print("--------------CASCADING EVAL END (CatBoost)--------------")
