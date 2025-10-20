"""
Cascaded experiments with a single global split (Experiments_Casc).

Pipeline:
1) Load & clean source CSV.
2) Harmonize 'Obfuscation' column name and strip spaces from column names.
3) Remove rows whose entire feature vector is zero.
3b) Remove rows whose obfuscation is exactly 'Ident' (global drop, before split).
4) Make ONE stratified train/test split on 'Obfuscation'.
5) For experiments 0..5, apply experiment-specific filters/targets
   to the *already split* train/test data; save outputs and stats.

Folder layout:
  ./Experiments_Casc/
    BASE/
      DATA/{cleaned.csv, train.csv, test.csv, data_stats*.txt}
    experiment_0/
      DATA/{cleaned.csv, train.csv, test.csv, data_stats*.txt}
    ...
    experiment_5/
      DATA/{cleaned.csv, train.csv, test.csv, data_stats*.txt}

Experiments:
  Exp 0 (multiclass): original 'Obfuscation' (no filtering).
  Exp 1 (binary):     Obfuscated vs Non_Obfuscated, DROP 'Ident'.
  Exp 2 (binary):     Single_Obf vs Layered_Obf, DROP Non_Obf and 'Ident'.
  Exp 3 (multiclass): only single obfuscations, classify single method.
  Exp 4 (multiclass): only layered obfuscations, classify layered label.
  Exp 5 (multiclass): O-level identification (O-level or 'No_O').

Notes:
- 'non-obfuscated' (case-insensitive) is treated as Non_Obf.
- 'Ident' rows are removed globally (step 3b) before the split.
- Layered vs Single is inferred by detecting multiple method tokens
  in the class label (e.g., 'FlattenEncodeArithmetic' -> layered).
  'TigressRecipe1/2' are treated as layered.
"""

import os
from difflib import get_close_matches
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ----------------------------------------------------
# Config
# ----------------------------------------------------
SRC_CSV = "./DATA/program_vectors_done_cleaned_06032025.csv"

CASC_ROOT = "./Experiments_Casc"
BASE_DIR  = os.path.join(CASC_ROOT, "BASE", "DATA")

RANDOM_STATE = 42
TEST_SIZE    = 0.2

# Prefer feature columns "0".."299"; fallback to numeric columns (excl. targets)
PREFERRED_FEATURE_COUNT = 300

# ----------------------------------------------------
# Utilities
# ----------------------------------------------------
def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _write_stats(
    out_dir: str,
    full_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_name: str,
) -> None:
    num_features = len(full_df.columns)
    num_samples_total = len(full_df)
    num_samples_train = len(train_df)
    num_samples_test = len(test_df)

    if target_name in full_df.columns:
        class_distribution_train = train_df[target_name].value_counts(dropna=False)
        class_distribution_test = test_df[target_name].value_counts(dropna=False)
    else:
        class_distribution_train = pd.Series(dtype=int)
        class_distribution_test = pd.Series(dtype=int)

    stats_human = [
        "=== DATASET STATISTICS (HUMAN READABLE) ===",
        f"Target column: {target_name}",
        f"Total samples: {num_samples_total}",
        f"Number of features (columns): {num_features}",
        f"Training samples: {num_samples_train}",
        f"Testing samples: {num_samples_test}",
        "Class distribution in training set:",
        *[f"  Class: {cls}, Count: {count}" for cls, count in class_distribution_train.items()],
        "Class distribution in testing set:",
        *[f"  Class: {cls}, Count: {count}" for cls, count in class_distribution_test.items()],
    ]
    stats_latex = [
        "=== DATASET STATISTICS (LATEX FORMAT) ===",
        f"Target column: {target_name} \\\\",
        f"Total samples: {num_samples_total} \\\\",
        f"Number of features (columns): {num_features} \\\\",
        f"Training samples: {num_samples_train} \\\\",
        f"Testing samples: {num_samples_test} \\\\",
        "Class distribution in training set:\\\\",
        *[f"  Class: {cls}, Count: {count} \\\\" for cls, count in class_distribution_train.items()],
        "Class distribution in testing set:\\\\",
        *[f"  Class: {cls}, Count: {count} \\\\" for cls, count in class_distribution_test.items()],
    ]
    with open(os.path.join(out_dir, "data_stats.txt"), "w") as f:
        f.write("\n".join(stats_human) + "\n")
    with open(os.path.join(out_dir, "data_stats_latex.txt"), "w") as f:
        f.write("\n".join(stats_latex) + "\n")

def _infer_feature_cols(df: pd.DataFrame, target_cols: List[str]) -> List[str]:
    pref = [str(i) for i in range(PREFERRED_FEATURE_COUNT)]
    pref_present = [c for c in pref if c in df.columns]
    if len(pref_present) == PREFERRED_FEATURE_COUNT:
        return pref
    # fallback: all numeric columns except targets
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for t in target_cols:
        if t in num_cols:
            num_cols.remove(t)
    if not num_cols:
        raise ValueError("No numeric feature columns found.")
    return num_cols

# ----------------------------------------------------
# Label normalization & layered/single inference
# ----------------------------------------------------
NON_OBF_TOKENS = {
    "non-obfuscated", "none", "no_o", "noo", "noobf", "original",
    "none_obfuscation", "noneobfuscation", "noobfuscation", ""
}

# Base method tokens observed in the class list
METHOD_TOKENS = [
    "AntiAliasAnalysis",
    "AntiTaintAnalysis",
    "EncodeArithmetic",
    "EncodeBranches",
    "EncodeLiterals",
    "SelfModify",
    "Virtualize",
    "Flatten",
    "Split",
    "Ident",
    "TigressRecipe2",
    "TigressRecipe1",
]

# Sort tokens by length (desc) to avoid partial overlaps when matching
METHOD_TOKENS = sorted(METHOD_TOKENS, key=len, reverse=True)

def _norm(s) -> str:
    if pd.isna(s):
        return ""
    return str(s).strip()

def _is_non_obfuscated(label: str) -> bool:
    return _norm(label).lower() in NON_OBF_TOKENS

def _parse_methods(label: str) -> List[str]:
    s = _norm(label)
    methods = []
    if not s:
        return methods
    for tok in METHOD_TOKENS:
        if tok in s:
            methods.append(tok)
    # Special case: if label explicitly contains "TigressRecipe", keep only the recipe token(s)
    recipe_methods = [m for m in methods if m.startswith("TigressRecipe")]
    if recipe_methods:
        # Treat recipe as the defining method set
        return recipe_methods
    return methods

def _layer_type(label: str) -> str:
    """
    Returns one of {'No_Obf','Single_Obf','Layered_Obf','Ident'}.
    - 'No_Obf'       : explicit non-obfuscation tokens (e.g., 'non-obfuscated').
    - 'Ident'        : class exactly containing 'Ident' and nothing else.
    - 'Single_Obf'   : exactly one method token (not 'Ident' and not recipe).
    - 'Layered_Obf'  : >=2 method tokens OR any 'TigressRecipe*'.
    """
    if _is_non_obfuscated(label):
        return "No_Obf"
    methods = _parse_methods(label)
    # Explicit Ident
    if methods == ["Ident"]:
        return "Ident"
    # Any TigressRecipe => layered
    if any(m.startswith("TigressRecipe") for m in methods):
        return "Layered_Obf"
    # Count methods
    if len(methods) == 0:
        # Unknown tokenization; conservatively treat as Single if non-empty label
        return "Single_Obf"
    if len(methods) == 1:
        if methods[0] == "Ident":
            return "Ident"
        return "Single_Obf"
    return "Layered_Obf"

def _single_method_name(label: str) -> str:
    """
    For single obfuscations return the canonical single method name.
    Assumes caller checked _layer_type(label) == 'Single_Obf'.
    """
    methods = _parse_methods(label)
    if methods:
        return methods[0]
    # Fallback: return original label
    return _norm(label)

# ----------------------------------------------------
# 1) Load & clean
# ----------------------------------------------------
df = pd.read_csv(SRC_CSV)

# Harmonize 'Obfuscation' column name
target_col = "Obfuscation"
possible_matches = get_close_matches(target_col, df.columns, cutoff=0.7)
if possible_matches:
    closest = possible_matches[0]
    if closest != target_col:
        df.rename(columns={closest: target_col}, inplace=True)
else:
    if target_col not in df.columns:
        raise KeyError("Column 'Obfuscation' not found and no close match detected.")

# Strip spaces from all column names
df.columns = [c.replace(" ", "") for c in df.columns]

# Detect O-level column if present
o_level_col = None
for c in df.columns:
    if c.replace(" ", "").lower() in {"o-level", "olevel", "o_level"}:
        o_level_col = c
        break

# Identify feature columns and drop all-zero vectors
feature_cols = _infer_feature_cols(df, target_cols=[target_col] + ([o_level_col] if o_level_col else []))
zero_vec_mask = (df[feature_cols].astype(float) == 0).all(axis=1)
n_zero = int(zero_vec_mask.sum())
if n_zero > 0:
    print(f"[INFO] Dropping {n_zero} rows with all-zero feature vectors.")
df_nz = df.loc[~zero_vec_mask].copy()

# NEW: remove all 'Ident' rows globally before the split
ident_mask = df_nz[target_col].apply(lambda s: _layer_type(s) == "Ident")
n_ident = int(ident_mask.sum())
if n_ident > 0:
    print(f"[INFO] Dropping {n_ident} rows with obfuscation 'Ident'.")
df_nz = df_nz.loc[~ident_mask].copy()

# ----------------------------------------------------
# 2) ONE global split (stratified by 'Obfuscation')
# ----------------------------------------------------
train_base, test_base = train_test_split(
    df_nz, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True, stratify=df_nz[target_col]
)

# Save BASE snapshot
_safe_mkdir(BASE_DIR)
df_nz.to_csv(os.path.join(BASE_DIR, "cleaned.csv"), index=False)
train_base.to_csv(os.path.join(BASE_DIR, "train.csv"), index=False)
test_base.to_csv(os.path.join(BASE_DIR, "test.csv"), index=False)
_write_stats(BASE_DIR, df_nz, train_base, test_base, target_name=target_col)
print("[BASE] Global split saved (stratified on 'Obfuscation').")

# ----------------------------------------------------
# Helper to build & save each experiment from the base split
# ----------------------------------------------------
def _save_experiment_from_base(
    exp_idx: int,
    build_targets_and_filter_fn,
    target_name: str,
    note: str,
) -> None:
    """
    Applies experiment-specific filtering and target creation on top of the
    already-split base train/test DataFrames, then saves to disk with stats.
    """
    exp_dir = os.path.join(CASC_ROOT, f"experiment_{exp_idx}", "DATA")
    _safe_mkdir(exp_dir)

    # Start from the unfiltered base split
    tr = train_base.copy()
    te = test_base.copy()

    # Create targets and filter in train/test consistently
    tr = build_targets_and_filter_fn(tr)
    te = build_targets_and_filter_fn(te)

    # Drop rows with missing target after filtering (safety)
    tr = tr.loc[tr[target_name].astype(str).str.len() > 0]
    te = te.loc[te[target_name].astype(str).str.len() > 0]

    # Save cleaned (concatenated) view for the experiment
    cleaned = pd.concat([tr, te], axis=0, ignore_index=True)
    cleaned.to_csv(os.path.join(exp_dir, "cleaned.csv"), index=False)

    # Save splits
    tr.to_csv(os.path.join(exp_dir, "train.csv"), index=False)
    te.to_csv(os.path.join(exp_dir, "test.csv"), index=False)

    # Stats
    _write_stats(exp_dir, cleaned, tr, te, target_name=target_name)
    print(f"[experiment_{exp_idx}] Saved. Target='{target_name}'. {note}")

# ----------------------------------------------------
# Experiment builders
# ----------------------------------------------------
# Exp 0: Original multiclass 'Obfuscation' (no filtering)
def _exp0_builder(df_in: pd.DataFrame) -> pd.DataFrame:
    # No new target; ensure column exists as target
    return df_in.copy()

# Exp 1: Obfuscated vs Non_Obfuscated (drop 'Ident' — already removed globally)
def _exp1_builder(df_in: pd.DataFrame) -> pd.DataFrame:
    d = df_in.copy()
    d["ObfBinary"] = d[target_col].apply(lambda s: "Non_Obfuscated" if _is_non_obfuscated(s) else "Obfuscated")
    return d

# Exp 2: Single_Obf vs Layered_Obf (drop Non_Obf; 'Ident' already removed globally)
def _exp2_builder(df_in: pd.DataFrame) -> pd.DataFrame:
    d = df_in.copy()
    lt = d[target_col].apply(_layer_type)
    keep_mask = lt.isin(["Single_Obf", "Layered_Obf"])
    d = d.loc[keep_mask].copy()
    d["SingleVsLayer"] = d[target_col].apply(lambda s: "Single_Obf" if _layer_type(s) == "Single_Obf" else "Layered_Obf")
    return d

# Exp 3: Only single obfuscations, classify single method (multiclass)
def _exp3_builder(df_in: pd.DataFrame) -> pd.DataFrame:
    d = df_in.copy()
    lt = d[target_col].apply(_layer_type)
    d = d.loc[lt == "Single_Obf"].copy()
    d["SingleMethod"] = d[target_col].apply(_single_method_name)
    return d

# Exp 4: Only layered obfuscations, classify layered label (multiclass)
def _exp4_builder(df_in: pd.DataFrame) -> pd.DataFrame:
    d = df_in.copy()
    lt = d[target_col].apply(_layer_type)
    d = d.loc[lt == "Layered_Obf"].copy()
    # Use the original 'Obfuscation' string as the class label
    d["LayeredLabel"] = d[target_col].astype(str)
    return d

# Exp 5: O-level vs specific O-levels (or No_O)
def _exp5_builder(df_in: pd.DataFrame) -> pd.DataFrame:
    d = df_in.copy()
    if o_level_col is None:
        # Proxy: if non-obfuscated => 'No_O' else 'O_Unknown'
        d["OLevel_or_NoO"] = d[target_col].apply(lambda s: "No_O" if _is_non_obfuscated(s) else "O_Unknown")
    else:
        tmp = d[o_level_col].astype("object")
        tmp = tmp.where(~tmp.isna(), "No_O")
        tmp = tmp.apply(lambda v: "No_O" if _norm(v) == "" else v)
        # Force No_O if non-obfuscated
        d["OLevel_or_NoO"] = [
            "No_O" if _is_non_obfuscated(ov) else (tv if _norm(tv) != "" else "O_Unknown")
            for ov, tv in zip(d[target_col], tmp)
        ]
    return d

# ----------------------------------------------------
# Build and save all experiments from the single base split
# ----------------------------------------------------
# Exp 0
_save_experiment_from_base(
    exp_idx=0,
    build_targets_and_filter_fn=_exp0_builder,
    target_name=target_col,
    note="Multiclass original 'Obfuscation'.",
)

# Exp 1
_save_experiment_from_base(
    exp_idx=1,
    build_targets_and_filter_fn=_exp1_builder,
    target_name="ObfBinary",
    note="Binary Obfuscated vs Non_Obfuscated (Ident removed globally).",
)

# Exp 2
_save_experiment_from_base(
    exp_idx=2,
    build_targets_and_filter_fn=_exp2_builder,
    target_name="SingleVsLayer",
    note="Binary Single_Obf vs Layered_Obf (Non_Obf kept out; Ident removed globally).",
)

# Exp 3
_save_experiment_from_base(
    exp_idx=3,
    build_targets_and_filter_fn=_exp3_builder,
    target_name="SingleMethod",
    note="Multiclass among single obfuscations only.",
)

# Exp 4
_save_experiment_from_base(
    exp_idx=4,
    build_targets_and_filter_fn=_exp4_builder,
    target_name="LayeredLabel",
    note="Multiclass among layered obfuscations only.",
)

# Exp 5
_save_experiment_from_base(
    exp_idx=5,
    build_targets_and_filter_fn=_exp5_builder,
    target_name="OLevel_or_NoO",
    note="O-level identification (incl. No_O).",
)

print("All cascaded experiments (0..5) prepared under ./Experiments_Casc/")
