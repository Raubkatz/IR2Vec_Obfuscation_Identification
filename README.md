# Classification of Obfuscation Techniques in LLVM IR: A Cascaded and Combined Evaluation Framework

## Overview
This repository provides a complete experimental pipeline for classifying obfuscation techniques applied at the LLVM Intermediate Representation (IR) level.  
The workflow extends the original IR2Vec-based classification approach by introducing **cascaded experiments** (Exp0–Exp5) and **final combined evaluations** using both **CatBoost** and **ExtraTrees** classifiers.

The cascaded structure progressively learns to:
1. Detect obfuscated vs non-obfuscated code (binary classification).
2. Differentiate between single and layered obfuscations.
3. Classify specific single obfuscation methods.
4. Identify layered combinations.
5. Optionally detect O-level or non-obfuscated states.

The **combined evaluation scripts** merge the predictions from multiple experiment stages into a unified label prediction process, producing detailed confusion matrices and accuracy reports.

## Features
- **Cascaded Experiment Generation:** Performs a global train-test split and derives six experiment datasets (Exp0–Exp5) with hierarchical filtering logic.
- **Model Training (CatBoost & ExtraTrees):** Supports both out-of-the-box and Bayesian-optimized configurations.
- **Per-Experiment Evaluation:** Computes confusion matrices and LaTeX classification reports for all cascaded experiments.
- **Cascading Model Evaluation:** Performs full hierarchical classification by sequentially combining Exp1→Exp2→(Exp3|Exp4).
- **Combined Evaluation Framework:** Merges outputs from cascaded CatBoost and ExtraTrees classifiers for holistic benchmarking.
- **Automatic Artifact Handling:** Detects newest model/scaler artifacts from result directories.
- **Automated Visualization:** Generates both full and chunked confusion matrices for clearer per-class analysis.

## Repository Structure

📂 IR2Vec_Obfuscation_Identification

├── DATA/

│ └── program_vectors_done_cleaned_06032025.csv # Source feature dataset

│

├── Experiments_Casc/

│ ├── BASE/DATA/ # Global cleaned and split dataset

│ ├── experiment_0/DATA/ # Exp0: All samples (Obfuscation)

│ ├── experiment_1/DATA/ # Exp1: ObfBinary (Non_Obf vs Obf)

│ ├── experiment_2/DATA/ # Exp2: SingleVsLayer

│ ├── experiment_3/DATA/ # Exp3: SingleMethod

│ ├── experiment_4/DATA/ # Exp4: LayeredLabel

│ ├── experiment_5/DATA/ # Exp5: OLevel_or_NoO

│

├── Results_CatBoost_Classifier/

│ ├── experiment_0/…experiment_5/ # Per-experiment models, reports

│ └── experiment_CASCADING/ # Combined cascading evaluation outputs

│

├── Results_ExtraTrees_Classifier/

│ ├── experiment_0/…experiment_5/ # Per-experiment models, reports

│ └── experiment_CASCADING/ # Combined cascading evaluation outputs

│

├── 00c_cascading_experimental_setup_train_test_split.py # Global split and experiment builder

├── 01c_cascading_CatBoostClassification.py # CatBoost training across Exp0–Exp5

├── 01c_cascading_ExtraTreesClassification.py # ExtraTrees training across Exp0–Exp5

├── 02c_Cascading_Evaluation_CatBoost.py # Evaluate CatBoost models per experiment

├── 02c_Cascading_Evaluation_ExtraTress.py # Evaluate ExtraTrees models per experiment

├── 03c_CombinedModelsEvaluationCatBoost.py # Full cascading CatBoost evaluation (Exp1→Exp4)

├── 03c_CombinedModelsEvalautionExtraTrees.py # Full cascading ExtraTrees evaluation (Exp1→Exp4)

├── README.md # Project documentation


## Running the Code

1. Generate Cascaded Experiments
--------------------------------
Build all experiment datasets (Exp0–Exp5) using the global split:

    python 00c_cascading_experimental_setup_train_test_split.py

This script:
- Loads and cleans the raw IR2Vec-based dataset.
- Removes zero-vector and “Ident” samples.
- Normalizes the Obfuscation column and harmonizes labels.
- Performs one global stratified train-test split.
- Creates and saves experiment-specific subsets (Exp0–Exp5) under `Experiments_Casc/`.

2. Train Classification Models
------------------------------

Train CatBoost Models
---------------------
    python 01c_cascading_CatBoostClassification.py

This script:
- Trains CatBoost models for a specified experiment (`exp_nr = 0..5`).
- Performs both Out-of-the-Box and Bayesian optimization phases.
- Saves best-performing `.cbm` and `.cbm.pkl` models and their corresponding scalers.
- Logs results and hyperparameters under `Results_CatBoost_Classifier/experiment_{exp_nr}/`.

Train ExtraTrees Models
-----------------------
    python 01c_cascading_ExtraTreesClassification.py

This script:
- Trains ExtraTrees classifiers across Exp0–Exp5.
- Performs Out-of-the-Box and Bayesian optimization using `skopt`.
- Stores the best model and scaler in the experiment’s `models/` folder.

3. Evaluate Per-Experiment Models
---------------------------------
Evaluate CatBoost
-----------------
    python 02c_Cascading_Evaluation_CatBoost.py

Evaluate ExtraTrees
-------------------
    python 02c_Cascading_Evaluation_ExtraTress.py

Both scripts:
- Automatically locate the newest saved models and scalers.
- Evaluate performance on the corresponding experiment test set.
- Output accuracy, classification reports, and confusion matrices in PNG and EPS formats.
- Save detailed LaTeX reports under each `result_txts/` directory.

4. Perform Combined Cascading Evaluation
----------------------------------------
Cascading CatBoost Evaluation
-----------------------------
    python 03c_CombinedModelsEvaluationCatBoost.py

Cascading ExtraTrees Evaluation
-------------------------------
    python 03c_CombinedModelsEvalautionExtraTrees.py

These scripts:
- Sequentially apply models from Exp1→Exp2→(Exp3|Exp4) to classify the Exp0 test set.
- Automatically detect newest artifacts.
- Merge hierarchical predictions into a final global label.
- Generate multi-level confusion matrices (full and chunked versions).
- Save all reports under:
      ./Results_CatBoost_Classifier/experiment_CASCADING/
      ./Results_ExtraTrees_Classifier/experiment_CASCADING/


## Requirements

To run this project, install the following dependencies:


    matplotlib==3.6.3
    numpy==1.24.0
    pandas==1.5.3
    scikit-learn==1.1.3
    scipy==1.9.3
    seaborn==0.11.2
    shap==0.39.0
    tqdm==4.64.0
    catboost==1.2.7
    scikit-optimize==0.10.2

