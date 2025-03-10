# Classification of Obfuscation Techniques in LLVM IR: A CatBoost-Based Approach


## Overview
This repository contains code and data for classifying obfuscation techniques applied at the LLVM Intermediate Representation (IR) level. The approach combines IR2Vec embeddings with machine learning classifiers, including CatBoost and ExtraTrees, to distinguish between various obfuscation transformations. The study focuses on identifying obfuscation patterns in LLVM IR generated from Tigress-obfuscated C programs.


## Features
- **Data Preprocessing:** Cleans and splits the dataset into training and test sets while ensuring stratification across obfuscation classes.
- **Obfuscation Classification:** Implements CatBoost and ExtraTrees classifiers for detecting LLVM IR obfuscations based on IR2Vec vector embeddings.
- **Model Evaluation:** Generates confusion matrices, classification reports, and feature importance visualizations to assess model performance.
- **Automated Hyperparameter Tuning:** Uses Bayesian optimization for selecting optimal model parameters.


## Repository Structure
```

📂 IR2Vec_Obfuscation_Identification
├── DATA/                    # Folder containing dataset files
│   ├── program_vectors_done_cleaned_06032025.csv # needs to be extracted from the zip files
│   ├── data_stats.txt        # Human-readable dataset statistics, will be created by running the code
│   └── data_stats_latex.txt  # LaTeX-formatted dataset statistics, will be created by running the code
│
├── Results_CatBoost_Classifier/ # Will be created by running the code
│   ├── models/               # Saved CatBoost models and scalers
│   ├── confusion_matrices/   # Confusion matrices (PNG, EPS formats)
│   ├── result_txts/          # Classification reports and logs
│
├── Results_ExtraTrees_Classifier/ # Will be created by running the code
│   ├── models/               # Saved ExtraTrees models and scalers
│   ├── confusion_matrices/   # Confusion matrices (PNG, EPS formats)
│   ├── result_txts/          # Classification reports and logs
│
├── 00_train_test_split.py     # Data preprocessing and train-test split
├── 01_CatBoost_Classification.py  # CatBoost model training and tuning
├── 01_ExtraTreesClassification.py # ExtraTrees model training and tuning
├── 02_evaluation_CatBoost.py  # Model evaluation for CatBoost
├── 02_evaluation_ExtraTrees.py # Model evaluation for ExtraTrees
├── README.md                  # Project documentation
```

## Installation and Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/Classification_LLVM_IR_Obfuscation.git
cd Classification_LLVM_IR_Obfuscation
```

### 2. Install Dependencies
Ensure you have Python 3.x installed, then install the required packages:
```bash
pip install -r requirements.txt
```

Alternatively, set up a conda environment:
```bash
conda create --name llvm_ir_obfuscation python=3.9
conda activate llvm_ir_obfuscation
pip install -r requirements.txt
```

## Usage

### 1. Preprocessing the Dataset
Run the data preprocessing script to clean the dataset and create train-test splits:
```bash
python 00_train_test_split.py
```
This script:
- Loads the IR2Vec-based dataset
- Renames misformatted column names
- Removes spaces in column headers
- Splits data into training (80%) and testing (20%) subsets with stratification
- Outputs dataset statistics in human-readable and LaTeX formats

### 2. Train Classification Models

#### Train CatBoost Classifier:
```bash
python 01_CatBoost_Classification.py
```
This script:
- Trains a baseline CatBoost classifier
- Performs Bayesian optimization for hyperparameter tuning
- Saves the best model and scaler to `Results_CatBoost_Classifier/models/`
- Logs accuracy scores and best hyperparameters

#### Train ExtraTrees Classifier:
```bash
python 01_ExtraTreesClassification.py
```
This script:
- Trains a baseline ExtraTrees classifier
- Performs Bayesian optimization for hyperparameter tuning
- Saves the best model and scaler to `Results_ExtraTrees_Classifier/models/`
- Logs accuracy scores and best hyperparameters

### 3. Evaluate Trained Models

#### Evaluate CatBoost Model:
```bash
python 02_evaluation_CatBoost.py
```
This script:
- Loads the trained CatBoost model and scaler
- Computes classification reports and confusion matrices
- Saves results to `Results_CatBoost_Classifier/result_txts/`
- Generates and saves visualization plots

#### Evaluate ExtraTrees Model:
```bash
python 02_evaluation_ExtraTrees.py
```
This script:
- Loads the trained ExtraTrees model and scaler
- Computes classification reports and confusion matrices
- Saves results to `Results_ExtraTrees_Classifier/result_txts/`
- Generates and saves visualization plots

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

