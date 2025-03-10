import os
import pandas as pd
from difflib import get_close_matches
from sklearn.model_selection import train_test_split

# -----------------------------
# Paths & Filenames
# -----------------------------
INPUT_CSV = "./DATA/program_vectors_done_cleaned_06032025.csv"
CLEANED_CSV = "./DATA/program_vectors_done_cleaned_06032025.csv"
TRAIN_CSV = "./DATA/train_obfuscation_data_programs_cleaned_06032025.csv"
TEST_CSV  = "./DATA/test_obfuscation_data_programs_cleaned_06032025.csv"

# -----------------------------
# Load the Data
# -----------------------------
df = pd.read_csv(INPUT_CSV)

# -----------------------------
# Correct column name that may be close to "Obfuscation"
# -----------------------------
target_col = "Obfuscation"
possible_matches = get_close_matches(target_col, df.columns, cutoff=0.7)

if possible_matches:
    closest_match = possible_matches[0]
    if closest_match != target_col:
        df.rename(columns={closest_match: target_col}, inplace=True)
        print(f"Column '{closest_match}' was renamed to '{target_col}' for consistency.")
else:
    print("No columns close to 'Obfuscation' were found to rename.")

# -----------------------------
# Remove spaces from column names
# -----------------------------
df.columns = [col.replace(" ", "") for col in df.columns]

# Save cleaned data
df.to_csv(CLEANED_CSV, index=False)
print("Cleaned data saved successfully.")

# -----------------------------
# Split (80% Train, 20% Test) with stratification on "Obfuscation"
# -----------------------------
try:
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        shuffle=True,
        stratify=df["Obfuscation"]
    )
except KeyError:
    raise KeyError("Column 'Obfuscation' is missing. Please ensure the dataset contains this column.")

# -----------------------------
# Save to Disk
# -----------------------------
os.makedirs("./DATA", exist_ok=True)
train_df.to_csv(TRAIN_CSV, index=False)
test_df.to_csv(TEST_CSV, index=False)
print("Training and testing sets have been created and saved with stratification on 'Obfuscation'.")

# -----------------------------
# Generate Additional Data Statistics
# -----------------------------
num_features = len(df.columns)
num_samples_total = len(df)
num_samples_train = len(train_df)
num_samples_test = len(test_df)

if "Obfuscation" in df.columns:
    class_distribution_train = train_df["Obfuscation"].value_counts()
    class_distribution_test = test_df["Obfuscation"].value_counts()
else:
    class_distribution_train = {}
    class_distribution_test = {}

# Prepare human-readable stats
stats_human = []
stats_human.append("=== DATASET STATISTICS (HUMAN READABLE) ===")
stats_human.append(f"Total samples: {num_samples_total}")
stats_human.append(f"Number of features (columns): {num_features}")
stats_human.append(f"Training samples: {num_samples_train}")
stats_human.append(f"Testing samples: {num_samples_test}")
stats_human.append("Class distribution in training set:")
for cls, count in class_distribution_train.items():
    stats_human.append(f"  Class: {cls}, Count: {count}")
stats_human.append("Class distribution in testing set:")
for cls, count in class_distribution_test.items():
    stats_human.append(f"  Class: {cls}, Count: {count}")

# Prepare LaTeX-friendly stats
stats_latex = []
stats_latex.append("=== DATASET STATISTICS (LATEX FORMAT) ===")
stats_latex.append(f"Total samples: {num_samples_total} \\\\")
stats_latex.append(f"Number of features (columns): {num_features} \\\\")
stats_latex.append(f"Training samples: {num_samples_train} \\\\")
stats_latex.append(f"Testing samples: {num_samples_test} \\\\")
stats_latex.append("Class distribution in training set:\\\\")
for cls, count in class_distribution_train.items():
    stats_latex.append(f"  Class: {cls}, Count: {count} \\\\")
stats_latex.append("Class distribution in testing set:\\\\")
for cls, count in class_distribution_test.items():
    stats_latex.append(f"  Class: {cls}, Count: {count} \\\\")

# Write stats to text files
with open("./DATA/data_stats.txt", "w") as f:
    f.write("\n".join(stats_human) + "\n")

with open("./DATA/data_stats_latex.txt", "w") as f:
    f.write("\n".join(stats_latex) + "\n")
