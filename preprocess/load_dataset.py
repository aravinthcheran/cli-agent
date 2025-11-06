import os
from datasets import load_dataset

# Get the project root directory (parent of preprocess)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets")

# Create datasets directory if it doesn't exist
os.makedirs(DATASETS_DIR, exist_ok=True)

print(f"📁 Datasets will be saved to: {DATASETS_DIR}\n")

# Load train data
print("🚀 Loading train dataset from HuggingFace...")
# Login using e.g. `huggingface-cli login` to access this dataset
ds_train = load_dataset("westenfelder/NL2SH-ALFA", "train")

print(ds_train)
print("Sample train record:")
print(ds_train["train"][0])

train_csv_path = os.path.join(DATASETS_DIR, "train.csv")
ds_train["train"].to_csv(train_csv_path)
print(f"✅ Train dataset saved as {train_csv_path}\n")


# Load test data
print("🚀 Loading test dataset from HuggingFace...")
ds_test = load_dataset("westenfelder/NL2SH-ALFA", "test")

print(ds_test)
print("Sample test record:")
print(ds_test["train"][0])

test_csv_path = os.path.join(DATASETS_DIR, "test.csv")
ds_test["train"].to_csv(test_csv_path)
print(f"✅ Test dataset saved as {test_csv_path}")
