import os
import pandas as pd
import json

# Get the project root directory (parent of preprocess)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets")

print(f"📁 Working with datasets in: {DATASETS_DIR}\n")

# ---- Convert Train Dataset ----
def convert_train_csv_to_jsonl(train_csv, output_file):
    df = pd.read_csv(train_csv)
    with open(output_file, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            record = {
                "instruction": row["nl"],
                "response": row["bash"]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"✅ Train dataset saved to {output_file}")


# ---- Convert Test Dataset ----
def convert_test_csv_to_jsonl(test_csv, output_file):
    df = pd.read_csv(test_csv)
    with open(output_file, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            responses = [row["bash"]]
            if "bash2" in df.columns and pd.notna(row["bash2"]):
                responses.append(row["bash2"])
            
            record = {
                "instruction": row["nl"],
                "responses": responses,
                "difficulty": int(row["difficulty"])
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"✅ Test dataset saved to {output_file}")


if __name__ == "__main__":
    # Define input and output paths
    train_input = os.path.join(DATASETS_DIR, "train_clean.csv")
    train_output = os.path.join(DATASETS_DIR, "train.jsonl")
    test_input = os.path.join(DATASETS_DIR, "test_clean.csv")
    test_output = os.path.join(DATASETS_DIR, "test.jsonl")
    
    # Check if input files exist
    if not os.path.exists(train_input):
        print(f"❌ Error: {train_input} not found. Please run preprocess.py first.")
        exit(1)
    if not os.path.exists(test_input):
        print(f"❌ Error: {test_input} not found. Please run preprocess.py first.")
        exit(1)
    
    # Run conversions
    convert_train_csv_to_jsonl(train_input, train_output)
    convert_test_csv_to_jsonl(test_input, test_output)
