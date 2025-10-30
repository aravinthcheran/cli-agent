import pandas as pd
import json

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


convert_train_csv_to_jsonl("train_clean.csv", "train.jsonl")
convert_test_csv_to_jsonl("test_clean.csv", "test.jsonl")
