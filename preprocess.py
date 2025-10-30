import pandas as pd
import re

# ---- Utility Functions ----
def clean_text(text: str) -> str:
    """
    Basic cleaning for natural language (nl) and bash commands:
    - Strip leading/trailing spaces
    - Collapse multiple spaces
    - Remove invisible control characters
    """
    if pd.isna(text):
        return ""
    text = str(text).strip()
    # text = re.sub(r"\s+", " ", text)  # collapse multiple spaces
    # text = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)  # remove control chars
    return text


def check_missing_values(df, name="dataset"):
    """
    Check for missing or empty rows in important columns.
    """
    print(f"\n🔍 Checking missing values in {name}...")
    for col in df.columns:
        missing = df[col].isna().sum()
        empty = (df[col].astype(str).str.strip() == "").sum()
        print(f"   → Column '{col}': {missing} missing, {empty} empty")
    print("✅ Missing value check done.")


def remove_duplicates(df, subset, name="dataset"):
    """
    Print and remove duplicate rows based on a subset of columns.
    """
    before = len(df)
    duplicate_rows = df[df.duplicated(subset=subset, keep="first")]

    if not duplicate_rows.empty:
        print(f"\n⚠️ Found {len(duplicate_rows)} duplicate rows in {name}:")
        print(duplicate_rows.to_string(index=False))  # print full duplicate rows
    else:
        print(f"\n✅ No duplicate rows found in {name}.")

    df = df.drop_duplicates(subset=subset, keep="first")
    after = len(df)
    print(f"🧹 Removed {before - after} duplicate rows from {name}.")
    return df


# ---- Preprocess Train ----
def preprocess_train(train_csv, output_csv):
    print(f"\n🚀 Preprocessing train dataset: {train_csv}")

    df = pd.read_csv(train_csv)
    print(f"📊 Train shape before preprocessing: {df.shape[0]} rows, {df.shape[1]} cols")

    # Clean
    df["nl"] = df["nl"].apply(clean_text)
    df["bash"] = df["bash"].apply(clean_text)

    # Check missing
    check_missing_values(df, "train")

    # Remove empty rows
    df = df[(df["nl"] != "") & (df["bash"] != "")]

    # Remove duplicates
    df = remove_duplicates(df, subset=["nl", "bash"], name="train")

    print(f"📊 Train shape after preprocessing: {df.shape[0]} rows, {df.shape[1]} cols")

    df.to_csv(output_csv, index=False)
    print(f"✅ Cleaned train dataset saved to {output_csv}")


# ---- Preprocess Test ----
def preprocess_test(test_csv, output_csv):
    print(f"\n🚀 Preprocessing test dataset: {test_csv}")

    df = pd.read_csv(test_csv)
    print(f"📊 Test shape before preprocessing: {df.shape[0]} rows, {df.shape[1]} cols")

    # Clean
    df["nl"] = df["nl"].apply(clean_text)
    df["bash"] = df["bash"].apply(clean_text)
    if "bash2" in df.columns:
        df["bash2"] = df["bash2"].apply(clean_text)

    # Check missing
    check_missing_values(df, "test")

    # Remove empty rows
    df = df[(df["nl"] != "") & (df["bash"] != "")]

    # Remove duplicates (based on nl + bash)
    df = remove_duplicates(df, subset=["nl", "bash"], name="test")

    print(f"📊 Test shape after preprocessing: {df.shape[0]} rows, {df.shape[1]} cols")

    df.to_csv(output_csv, index=False)
    print(f"✅ Cleaned test dataset saved to {output_csv}")


# ---- Run ----
preprocess_train("train.csv", "train_clean.csv")
preprocess_test("test.csv", "test_clean.csv")
