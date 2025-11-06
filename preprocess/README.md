# Dataset Preprocessing Pipeline

This directory contains scripts for loading, preprocessing, and converting the NL2SH-ALFA dataset.

## Directory Structure

```
preprocess/
├── README.md                  # This file
├── load_dataset.py           # Downloads dataset from HuggingFace
├── preprocess.py             # Cleans and preprocesses the data
└── dataset_conversion.py     # Converts CSV to JSONL format
```

## Workflow

The scripts should be run in the following order:

### 1. Load Dataset (`load_dataset.py`)
Downloads the NL2SH-ALFA dataset from HuggingFace and saves it to the `../datasets/` directory.

**Input:** HuggingFace dataset `westenfelder/NL2SH-ALFA`

**Output:**
- `../datasets/train.csv` - Raw training data
- `../datasets/test.csv` - Raw test data

**Usage:**
```bash
python preprocess/load_dataset.py
```

**Note:** You may need to login to HuggingFace first:
```bash
huggingface-cli login
```

---

### 2. Preprocess Data (`preprocess.py`)
Cleans and preprocesses the raw CSV files by:
- Stripping whitespace
- Removing empty rows
- Checking for missing values
- Removing duplicates
- Validating data quality

**Input:**
- `../datasets/train.csv`
- `../datasets/test.csv`

**Output:**
- `../datasets/train_clean.csv` - Cleaned training data
- `../datasets/test_clean.csv` - Cleaned test data

**Usage:**
```bash
python preprocess/preprocess.py
```

---

### 3. Convert to JSONL (`dataset_conversion.py`)
Converts cleaned CSV files to JSONL format for model training.

**Input:**
- `../datasets/train_clean.csv`
- `../datasets/test_clean.csv`

**Output:**
- `../datasets/train.jsonl` - Training data in JSONL format
  ```json
  {"instruction": "...", "response": "..."}
  ```
- `../datasets/test.jsonl` - Test data in JSONL format
  ```json
  {"instruction": "...", "responses": ["...", "..."], "difficulty": 1}
  ```

**Usage:**
```bash
python preprocess/dataset_conversion.py
```

---

## Complete Pipeline

Run all steps in sequence:

```bash
# Step 1: Download data
python preprocess/load_dataset.py

# Step 2: Clean and preprocess
python preprocess/preprocess.py

# Step 3: Convert to JSONL
python preprocess/dataset_conversion.py
```

---

## Output Location

All processed datasets are saved in the `datasets/` directory at the project root:

```
datasets/
├── train.csv           # Raw training data
├── test.csv            # Raw test data
├── train_clean.csv     # Cleaned training data
├── test_clean.csv      # Cleaned test data
├── train.jsonl         # Training data in JSONL format
└── test.jsonl          # Test data in JSONL format
```

---

## Data Format

### Train JSONL Format
```json
{
  "instruction": "list all files in current directory",
  "response": "ls -la"
}
```

### Test JSONL Format
```json
{
  "instruction": "show disk usage",
  "responses": ["df -h", "du -sh *"],
  "difficulty": 2
}
```

---

## Requirements

Make sure you have the required packages installed:
```bash
pip install pandas datasets
```
