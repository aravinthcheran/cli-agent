from datasets import load_dataset

#load train data
# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("westenfelder/NL2SH-ALFA", "train")

print(ds)
print(ds["train"][0])

ds["train"].to_csv("train.csv")
print("Dataset saved as train.csv")


# load test data
ds = load_dataset("westenfelder/NL2SH-ALFA", "test")

print(ds)
print(ds["train"][0])

ds["train"].to_csv("test.csv")
print("Dataset saved as test.csv")
