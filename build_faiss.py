import json
import pickle
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

DATA_FILE = "NL2SH-ALFA_train_simple.json"
INDEX_FILE = "bash_commands.index"
META_FILE = "metadata.pkl"

# Load dataset
with open(DATA_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# Check if index already exists
if os.path.exists(INDEX_FILE) and os.path.exists(META_FILE):
    print("Loading FAISS index and metadata from disk...")
    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, "rb") as f:
        data = pickle.load(f)
else:
    print("Building FAISS index from scratch...")
    texts = [item["nl"] for item in data]

    # Embeddings
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = embedder.encode(texts, convert_to_numpy=True)

    # Build FAISS index
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)

    # Save index and metadata
    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "wb") as f:
        pickle.dump(data, f)

    print(f"Indexed {len(texts)} commands into FAISS and saved to disk!")

# Example retrieval function
def retrieve(query, k=5):
    query_vec = embedder.encode([query], convert_to_numpy=True)
    D, I = index.search(query_vec, k)
    results = [data[idx] for idx in I[0]]
    return results

# Example usage
query = "show free space on all filesystems"
results = retrieve(query)
for r in results:
    print(r)