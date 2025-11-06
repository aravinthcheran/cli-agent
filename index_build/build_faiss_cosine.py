import json
import time
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os

# Get the project root directory (parent of index_build)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")

# Create cache directory if it doesn't exist
os.makedirs(CACHE_DIR, exist_ok=True)

DATA_FILE = os.path.join(PROJECT_ROOT, "datasets", "train.jsonl")
INDEX_FILE = os.path.join(CACHE_DIR, "bash_commands_cosine.bin")
META_FILE = os.path.join(CACHE_DIR, "metadata_cosine.npz")

# Load dataset (JSONL format - one JSON object per line)
with open(DATA_FILE, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

# Check if index already exists
if os.path.exists(INDEX_FILE) and os.path.exists(META_FILE):
    print("Loading FAISS index and metadata from disk...")
    load_start = time.time()
    index = faiss.read_index(INDEX_FILE)
    meta_data = np.load(META_FILE, allow_pickle=True)
    data = meta_data['data'].tolist()
    load_time = time.time() - load_start
    print(f"✓ Loaded in {load_time:.4f} seconds")
else:
    print("Building FAISS index with cosine similarity from scratch...")
    texts = [item["instruction"] for item in data]

    # Embeddings
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print(f"Encoding {len(texts)} texts...")
    embeddings = embedder.encode(texts, convert_to_numpy=True)

    # Normalize embeddings for cosine similarity
    print("Normalizing embeddings...")
    faiss.normalize_L2(embeddings)

    # Build FAISS index
    print("Building and indexing...")
    build_start = time.time()
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)  # Inner Product for cosine similarity
    index.add(embeddings)
    build_time = time.time() - build_start

    # Save index and metadata
    print("Saving index and metadata...")
    faiss.write_index(index, INDEX_FILE)
    np.savez_compressed(META_FILE, data=np.array(data, dtype=object))

    print(f"✓ Indexed {len(texts)} commands into FAISS (Cosine Similarity)")
    print(f"✓ Index building time: {build_time:.4f} seconds")
    print(f"✓ Files saved: {INDEX_FILE}, {META_FILE}")

# Example retrieval function
def retrieve(query, k=5):
    query_vec = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_vec)  # Normalize query vector
    D, I = index.search(query_vec, k)
    results = [data[idx] for idx in I[0]]
    return results

# Example usage
if __name__ == "__main__":
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    query = "show free space on all filesystems"
    results = retrieve(query)
    print("\nExample retrieval results:")
    for r in results:
        print(r)
