import json
import time
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os

DATA_FILE = "../../train.jsonl"
INDEX_FILE = "../bash_commands_l2_e5.bin"
META_FILE = "../metadata_l2_e5.npz"

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
    print("Building FAISS index with L2 distance from scratch using e5-base-v2...")
    texts = [item["instruction"] for item in data]

    # Embeddings
    embedder = SentenceTransformer("intfloat/e5-base-v2")
    print(f"Encoding {len(texts)} texts with e5-base-v2...")
    # E5 models require "passage: " prefix for documents
    texts_with_prefix = [f"passage: {text}" for text in texts]
    embeddings = embedder.encode(texts_with_prefix, convert_to_numpy=True, show_progress_bar=True)

    # Build FAISS index
    print("Building and indexing...")
    build_start = time.time()
    d = embeddings.shape[1]
    print(f"Embedding dimension: {d}")
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    build_time = time.time() - build_start

    # Save index and metadata
    print("Saving index and metadata...")
    faiss.write_index(index, INDEX_FILE)
    np.savez_compressed(META_FILE, data=np.array(data, dtype=object))

    print(f"✓ Indexed {len(texts)} commands into FAISS (L2 Distance)")
    print(f"✓ Index building time: {build_time:.4f} seconds")
    print(f"✓ Files saved: {INDEX_FILE}, {META_FILE}")

# Example retrieval function
def retrieve(query, k=5):
    query_vec = embedder.encode([f"query: {query}"], convert_to_numpy=True)
    D, I = index.search(query_vec, k)
    results = [data[idx] for idx in I[0]]
    return results, D[0]

# Example usage
if __name__ == "__main__":
    embedder = SentenceTransformer("intfloat/e5-base-v2")
    query = "show free space on all filesystems"
    results, distances = retrieve(query)
    print("\nExample retrieval results:")
    for i, r in enumerate(results):
        print(f"{i+1}. [Distance: {distances[i]:.4f}] {r['instruction']} -> {r['response']}")
