#!/usr/bin/env python3

import pickle
from sentence_transformers import SentenceTransformer
import faiss

# Load components
INDEX_FILE = "bash_commands.index"
META_FILE = "metadata.pkl"
TOP_K = 5

index = faiss.read_index(INDEX_FILE)
with open(META_FILE, "rb") as f:
    data = pickle.load(f)

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def retrieve(query, top_k=TOP_K):
    """Retrieve top-k similar examples from FAISS"""
    vec = embedder.encode([query], convert_to_numpy=True)
    D, I = index.search(vec, top_k)
    neighbors = []
    for i, idx in enumerate(I[0]):
        neighbors.append({
            'data': data[idx],
            'distance': D[0][i]
        })
    return neighbors

# Test the specific problematic query
query = "list all files in my parent directory"
print(f"Query: '{query}'")
print("Retrieved examples:")
print("=" * 50)

neighbors = retrieve(query)
for i, item in enumerate(neighbors, 1):
    print(f"{i}. Distance: {item['distance']:.3f}")
    print(f"   NL: {item['data']['nl']}")
    print(f"   Bash: {item['data']['bash']}")
    print()