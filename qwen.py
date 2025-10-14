import json
import pickle
from sentence_transformers import SentenceTransformer
import faiss

# ===========================
# Configuration
# ===========================
INDEX_FILE = "bash_commands.index"
META_FILE = "metadata.pkl"
TOP_K = 5  # number of examples to retrieve

# ===========================
# Load FAISS index and metadata
# ===========================
index = faiss.read_index(INDEX_FILE)
with open(META_FILE, "rb") as f:
    data = pickle.load(f)

# Load embedder
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ===========================
# Check function
# ===========================
def check_command(query, top_k=TOP_K):
    query_vec = embedder.encode([query], convert_to_numpy=True)
    D, I = index.search(query_vec, top_k)
    
    print(f"\nTop {top_k} retrieved examples for query:\n'{query}'\n")
    for dist, idx in zip(D[0], I[0]):
        nl = data[idx]["nl"]
        bash = data[idx]["bash"]
        print(f"Distance: {dist:.4f}  |  NL: {nl}  |  Bash: {bash}")

# ===========================
# Example query
# ===========================
if __name__ == "__main__":
    query = "create a file named text.py"
    check_command(query)
