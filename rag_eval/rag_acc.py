import time
from sentence_transformers import SentenceTransformer
import faiss
import json
import random
import numpy as np
import os

# -------------------------
# 1. Load embedding model (offline/online fallback)
# -------------------------
print("Loading embedding model...")

local_model_path = "./models/sentence-transformers/all-MiniLM-L6-v2"  

if os.path.exists(local_model_path):
    print("Found local model, loading from:", local_model_path)
    model = SentenceTransformer(local_model_path)
else:
    print("No local model found. Downloading (requires internet)...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    # Save it for future offline runs
    os.makedirs(local_model_path, exist_ok=True)
    model.save(local_model_path)

print("✅ Model loaded.\n")

# -------------------------
# 2. Load and prepare train.jsonl (knowledge base)
# -------------------------
print("Loading train.jsonl (knowledge base)...")
train_data = []
with open("train.jsonl", "r") as f:
    for line in f:
        train_data.append(json.loads(line))

print(f"✅ Loaded {len(train_data)} training examples.\n")

train_instructions = [item["instruction"] for item in train_data]
train_responses = [item["response"] for item in train_data]

# to measure total time taken
total_start_time = time.time()

# -------------------------
# 3. Load or create embeddings and FAISS index
# -------------------------
CACHE_DIR = "./cache"
os.makedirs(CACHE_DIR, exist_ok=True)  # Create cache directory if it doesn't exist

TRAIN_EMB_PATH = os.path.join(CACHE_DIR, "train_embs.npz")
FAISS_INDEX_PATH = os.path.join(CACHE_DIR, "faiss_index.bin")

embeddings_exist = os.path.exists(TRAIN_EMB_PATH)
index_exist = os.path.exists(FAISS_INDEX_PATH)

if embeddings_exist and index_exist:
    print("Loading cached embeddings and FAISS index...")
    train_embeddings = np.load(TRAIN_EMB_PATH)["embs"]
    index = faiss.read_index(FAISS_INDEX_PATH)
    print(f"✅ Loaded embeddings with shape {train_embeddings.shape}")
    print(f"✅ Loaded FAISS index with {index.ntotal} vectors\n")
    # Set dummy timing values for reporting
    emb_start = emb_end = faiss_start = faiss_end = 0.0
else:
    print("Creating embeddings for training instructions...")
    emb_start = time.time()
    train_embeddings = model.encode(
        train_instructions,
        convert_to_numpy=True
    )
    emb_end = time.time()
    print(f"✅ Created embeddings with shape {train_embeddings.shape}")
    print(f"⏱ Embedding creation took {emb_end - emb_start:.2f} seconds\n")

    # Save embeddings
    np.savez_compressed(TRAIN_EMB_PATH, embs=train_embeddings)
    print(f"✅ Embeddings saved to {TRAIN_EMB_PATH}")

    # Build and save FAISS index
    dimension = train_embeddings.shape[1]
    print(f"Building FAISS index with dimension {dimension}...")
    faiss_start = time.time()
    index = faiss.IndexFlatL2(dimension)
    index.add(train_embeddings)
    faiss.write_index(index, FAISS_INDEX_PATH)
    faiss_end = time.time()
    print(f"✅ Added {index.ntotal} vectors to FAISS index.")
    print(f"✅ FAISS index saved to {FAISS_INDEX_PATH}")
    print(f"⏱ FAISS index build took {faiss_end - faiss_start:.2f} seconds\n")


# -------------------------
# 5. Define retrieval function
# -------------------------
def retrieve(query, top_k=3):
    """Retrieve top_k similar instructions and responses from FAISS"""
    query_vec = model.encode([query], convert_to_numpy=True)

    D, I = index.search(query_vec, top_k)

    results = []
    for rank, i in enumerate(I[0]):
        results.append({
            "instruction": train_instructions[i],
            "response": train_responses[i],
            "distance": float(D[0][rank])
        })
    return results

# -------------------------
# 6. Helper: cosine similarity
# -------------------------
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def best_match(true_responses, candidate_response, threshold=0.85):
    """
    Compare candidate_response with all possible true_responses.
    Return (match_found, best_similarity_score).
    """
    best_sim = 0.0
    match_found = False
    for resp in true_responses:
        emb1 = model.encode(resp, convert_to_numpy=True)
        emb2 = model.encode(candidate_response, convert_to_numpy=True)
        sim = cosine_sim(emb1, emb2)
        if sim > best_sim:
            best_sim = sim
        if sim >= threshold:
            match_found = True
            break
    return match_found, best_sim

# -------------------------
# 7. Load test.jsonl
# -------------------------
print("Loading test.jsonl...")
test_data = []
with open("test.jsonl", "r") as f:
    for line in f:
        test_data.append(json.loads(line))

print(f"✅ Loaded {len(test_data)} test examples.\n")

# -------------------------
# 8. Try with one random test query
# -------------------------
sample = random.choice(test_data)
query = sample["instruction"]

print("===== Test Example =====")
print("User Query:", query)
print("True Response (gold):", sample["responses"])
print("Difficulty:", sample.get("difficulty", "N/A"))

retrieved = retrieve(query, top_k=3)

print("\nTop Retrieved Knowledge:")
for r in retrieved:
    equal, sim = best_match(sample["responses"], r["response"])
    print(f"- Instruction: {r['instruction']}")
    print(f"  Response: {r['response']}")
    print(f"  Distance: {r['distance']:.4f}, Similarity: {sim:.4f}, Match: {equal}")
print("========================\n")

# -------------------------
# 9. Evaluate accuracy over all test examples
# -------------------------
print("Evaluating retrieval accuracy with semantic similarity...")
eval_start = time.time()

correct = 0
total = len(test_data)
difficulty_stats = {}

for item in test_data:
    query = item["instruction"]
    true_response = item["responses"]
    difficulty = item.get("difficulty", "N/A")

    retrieved = retrieve(query, top_k=3)

    # Check if any retrieved response is semantically equal
    match_found = False
    for r in retrieved:
        equal, sim = best_match(true_response, r["response"])
        if equal:
            match_found = True
            break

    if match_found:
        correct += 1
        result = "✅ Correct"
    else:
        result = "❌ Wrong"

    # Difficulty-wise stats
    if difficulty not in difficulty_stats:
        difficulty_stats[difficulty] = {"correct": 0, "total": 0}
    difficulty_stats[difficulty]["total"] += 1
    if match_found:
        difficulty_stats[difficulty]["correct"] += 1

    # Print only first 5 evaluations for clarity
    if total <= 5 or correct <= 5:
        print(f"Query: {query}")
        print(f"Expected: {true_response}")
        print(f"Retrieved: {[r['response'] for r in retrieved]}")
        print(f"Result: {result}\n")

eval_end = time.time()

# Final accuracy
accuracy = correct / total * 100
print(f"\n✅ Overall Retrieval Accuracy (semantic): {accuracy:.2f}%")

# Difficulty breakdown
print("\nAccuracy by difficulty:")
for diff, stats in difficulty_stats.items():
    acc = stats["correct"] / stats["total"] * 100
    print(f"- Difficulty {diff}: {acc:.2f}% ({stats['correct']}/{stats['total']})")

# Timings
print(f"\n⏱Embedding creation: {emb_end - emb_start:.2f} seconds")
print(f"⏱ FAISS index build: {faiss_end - faiss_start:.2f} seconds")
print(f"⏱ Evaluation time: {eval_end - eval_start:.2f} seconds")

# Total time taken
total_end_time = time.time()
elapsed = total_end_time - total_start_time
print(f"⏱ Total time (embedding → evaluation): {elapsed:.2f} seconds")