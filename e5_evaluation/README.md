# E5-Base-V2 Evaluation

This directory contains scripts for evaluating RAG accuracy using the **e5-base-v2** embedding model from Microsoft's E5 family. This is kept separate from the original codebase to avoid conflicts.

## 📁 Directory Structure

```
e5_evaluation/
├── index_build/
│   ├── build_faiss_l2_e5.py       # Build L2 distance index with e5-base-v2
│   └── build_faiss_cosine_e5.py   # Build cosine similarity index with e5-base-v2
├── evaluate_rag_accuracy_e5.py    # Evaluation script using e5-base-v2
├── bash_commands_l2_e5.bin        # L2 FAISS index (generated)
├── metadata_l2_e5.npz             # L2 metadata (generated)
├── bash_commands_cosine_e5.bin    # Cosine FAISS index (generated)
├── metadata_cosine_e5.npz         # Cosine metadata (generated)
└── rag_accuracy_comparison_e5.png # Results graph (generated)
```

## 🚀 Usage

### Step 1: Build FAISS Indexes

First, navigate to the e5_evaluation directory and build the indexes:

```bash
cd e5_evaluation

# Build L2 distance index (will download e5-base-v2 on first run)
python index_build/build_faiss_l2_e5.py

# Build cosine similarity index
python index_build/build_faiss_cosine_e5.py
```

**Note:** The first run will download the e5-base-v2 model (~400MB) from Hugging Face.

### Step 2: Run Evaluation

Once both indexes are built, run the evaluation:

```bash
python evaluate_rag_accuracy_e5.py
```

This will:
- Load both L2 and Cosine similarity indexes
- Test on 300 test cases
- Generate accuracy metrics by difficulty level
- Create a comparison graph (`rag_accuracy_comparison_e5.png`)

## 📊 Model Information

**Model:** intfloat/e5-base-v2
- **Embedding Dimension:** 768 (vs 384 for all-MiniLM-L6-v2)
- **Parameters:** ~110M
- **Performance:** Higher quality embeddings, especially for semantic search

### E5 Model Prefixes

E5 models require specific prefixes:
- **Queries:** `"query: <your query>"`
- **Passages/Documents:** `"passage: <your text>"`

These prefixes are automatically applied in the scripts.

## 🔄 Comparison with Original

| Aspect | Original (all-MiniLM-L6-v2) | E5 Evaluation |
|--------|----------------------------|---------------|
| Model | all-MiniLM-L6-v2 | intfloat/e5-base-v2 |
| Dimensions | 384 | 768 |
| Size | ~80MB | ~400MB |
| Speed | Faster | Slower |
| Quality | Good | Better |
| Directory | Root | e5_evaluation/ |

## 📊 Evaluation Methodology

The evaluation uses **cosine similarity** as the comparison metric for both retrieval methods:

1. **Retrieval Methods Being Compared:**
   - **L2 Distance:** Uses Euclidean distance for finding similar commands
   - **Cosine Similarity:** Uses inner product of normalized vectors

2. **Matching Criterion (Same for Both):**
   - Commands are compared using **cosine similarity** between embeddings
   - Threshold: 0.85 (configurable via `COSINE_SIMILARITY_THRESHOLD`)
   - A retrieved command is considered correct if its cosine similarity with any expected command ≥ threshold

This approach ensures fair comparison since both methods use the same matching metric (cosine similarity), differing only in their retrieval strategy.

## 📈 Expected Results

E5-base-v2 typically provides:
- Better semantic understanding compared to smaller models
- Improved accuracy on complex queries
- Better handling of paraphrases
- Higher overall retrieval accuracy due to larger embedding dimension (768 vs 384)

## 🔧 Troubleshooting

### Dimension Mismatch Error
If you see `AssertionError: d == self.d`, it means the index was built with a different model. Delete the index files and rebuild:

```bash
rm bash_commands_*.bin metadata_*.npz
python index_build/build_faiss_l2_e5.py
python index_build/build_faiss_cosine_e5.py
```

### Model Download Issues
If model download fails, ensure you have internet connection and try again. The model will be cached in `~/.cache/huggingface/`.

## 📝 Notes

- All file paths are configured to avoid conflicts with the original evaluation code
- Index files use `_e5` suffix to distinguish them
- Results graph is saved as `rag_accuracy_comparison_e5.png`
- Original code remains unchanged and functional
