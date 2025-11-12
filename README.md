# Linux Bash CLI Agent 🤖

A powerful RAG (Retrieval-Augmented Generation) powered CLI agent that converts natural language instructions into executable bash commands. The agent uses FAISS vector search for semantic retrieval and Ollama for intelligent command generation.

## ✨ Features

- 🧠 **Natural Language Processing**: Convert plain English to bash commands
- 🔍 **RAG-Powered Retrieval**: Uses FAISS vector database for semantic command search
- 🛡️ **Safety Checks**: Built-in dangerous command detection and user confirmation
- 🔄 **Auto-Retry**: Intelligent retry mechanism with error context for failed commands
- 🎯 **Two Operating Modes**: Normal (clean output) and Debug (detailed internal workings)
- 📊 **Comprehensive Evaluation**: Built-in accuracy evaluation with multiple embedding models
- 🎨 **Rich CLI Interface**: Beautiful terminal UI with color-coded output

## 🏗️ Architecture

The agent uses a modular architecture combining:
- **Embedding Models**: sentence-transformers (all-MiniLM-L6-v2, e5-base-v2)
- **Vector Database**: FAISS with L2 and Cosine similarity indexes
- **LLM**: Ollama (configurable model, default: mistral)
- **Dataset**: NL2SH-ALFA (Natural Language to Shell commands)

## 📋 Requirements

- Python 3.8+
- [Ollama](https://ollama.ai/) running locally
- Linux/Unix-based operating system (recommended)

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/aravinthcheran/cli-agent.git
cd cli-agent
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
- faiss-cpu
- numpy
- sentence-transformers
- transformers
- ollama
- pandas
- requests
- accelerate
- rich
- matplotlib

### 3. Install and Run Ollama

Download and install Ollama from [ollama.ai](https://ollama.ai/)

```bash
# Start Ollama service
ollama serve

# In another terminal, pull the model
ollama pull mistral
```

### 4. Prepare the Dataset

#### Download and Preprocess

```bash
# Step 1: Download NL2SH-ALFA dataset from HuggingFace
python preprocess/load_dataset.py

# Step 2: Clean and preprocess the data
python preprocess/preprocess.py

# Step 3: Convert to JSONL format
python preprocess/dataset_conversion.py
```

### 5. Build FAISS Indexes

```bash
# Build L2 distance index
python index_build/build_faiss.py

# Build Cosine similarity index
python index_build/build_faiss_cosine.py
```

The indexes will be cached in the `cache/` directory for fast startup.

## 🎮 Usage

### Basic Usage

Run the modular CLI agent:

```bash
python cli_agent_modular.py
```

Or use the standalone version:

```bash
python cli_agent.py
```

### Interactive Session Example

```
=== Linux Bash CLI Agent (RAG + Ollama) ===
Platform: Linux

Select mode:
  1. Normal mode - Clean output with minimal details
  2. Debug mode - Show all internal workings

Choose mode [1]: 1

✓ Normal mode active

Type a natural language instruction (or 'exit' to quit)
Commands: 'exit'/'quit' to exit, 'toggle debug' to switch modes

> list all files with details
Generated command(s):
ls -la

Do you want to execute this command? [y/n]: y

✓ Command executed successfully

> find large files over 100MB

Generated command(s):
find . -type f -size +100M

Do you want to execute this command? [y/n]: y
```

### Debug Mode

Toggle debug mode to see internal workings:

```bash
> toggle debug
🔧 Debug mode ENABLED - Showing all internal workings

> compress a folder
🔍 Encoding query for FAISS retrieval: 'compress a folder'
🔍 Searching FAISS index for top 5 results...
✓ Retrieved 5 examples from knowledge base

📚 Retrieved RAG Examples:
[1] Instruction: compress a directory
    Response: tar -czf archive.tar.gz directory/

Generated command(s):
tar -czf folder.tar.gz folder/
```

## 📁 Project Structure

```
cli-agent/
├── cli_agent.py              # Standalone CLI agent (legacy)
├── cli_agent_modular.py      # Modular CLI agent (main entry point)
├── requirements.txt          # Python dependencies
├── datasets/                 # Dataset files (JSONL, CSV)
│   ├── train.jsonl          # Training data
│   ├── test.jsonl           # Test data
│   └── *.csv                # Raw CSV files
├── src/                     # Modular source code
│   ├── __init__.py
│   ├── config.py            # Configuration constants
│   ├── rag.py               # RAG retrieval logic
│   ├── utils.py             # Utility functions
│   ├── clarification.py     # Query clarification
│   ├── command_processor.py # Command generation and execution
│   └── ollama_client.py     # Ollama API client
├── index_build/             # FAISS index building scripts
│   ├── build_faiss.py       # Build L2 index
│   └── build_faiss_cosine.py # Build Cosine index
├── preprocess/              # Data preprocessing pipeline
│   ├── load_dataset.py      # Download dataset
│   ├── preprocess.py        # Clean data
│   └── dataset_conversion.py # Convert to JSONL
├── e5_evaluation/           # E5-base-v2 model evaluation
│   ├── index_build/         # E5 index builders
│   └── evaluate_rag_accuracy_e5.py
├── metrics/                 # Evaluation metrics and graphs
│   ├── compare_retrieval_methods.py
│   ├── build_time_comparison.py
│   └── *.png                # Generated graphs
├── rag_eval/                # RAG accuracy evaluation
│   └── rag_acc.py
├── evaluate_rag_accuracy.py # Main evaluation script
└── cache/                   # Cached FAISS indexes (generated)
    ├── bash_commands_l2.bin
    └── metadata_l2.npz
```

## ⚙️ Configuration

Edit `src/config.py` to customize:

```python
# FAISS & RAG Configuration
INDEX_FILE = "cache/bash_commands_l2.bin"
META_FILE = "cache/metadata_l2.npz"
TOP_K = 5  # Number of examples to retrieve

# Ollama Configuration
OLLAMA_MODEL = "mistral"  # Can use: llama2, codellama, etc.
OLLAMA_URL = "http://localhost:11434/api/generate"

# Execution Configuration
MAX_RETRIES = 2
COMMAND_TIMEOUT = 260
```

### Environment Variables

```bash
# Set Ollama model
export OLLAMA_MODEL=mistral

# Or use a different model
export OLLAMA_MODEL=llama2
```

## 🔒 Safety Features

The agent includes built-in dangerous command detection:

- Recursive force deletion (`rm -rf /`, `rm -rf *`)
- Direct disk operations (`dd`, `mkfs`)
- System modifications (`chmod -R 777 /`, `chown -R`)
- Download and execute (`curl | bash`)
- System control (`shutdown`, `reboot`)
- Fork bombs and destructive patterns

When a dangerous command is detected:
1. User is warned with specific risk description
2. Confirmation is required before execution
3. Option to abort provided

## 📊 Model Evaluation

### Running Accuracy Evaluation

```bash
# Evaluate with all-MiniLM-L6-v2 (default)
python evaluate_rag_accuracy.py

# Evaluate with e5-base-v2
cd e5_evaluation
python index_build/build_faiss_l2_e5.py
python index_build/build_faiss_cosine_e5.py
python evaluate_rag_accuracy_e5.py
```

### Compare Retrieval Methods

```bash
# Compare L2 vs Cosine similarity
python metrics/compare_retrieval_methods.py

# Compare index build times
python metrics/build_time_comparison.py

# Generate graphs from values
python metrics/generate_graph_from_values.py
```

### Evaluation Metrics

The evaluation compares:
- **L2 Distance**: Euclidean distance for similarity
- **Cosine Similarity**: Inner product of normalized vectors
- **Semantic Matching**: Cosine similarity with 0.85 threshold

Results are categorized by difficulty level (1-3) and visualized with matplotlib.

## 🎯 Dataset Information

**Dataset**: [NL2SH-ALFA](https://huggingface.co/datasets/westenfelder/NL2SH-ALFA)

The NL2SH-ALFA dataset contains:
- Natural language instructions in English
- Corresponding bash command(s)
- Difficulty levels (1-3)
- Multiple valid responses per instruction

### Data Format

**Training JSONL:**
```json
{
  "instruction": "list all files in current directory",
  "response": "ls -la"
}
```

**Test JSONL:**
```json
{
  "instruction": "show disk usage",
  "responses": ["df -h", "du -sh *"],
  "difficulty": 2
}
```

## 🔍 How It Works

1. **User Input**: Natural language instruction (e.g., "find large files")
2. **Embedding**: Convert query to vector using sentence-transformers
3. **RAG Retrieval**: Search FAISS index for similar historical commands
4. **LLM Generation**: Ollama generates bash command using:
   - Retrieved examples (context)
   - User query
   - Error feedback (if retry)
5. **Safety Check**: Scan for dangerous patterns
6. **User Confirmation**: Prompt for execution approval
7. **Execution**: Run command and capture output
8. **Error Handling**: Auto-retry with error context if failed

## 🛠️ Advanced Usage

### Using Different Embedding Models

The project supports multiple embedding models:

**Default: all-MiniLM-L6-v2**
- Dimensions: 384
- Size: ~80MB
- Speed: Fast
- Quality: Good

**Alternative: e5-base-v2**
- Dimensions: 768
- Size: ~400MB
- Speed: Slower
- Quality: Better

To use e5-base-v2, navigate to `e5_evaluation/` and follow the README instructions.

### Customizing Prompts

Edit `src/command_processor.py` to customize the LLM prompt:

```python
def generate_bash(query, error_context=None, show_rag_debug=False):
    # Modify the prompt template here
    prompt = f"""You are a bash expert. Generate a bash command for: {query}
    
    Examples from knowledge base:
    {rag_context}
    
    Return only the command, no explanation.
    """
```

### Running Specific Tests

```bash
# Test Ollama connection
python gemini_tester.py

# Test RAG accuracy on test set
python rag_eval/rag_acc.py
```

## 🐛 Troubleshooting

### Ollama Connection Error

```bash
# Check if Ollama is running
curl http://localhost:11434/api/generate

# Start Ollama if not running
ollama serve
```

### FAISS Index Not Found

```bash
# Rebuild indexes
python index_build/build_faiss.py
python index_build/build_faiss_cosine.py
```

### Dimension Mismatch Error

This happens when switching embedding models. Delete cache and rebuild:

```bash
rm -rf cache/*.bin cache/*.npz
python index_build/build_faiss.py
```

### Model Download Issues

Ensure internet connection and Hugging Face access:

```bash
# Login to Hugging Face (if needed)
huggingface-cli login

# Model will be cached in ~/.cache/huggingface/
```

## 📈 Performance

### Index Build Times (all-MiniLM-L6-v2)
- L2 Index: ~15-30 seconds for 50K commands
- Cosine Index: ~15-30 seconds for 50K commands

### Retrieval Speed
- Query encoding: ~10-50ms
- FAISS search: <5ms
- Total RAG retrieval: ~50-100ms

### Accuracy (Test Set)
- Overall: 70-85% (depending on difficulty)
- Easy (Level 1): 85-90%
- Medium (Level 2): 70-80%
- Hard (Level 3): 60-75%

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

1. **Model Enhancement**
   - Add support for more embedding models
   - Integrate fine-tuned models for bash commands
   - Experiment with larger LLMs

2. **Feature Additions**
   - Command history and learning
   - User preference learning
   - Multi-step command sequences
   - Command explanation mode

3. **Safety Improvements**
   - More comprehensive dangerous pattern detection
   - Sandbox execution option
   - Undo/rollback capabilities

4. **Documentation**
   - More usage examples
   - Video tutorials
   - API documentation

## 📄 License

This project is open source. Please check the repository for license information.

## 🙏 Acknowledgments

- **Dataset**: [NL2SH-ALFA](https://huggingface.co/datasets/westenfelder/NL2SH-ALFA) by westenfelder
- **Embedding Models**: 
  - [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) by sentence-transformers
  - [e5-base-v2](https://huggingface.co/intfloat/e5-base-v2) by Microsoft
- **LLM**: [Ollama](https://ollama.ai/) for local LLM inference
- **Vector Search**: [FAISS](https://github.com/facebookresearch/faiss) by Meta AI

## 📞 Contact

For issues, questions, or suggestions, please open an issue on the [GitHub repository](https://github.com/aravinthcheran/cli-agent).

---

**Note**: This tool generates and executes bash commands. Always review commands before execution, especially on production systems. Use with caution and at your own risk.
