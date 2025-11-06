"""RAG (Retrieval-Augmented Generation) module for the CLI agent."""
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rich.console import Console
from src.config import INDEX_FILE, META_FILE, TOP_K, DEBUG_MODE
from src.utils import debug_print

console = Console()

# Global variables for RAG resources
_index = None
_data = None
_embedder = None


def initialize_rag():
    """Initialize FAISS index, metadata, and sentence transformer."""
    global _index, _data, _embedder
    
    console.print("[bold cyan]Loading FAISS index and metadata...[/bold cyan]")
    _index = faiss.read_index(INDEX_FILE)
    
    with open(META_FILE, "rb") as f:
        meta_data = np.load(META_FILE, allow_pickle=True)
        _data = meta_data['data'].tolist()
    
    _embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    console.print("[bold green]✓ RAG resources loaded successfully[/bold green]\n")


def retrieve(query, top_k=TOP_K):
    """Retrieve top-k similar examples from FAISS."""
    if _index is None or _data is None or _embedder is None:
        raise RuntimeError("RAG not initialized. Call initialize_rag() first.")
    
    debug_print(f"🔍 Encoding query for FAISS retrieval: '{query[:50]}...'")
    vec = _embedder.encode([query], convert_to_numpy=True)
    debug_print(f"🔍 Searching FAISS index for top {top_k} results...")
    _, I = _index.search(vec, top_k)
    results = [_data[idx] for idx in I[0]]
    debug_print(f"✓ Retrieved {len(results)} examples from knowledge base", "green")
    return results


def display_rag_examples(examples, max_examples=3):
    """Display RAG retrieval examples for debugging."""
    console.print("\n[bold blue]📚 Retrieved RAG Examples:[/bold blue]")
    if examples and len(examples) > 0:
        for i, item in enumerate(examples[:max_examples], 1):
            console.print(f"\n[yellow]Example {i}:[/yellow]")
            console.print(f"  [dim]Query:[/dim] {item['instruction'][:100]}...")
            console.print(f"  [dim]Command:[/dim] {item['response'][:100]}...")
    else:
        console.print("[red]⚠️ No RAG examples retrieved[/red]")
