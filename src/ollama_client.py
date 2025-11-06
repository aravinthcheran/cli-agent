"""Ollama API client for LLM interactions."""
import requests
from rich.console import Console
from src.config import OLLAMA_MODEL, OLLAMA_URL, DEBUG_MODE
from src.utils import debug_print

console = Console()


def ollama_generate(prompt: str, max_tokens: int = 300, temperature: float = 0.4):
    """Send prompt to local Ollama model and get response."""
    debug_print(f"🔧 Calling Ollama API: model={OLLAMA_MODEL}, max_tokens={max_tokens}, temp={temperature}")
    
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens}
    }
    
    if DEBUG_MODE:
        debug_print("📝 Full prompt sent to Ollama:", "cyan")
        debug_print(f"{prompt}\n", "dim")
    
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json().get("response", "").strip()
        debug_print(f"✓ Ollama response received ({len(result)} chars)", "green")
        return result
    except Exception as e:
        console.print(f"[bold red]Error calling Ollama: {e}[/bold red]")
        return ""


def evaluate_information_sufficiency(query_with_answers):
    """Use the model to evaluate if provided information is sufficient for command generation."""
    console.print("\n[dim]🤖 Evaluating if provided information is sufficient...[/dim]")
    
    evaluation_prompt = f"""Analyze this user request and determine if it has SUFFICIENT information for generating a precise bash command:

User Request: {query_with_answers}

Respond with ONLY:
- "SUFFICIENT" if all necessary information is provided
- "MISSING: [list what's missing]" if information is incomplete

Consider what's needed:
- For file operations: filename is required, but directory path is OPTIONAL (defaults to current directory)
- For searches: what to search for is required, where to search is OPTIONAL (defaults to current directory)
- For copies/moves: source + destination paths (can be relative to current directory)
- For permissions: target file (can be relative path) + permission level
- For compression: source files + destination filename (can be in current directory)

Important rules:
1. If NO directory is mentioned, assume current directory - this is SUFFICIENT
2. If a subdirectory is mentioned (e.g., "data/file.txt"), treat it as relative path - this is SUFFICIENT
3. Only mark as MISSING if critical information like filename itself is absent
4. Absolute paths are not required - relative paths and current directory defaults are acceptable

Be practical - if a command can be executed with the given info, mark it SUFFICIENT."""
    
    result = ollama_generate(evaluation_prompt, max_tokens=100, temperature=0.3)
    
    if "SUFFICIENT" in result.upper():
        console.print("[green]✓ Information is sufficient![/green]")
        return True, None
    else:
        # Extract missing information
        missing = result.replace("MISSING:", "").strip()
        console.print(f"[yellow]⚠️  Model indicates missing info: {missing}[/yellow]")
        return False, missing
