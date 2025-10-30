import numpy as np
import re
import subprocess
import platform
import requests
from sentence_transformers import SentenceTransformer
import faiss
from rich.console import Console
from rich.prompt import Prompt, Confirm

console = Console()

# ===============================
# CONFIGURATION
# ===============================
INDEX_FILE = "bash_commands_l2.bin"
META_FILE = "metadata_l2.npz"
TOP_K = 5
OLLAMA_MODEL = "mistral:instruct"
OLLAMA_URL = "http://localhost:11434/api/generate"

# Global debug mode flag
DEBUG_MODE = False

# Dangerous command patterns
DANGEROUS_PATTERNS = [
    (r'\brm\s+-rf\s+/', "Recursive force deletion from root"),
    (r'\brm\s+-rf\s+\*', "Recursive force deletion of all files"),
    (r'\bdd\s+if=/dev/(?:zero|random)\s+of=/dev/(?:sd|hd|nvme)', "Direct disk write operation"),
    (r':\(\)\{\s*:\|:&\s*\};:', "Fork bomb"),
    (r'\bmkfs\b', "Filesystem formatting"),
    (r'\bwipefs\b', "Wipe filesystem signatures"),
    (r'\b>\s*/dev/sd', "Direct write to disk device"),
    (r'\bchmod\s+-R\s+777\s+/', "Recursive permission change from root"),
    (r'\bchown\s+-R.*\s+/', "Recursive ownership change from root"),
    (r'\bcurl\s+.*\|\s*(?:bash|sh)', "Download and execute script"),
    (r'\bwget\s+.*\|\s*(?:bash|sh)', "Download and execute script"),
    (r'\bmv\s+/(?:bin|boot|etc|lib|sbin|usr)\b', "Moving critical system directory"),
    (r'\brm\s+(?:-rf\s+)?/(?:bin|boot|etc|lib|sbin|usr)\b', "Deleting critical system directory"),
    (r'\biptables\s+-F', "Flushing firewall rules"),
    (r'\bshutdown\b', "System shutdown"),
    (r'\breboot\b', "System reboot"),
    (r'\binit\s+0', "System halt"),
    (r'\binit\s+6', "System reboot"),
]

# ===============================
# LOAD RESOURCES
# ===============================
console.print("[bold cyan]Loading FAISS index and metadata...[/bold cyan]")
index = faiss.read_index(INDEX_FILE)
with open(META_FILE, "rb") as f:
    meta_data = np.load(META_FILE, allow_pickle=True)
    data = meta_data['data'].tolist()

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ===============================
# HELPER FUNCTIONS
# ===============================
def debug_print(message: str, style: str = "dim"):
    """Print debug messages only when DEBUG_MODE is enabled."""
    if DEBUG_MODE:
        console.print(f"[{style}]{message}[/{style}]")

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

def check_dangerous_commands(command_str):
    """Check if commands contain dangerous patterns."""
    dangers = []
    for pattern, description in DANGEROUS_PATTERNS:
        if re.search(pattern, command_str, re.IGNORECASE):
            dangers.append(description)
    return dangers

def sanitize_bash(output: str) -> str:
    """Clean model output to valid bash."""
    # Remove markdown code blocks
    output = re.sub(r"```[a-zA-Z0-9]*\n?", "", output)
    output = re.sub(r"```\n?", "", output)
    
    # Remove explanation patterns
    output = re.sub(r"(?i)^(here'?s?|this|the command|explanation).*?:\s*", "", output)
    
    lines = output.split('\n')
    command_lines = []
    in_heredoc = False
    heredoc_delimiter = None
    heredoc_content = []
    
    for line in lines:
        # Heredoc handling
        if not in_heredoc and '<<' in line:
            heredoc_match = re.search(r'<<\s*["\']?(\w+)["\']?', line)
            if heredoc_match:
                in_heredoc = True
                heredoc_delimiter = heredoc_match.group(1)
                command_lines.append(line.strip())
                heredoc_content = []
                continue
        
        if in_heredoc:
            if line.strip() == heredoc_delimiter:
                command_lines.extend(heredoc_content)
                command_lines.append(heredoc_delimiter)
                in_heredoc = False
                heredoc_delimiter = None
                heredoc_content = []
                continue
            else:
                if not line.strip().isdigit():
                    heredoc_content.append(line.rstrip())
                continue
        
        line = line.strip()
        
        # Skip empty lines, digits, explanations
        if not line or (line.isdigit() and len(line) < 3):
            continue
        
        if re.match(r'^[A-Z][a-z]+\s+.*[.!?]$', line):
            continue
        
        skip_words = ['the script', 'this will', 'note that', 'explanation', 
                      'department', 'looking for', 'you can', 'make sure', 'save the file']
        if any(word in line.lower() for word in skip_words):
            continue
        
        if line.startswith('#') and not line.startswith('#!'):
            continue
        
        command_lines.append(line)
    
    if in_heredoc:
        command_lines.extend(heredoc_content)
        command_lines.append(heredoc_delimiter)
    
    return '\n'.join(command_lines)

def retrieve(query, top_k=TOP_K):
    """Retrieve top-k similar examples from FAISS."""
    debug_print(f"🔍 Encoding query for FAISS retrieval: '{query[:50]}...'")
    vec = embedder.encode([query], convert_to_numpy=True)
    debug_print(f"🔍 Searching FAISS index for top {top_k} results...")
    _, I = index.search(vec, top_k)
    results = [data[idx] for idx in I[0]]
    debug_print(f"✓ Retrieved {len(results)} examples from knowledge base", "green")
    return results

def extract_filename(query):
    """Extract filename from query if present."""
    # Look for common filename patterns
    filename_pattern = r'(?:delete|remove|rm|del)\s+(?:the\s+)?(?:file\s+)?([a-zA-Z0-9_\-\.]+(?:\.[a-zA-Z0-9]+)?)'
    match = re.search(filename_pattern, query, re.IGNORECASE)
    if match:
        return match.group(1)
    return None

def evaluate_information_sufficiency(query_with_answers):
    """Use the model to evaluate if provided information is sufficient for command generation."""
    console.print("\n[dim]🤖 Evaluating if provided information is sufficient...[/dim]")
    
    evaluation_prompt = f"""Analyze this user request and determine if it has SUFFICIENT information for generating a precise bash command:

User Request: {query_with_answers}

Respond with ONLY:
- "SUFFICIENT" if all necessary information is provided
- "MISSING: [list what's missing]" if information is incomplete

Consider what's needed:
- For file operations: full file path or filename + directory
- For searches: what to search for + where to search
- For copies/moves: source + destination paths
- For permissions: target file + permission level
- For compression: source files + destination filename

Be strict - paths should be specific, not vague."""
    
    result = ollama_generate(evaluation_prompt, max_tokens=100, temperature=0.3)
    
    if "SUFFICIENT" in result.upper():
        console.print("[green]✓ Information is sufficient![/green]")
        return True, None
    else:
        # Extract missing information
        missing = result.replace("MISSING:", "").strip()
        console.print(f"[yellow]⚠️  Model indicates missing info: {missing}[/yellow]")
        return False, missing

def check_and_clarify_delete_command(query):
    """Smart clarification for delete - uses model to verify if info is sufficient."""
    if not re.search(r'\bdelete\b|\brm\b|\bremove\b', query, re.IGNORECASE):
        return query, False
    
    filename = extract_filename(query)
    
    if not filename:
        return query, False
    
    console.print(f"\n[bold yellow]ℹ️  Delete command detected[/bold yellow]")
    console.print("[yellow]Asking clarifying questions:[/yellow]\n")
    
    # Build initial query
    initial_query = f"Delete file '{filename}'"
    
    # Use MODEL to check if we need more info
    is_sufficient, missing_info = evaluate_information_sufficiency(initial_query)
    
    if is_sufficient:
        clarified_query = f"Delete the file '{filename}'"
        console.print(f"\n[green]✓ I'll delete '{filename}'[/green]\n")
        return clarified_query, True
    
    # Model says we need directory - ask for it
    console.print(f"[yellow]Model says: {missing_info}[/yellow]")
    
    # Check if path is mentioned elsewhere in query
    path_patterns = [
        (r'folder\s+(?:named|called)\s+([a-zA-Z0-9_\-\.]+)', 'folder'),
        (r'directory\s+(?:named|called)\s+([a-zA-Z0-9_\-\.]+)', 'directory'),
        (r'in\s+([a-zA-Z0-9_/\-\.]+)(?:\s+folder)?', 'path'),
    ]
    
    detected_path = None
    for pattern, path_type in path_patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            detected_path = match.group(1)
            console.print(f"[dim]✓ Detected {path_type}: {detected_path}[/dim]")
            break
    
    if not detected_path:
        directory = Prompt.ask("[cyan]Q1: Directory path where this file is located[/cyan]", default=".").strip()
    else:
        directory = detected_path
    
    clarified_query = f"Delete the file '{directory}/{filename}'"
    
    # Final model check
    is_sufficient, _ = evaluate_information_sufficiency(clarified_query)
    if not is_sufficient:
        console.print("[yellow]Let me ask for more details:[/yellow]")
        more_info = Prompt.ask("[cyan]Any additional context?[/cyan]", default="").strip()
        if more_info:
            clarified_query += f" ({more_info})"
    
    console.print(f"\n[green]✓ I'll delete '{directory}/{filename}'[/green]\n")
    return clarified_query, True

def check_and_clarify_find_command(query):
    """Smart clarification for find/search commands - only asks for missing information."""
    if not re.search(r'\bfind\b|\bsearch\b|\blocate\b', query, re.IGNORECASE):
        return query, False
    
    # Check if search term is too generic
    if not re.search(r'\bfind\s+(?:files?|all)\b', query, re.IGNORECASE):
        return query, False
    
    console.print("\n[bold yellow]ℹ️  Find command detected[/bold yellow]")
    console.print("[yellow]Asking clarifying questions:[/yellow]\n")
    
    # Ask for search term
    search_term = Prompt.ask("[cyan]Q1: What are you looking for? (filename, pattern, or extension)[/cyan]").strip()
    
    # Ask for search location
    search_location = Prompt.ask("[cyan]Q2: Where should I search? (directory path)[/cyan]", default=".").strip()
    
    # Determine file type - check if already specified in search term
    if '*.' in search_term or '.' in search_term:
        console.print(f"[dim]✓ File type detected in search term[/dim]")
        file_type = "detected"
    else:
        file_type = Prompt.ask("[cyan]Q3: Specific file type? (e.g., txt, py, or 'all' for any)[/cyan]", default="all").strip()
    
    clarified_query = f"Find files matching '{search_term}' in '{search_location}' with type '{file_type}'"
    console.print(f"\n[green]✓ I'll search for '{search_term}' in '{search_location}'[/green]\n")
    return clarified_query, True

def check_and_clarify_copy_command(query):
    """Smart clarification for copy - uses model to verify if info is sufficient."""
    if not re.search(r'\bcopy\b|\bcp\b', query, re.IGNORECASE):
        return query, False
    
    # Check if source or destination are mentioned
    from_match = re.search(r'\bfrom\s+([^\s]+)', query, re.IGNORECASE)
    to_match = re.search(r'\bto\s+([^\s]+)', query, re.IGNORECASE)
    
    # Build initial query for model evaluation
    initial_query = "Copy operation"
    if from_match:
        initial_query += f" from '{from_match.group(1)}'"
    if to_match:
        initial_query += f" to '{to_match.group(1)}'"
    
    # Use MODEL to check if we have enough information
    is_sufficient, missing_info = evaluate_information_sufficiency(initial_query)
    
    if is_sufficient and from_match and to_match:
        clarified_query = f"Copy from '{from_match.group(1)}' to '{to_match.group(1)}'"
        console.print(f"\n[green]✓ I'll copy '{from_match.group(1)}' to '{to_match.group(1)}'[/green]\n")
        return clarified_query, True
    
    # Model says we need more info
    console.print("\n[bold yellow]ℹ️  Copy command detected[/bold yellow]")
    console.print("[yellow]Asking clarifying questions:[/yellow]\n")
    if missing_info:
        console.print(f"[yellow]Model says: {missing_info}[/yellow]\n")
    
    # Ask for source if not provided
    if not from_match:
        source = Prompt.ask("[cyan]Q1: Source file or directory path[/cyan]").strip()
    else:
        source = from_match.group(1)
        console.print(f"[dim]✓ Source: {source}[/dim]")
    
    # Ask for destination if not provided
    if not to_match:
        destination = Prompt.ask("[cyan]Q2: Destination path[/cyan]").strip()
    else:
        destination = to_match.group(1)
        console.print(f"[dim]✓ Destination: {destination}[/dim]")
    
    clarified_query = f"Copy from '{source}' to '{destination}'"
    
    # Model final check
    is_sufficient, _ = evaluate_information_sufficiency(clarified_query)
    if not is_sufficient:
        recursive_ans = Prompt.ask("[cyan]Q3: Copy recursively? (yes/no)[/cyan]", default="yes").strip().lower()
        clarified_query += f" {'recursively' if recursive_ans == 'yes' else ''}"
    
    console.print(f"\n[green]✓ I'll copy '{source}' to '{destination}'[/green]\n")
    return clarified_query, True

def check_and_clarify_create_command(query):
    """Smart clarification for create file commands - uses model to verify sufficiency."""
    if not re.search(r'\bcreate\b|\btouch\b|\bmake.*file\b', query, re.IGNORECASE):
        return query, False
    
    console.print(f"\n[bold yellow]ℹ️  Create file command detected[/bold yellow]")
    console.print("[yellow]I need some clarifications:[/yellow]\n")
    
    # Ask for filename
    filename = Prompt.ask("[cyan]Q1: Filename/path (e.g., py.txt or path/to/file.txt)[/cyan]").strip()
    if not filename:
        return query, False
    
    # Extract directory and filename if provided as full path
    if '/' in filename or '\\' in filename:
        parts = filename.replace('\\', '/').split('/')
        directory = '/'.join(parts[:-1]) if len(parts) > 1 else '.'
        filename_only = parts[-1]
    else:
        directory = '.'
        filename_only = filename
    
    # Build intermediate query for model evaluation
    intermediate_query = f"Create file at '{directory}/{filename_only}'"
    
    # Use MODEL to determine if we need more questions
    is_sufficient, missing_info = evaluate_information_sufficiency(intermediate_query)
    
    if is_sufficient:
        # Model says we have enough - create empty file
        clarified_query = f"Create empty file '{directory}/{filename_only}'"
        console.print(f"\n[green]✓ I'll create '{directory}/{filename_only}'[/green]\n")
        return clarified_query, True
    
    # Model says we need more info - ask follow-up questions
    console.print(f"[yellow]Model says: {missing_info}[/yellow]")
    
    # Ask about file type if extension missing
    needs_filetype = '.' not in filename_only
    if needs_filetype:
        file_type = Prompt.ask("[cyan]Q2: File type (e.g., txt, py, sh, json)[/cyan]", default="txt").strip()
        filetype_str = f".{file_type}" if file_type and not file_type.startswith('.') else file_type
        filename_only = f"{filename_only}{filetype_str}"
    
    # Ask about content
    has_content = Prompt.ask("[cyan]Q3: Add initial content? (yes/no)[/cyan]", default="no").strip().lower() == "yes"
    
    if has_content:
        content = Prompt.ask("[cyan]Q4: Initial content[/cyan]").strip()
        clarified_query = f"Create file '{directory}/{filename_only}' with content: {content}"
    else:
        clarified_query = f"Create empty file '{directory}/{filename_only}'"
    
    # Final model check
    is_sufficient, _ = evaluate_information_sufficiency(clarified_query)
    
    console.print(f"\n[green]✓ I'll create '{directory}/{filename_only}'[/green]\n")
    return clarified_query, True

def check_and_clarify_grep_command(query):
    """Smart clarification for grep/search in files - only asks for missing information."""
    if not re.search(r'\bgrep\b|\bsearch in\b|\bfind text\b', query, re.IGNORECASE):
        return query, False
    
    console.print("\n[bold yellow]ℹ️  Search in files detected[/bold yellow]")
    console.print("[yellow]Asking clarifying questions:[/yellow]\n")
    
    # Ask for search pattern
    search_pattern = Prompt.ask("[cyan]Q1: Text pattern to search for[/cyan]").strip()
    if not search_pattern:
        return query, False
    
    # Ask for location - check if already mentioned
    location_match = re.search(r'\bin\s+([^\s]+)', query, re.IGNORECASE)
    if location_match:
        search_location = location_match.group(1)
        console.print(f"[dim]✓ Search location: {search_location}[/dim]")
    else:
        search_location = Prompt.ask("[cyan]Q2: File or directory to search in[/cyan]", default=".").strip()
    
    # Check if case sensitivity matters - only ask if pattern has mixed case
    if search_pattern != search_pattern.lower() and search_pattern != search_pattern.upper():
        case_sensitive = Prompt.ask("[cyan]Q3: Case sensitive? (yes/no)[/cyan]", default="yes").strip().lower()
    else:
        console.print(f"[dim]✓ Pattern case is uniform, using default matching[/dim]")
        case_sensitive = "no"
    
    clarified_query = f"Search for '{search_pattern}' in '{search_location}' {'case sensitive' if case_sensitive == 'yes' else 'case insensitive'}"
    console.print(f"\n[green]✓ I'll search for '{search_pattern}' in '{search_location}'[/green]\n")
    return clarified_query, True

def check_and_clarify_rename_command(query):
    """Smart clarification for rename/move commands - only asks for missing information."""
    if not re.search(r'\brename\b|\bmv\b', query, re.IGNORECASE):
        return query, False
    
    # Check if both old and new names are specified
    from_match = re.search(r'\bfrom\s+([^\s]+)', query, re.IGNORECASE)
    to_match = re.search(r'\bto\s+([^\s]+)', query, re.IGNORECASE)
    
    # If already has enough info, don't clarify
    if from_match and to_match:
        return query, False
    
    console.print("\n[bold yellow]ℹ️  Rename command detected[/bold yellow]")
    console.print("[yellow]Asking clarifying questions:[/yellow]\n")
    
    # Ask for old name if not provided
    if not from_match:
        old_name = Prompt.ask("[cyan]Q1: Current file path[/cyan]").strip()
    else:
        old_name = from_match.group(1)
        console.print(f"[dim]✓ Current path: {old_name}[/dim]")
    
    # Ask for new name if not provided
    if not to_match:
        new_name = Prompt.ask("[cyan]Q2: New file path/name[/cyan]").strip()
    else:
        new_name = to_match.group(1)
        console.print(f"[dim]✓ New path: {new_name}[/dim]")
    
    clarified_query = f"Rename '{old_name}' to '{new_name}'"
    console.print(f"\n[green]✓ I'll rename '{old_name}' to '{new_name}'[/green]\n")
    return clarified_query, True

def check_and_clarify_permission_command(query):
    """Smart clarification for permission/chmod commands - only asks for missing information."""
    if not re.search(r'\bchmod\b|\bpermission\b|\bmake.*executable\b', query, re.IGNORECASE):
        return query, False
    
    console.print("\n[bold yellow]ℹ️  Permission change detected[/bold yellow]")
    console.print("[yellow]Asking clarifying questions:[/yellow]\n")
    
    # Check if file path is mentioned
    file_match = re.search(r'(?:on|for|to)\s+([^\s]+)', query, re.IGNORECASE)
    if file_match and '/' in file_match.group(1):
        file_path = file_match.group(1)
        console.print(f"[dim]✓ File path: {file_path}[/dim]")
    else:
        file_path = Prompt.ask("[cyan]Q1: File or directory path[/cyan]").strip()
        if not file_path:
            return query, False
    
    # Check if permission is already specified (755, 644, +x, etc.)
    perm_match = re.search(r'(?:to|as)\s+([0-7]{3}|[+\-][rwx]+)', query, re.IGNORECASE)
    if perm_match:
        permission = perm_match.group(1)
        console.print(f"[dim]✓ Permission mode: {permission}[/dim]")
    else:
        permission = Prompt.ask("[cyan]Q2: Permission mode (e.g., 755, 644, +x, u+rwx)[/cyan]").strip()
        if not permission:
            return query, False
    
    # Only ask about recursive if query mentions directory or path looks like directory
    if 'directory' in query.lower() or 'folder' in query.lower() or query.endswith('/'):
        recursive_ans = Prompt.ask("[cyan]Q3: Apply recursively? (yes/no)[/cyan]", default="no").strip().lower()
        recursive = "recursively" if recursive_ans == "yes" else ""
    else:
        recursive = ""
        console.print(f"[dim]✓ Applying to single item[/dim]")
    
    clarified_query = f"Change permissions of '{file_path}' to '{permission}' {recursive}"
    console.print(f"\n[green]✓ I'll set permissions on '{file_path}'[/green]\n")
    return clarified_query, True

def check_and_clarify_compress_command(query):
    """Smart clarification for compress/archive commands - only asks for missing information."""
    if not re.search(r'\bcompress\b|\bzip\b|\btar\b|\barchive\b', query, re.IGNORECASE):
        return query, False
    
    console.print("\n[bold yellow]ℹ️  Compression/Archive command detected[/bold yellow]")
    console.print("[yellow]Asking clarifying questions:[/yellow]\n")
    
    # Check if source is mentioned
    source_match = re.search(r'(?:compress|archive)\s+([^\s]+)', query, re.IGNORECASE)
    if source_match:
        source = source_match.group(1)
        console.print(f"[dim]✓ Source: {source}[/dim]")
    else:
        source = Prompt.ask("[cyan]Q1: Files or directory to compress[/cyan]").strip()
        if not source:
            return query, False
    
    # Check if output filename is mentioned
    output_match = re.search(r'(?:to|into|as)\s+([^\s]+)', query, re.IGNORECASE)
    if output_match:
        output = output_match.group(1)
        console.print(f"[dim]✓ Output: {output}[/dim]")
        # Infer compression from extension
        if '.tar.gz' in output or '.tgz' in output:
            compression = "tar.gz"
        elif '.tar' in output:
            compression = "tar"
        elif '.zip' in output:
            compression = "zip"
        elif '.7z' in output:
            compression = "7z"
        else:
            compression = "tar.gz"
        console.print(f"[dim]✓ Format detected from extension: {compression}[/dim]")
    else:
        output = Prompt.ask("[cyan]Q2: Output archive name (with extension)[/cyan]").strip()
        if not output:
            return query, False
        compression = Prompt.ask("[cyan]Q3: Compression type (tar, zip, tar.gz, 7z)[/cyan]", default="tar.gz").strip()
    
    clarified_query = f"Compress '{source}' to '{output}' using '{compression}' format"
    console.print(f"\n[green]✓ I'll compress '{source}' to '{output}'[/green]\n")
    return clarified_query, True

def check_and_clarify_extract_command(query):
    """Smart clarification for extract/unzip commands - only asks for missing information."""
    if not re.search(r'\bextract\b|\bunzip\b|\buntar\b', query, re.IGNORECASE):
        return query, False
    
    console.print("\n[bold yellow]ℹ️  Extract/Unarchive command detected[/bold yellow]")
    console.print("[yellow]Asking clarifying questions:[/yellow]\n")
    
    # Check if archive file is mentioned
    archive_match = re.search(r'(?:extract|unzip|untar)\s+([^\s]+)', query, re.IGNORECASE)
    if archive_match:
        archive = archive_match.group(1)
        console.print(f"[dim]✓ Archive: {archive}[/dim]")
    else:
        archive = Prompt.ask("[cyan]Q1: Archive file path[/cyan]").strip()
        if not archive:
            return query, False
    
    # Check if destination is mentioned
    dest_match = re.search(r'(?:to|into)\s+([^\s]+)', query, re.IGNORECASE)
    if dest_match:
        destination = dest_match.group(1)
        console.print(f"[dim]✓ Destination: {destination}[/dim]")
    else:
        # Only ask if not mentioned
        ask_dest = Prompt.ask("[cyan]Q2: Extract to directory (leave blank for current)[/cyan]", default=".").strip()
        destination = ask_dest if ask_dest else "."
    
    clarified_query = f"Extract '{archive}' to '{destination}'"
    console.print(f"\n[green]✓ I'll extract '{archive}' to '{destination}'[/green]\n")
    return clarified_query, True

def generate_bash(query, error_context=None, show_rag_debug=False):
    """Generate bash commands using retrieval + Ollama."""
    debug_print("=" * 60, "bold cyan")
    debug_print("🚀 Starting command generation pipeline", "bold cyan")
    debug_print("=" * 60, "bold cyan")
    
    examples = retrieve(query)
    
    # Show RAG debug info if explicitly requested OR in debug mode
    if show_rag_debug or DEBUG_MODE:
        console.print("\n[bold blue]🔍 RAG Retrieval Debug Info:[/bold blue]")
        console.print(f"[cyan]Query:[/cyan] {query}")
        console.print(f"[cyan]Number of examples retrieved:[/cyan] {len(examples)}")
        for i, item in enumerate(examples[:3], 1):
            console.print(f"\n[yellow]Example {i}:[/yellow]")
            console.print(f"  NL: {item['instruction'][:100]}...")
            console.print(f"  Bash: {item['response'][:100]}...")
        console.print()
    
    prompt = """You are a Linux Bash command expert. Generate ONLY executable Bash commands.

RULES:
1. Output ONLY valid bash commands, one per line
2. NO explanations, markdown, or comments
3. NO placeholders like 'path/to/file' - use actual filenames
4. Commands must be directly executable
5. PREFER SIMPLE bash utilities (grep, wc, awk, sed, find) over writing Python scripts
6. Only do what is explicitly asked. If asked to "create" a file, only create it - don't execute it
7. To run Python files, use: python3 filename.py (not ./filename.py)
8. For Python scripts ONLY, include #!/usr/bin/env python3 as first line
9. For plain text files, create EMPTY files or files with simple text content - NO shebangs
10. Match file type to request: text file = plain text, Python file = Python code with shebang
11. Don't add chmod +x for non-executable files
12. IMPORTANT: Use the EXAMPLES below as templates - they show the BEST approach for similar tasks

"""

    if error_context:
        prompt += f"\nPREVIOUS ERROR:\nCommand: {error_context['command']}\nError: {error_context['error']}\n"
        prompt += "Fix the error and generate correct commands.\n\n"

    prompt += "EXAMPLES FROM KNOWLEDGE BASE (follow these patterns closely):\n"
    for item in examples[:3]:
        prompt += f"Query: {item['instruction']}\nCommands: {item['response']}\n\n"

    prompt += f"Current Query: {query}\n"
    prompt += "IMPORTANT: Look at the examples above - they show simple, effective bash commands.\n"
    prompt += "Use similar patterns and tools (grep, wc, find, awk, etc.) for your response.\n"
    prompt += "Only write Python scripts if the task genuinely requires complex logic not possible with bash utilities.\n"
    prompt += "Commands:"

    result = ollama_generate(prompt, max_tokens=300)
    debug_print(f"📝 Raw model output:\n{result}\n", "dim")
    
    debug_print("🧹 Sanitizing bash commands...")
    cleaned = sanitize_bash(result)
    debug_print(f"✓ Sanitized to {len(cleaned.split(chr(10)))} lines", "green")
    
    # Validate heredoc completeness
    if '<<' in cleaned and 'EOF' in cleaned:
        lines = cleaned.split('\n')
        in_heredoc = False
        heredoc_start = -1
        
        for i, line in enumerate(lines):
            if '<<' in line and 'EOF' in line and not in_heredoc:
                in_heredoc = True
                heredoc_start = i
            elif in_heredoc and line.strip() == 'EOF':
                heredoc_content = '\n'.join(lines[heredoc_start+1:i])
                if heredoc_content.count('(') > heredoc_content.count(')'):
                    console.print("[bold red]⚠️ WARNING: Generated code appears incomplete.[/bold red]")
                break
    
    # Post-processing for file type matching
    query_lower = query.lower()
    
    if ('python' in query_lower or '.py' in query_lower) and '<<' in cleaned and '#!/usr/bin/env python' not in cleaned:
        console.print("[yellow]⚠️ Adding missing Python shebang...[/yellow]")
        lines = cleaned.split('\n')
        new_lines = []
        for i, line in enumerate(lines):
            new_lines.append(line)
            if '<<' in line and 'EOF' in line:
                if i + 1 < len(lines) and not lines[i + 1].strip().startswith('#!'):
                    new_lines.append('#!/usr/bin/env python3')
        cleaned = '\n'.join(new_lines)
    
    if ('text file' in query_lower or '.txt' in query_lower) and '#!/usr/bin/env python' in cleaned:
        console.print("[yellow]⚠️ Removing Python shebang from text file...[/yellow]")
        cleaned = re.sub(r'#!/usr/bin/env python\d?\n?', '', cleaned)
    
    if ('text file' in query_lower or 'txt file' in query_lower) and 'chmod +x' in cleaned:
        console.print("[yellow]⚠️ Removing unnecessary chmod +x for text file...[/yellow]")
        lines = cleaned.split('\n')
        cleaned = '\n'.join([line for line in lines if 'chmod +x' not in line])
    
    # Only show raw output in non-debug mode (debug mode shows it earlier)
    if not DEBUG_MODE:
        console.print(f"[dim]Raw model output:[/dim]\n[dim]{result}[/dim]\n")
    
    debug_print("✅ Command generation complete", "bold green")
    
    return cleaned, examples

def analyze_command_dependencies(commands):
    """Analyze dependencies between commands to determine execution strategy."""
    debug_print(f"🔗 Analyzing dependencies for {len(commands)} commands...")
    dependencies = []
    
    for i, cmd in enumerate(commands):
        dep_info = {
            'index': i,
            'command': cmd,
            'creates_files': [],
            'reads_files': [],
            'requires_success': True,  # Default: stop on error
            'is_idempotent': False
        }
        
        # Detect file creation
        if '>' in cmd or 'touch' in cmd or 'mkdir' in cmd or 'cat >' in cmd:
            # Extract potential filenames
            file_match = re.search(r'(?:>|touch|mkdir)\s+([^\s;|&]+)', cmd)
            if file_match:
                dep_info['creates_files'].append(file_match.group(1))
        
        # Detect file reading
        file_patterns = [
            r'\b([a-zA-Z0-9_\-\.\/]+\.(?:py|txt|sh|json|csv|log))\b',  # Common file extensions
            r'(?:cat|grep|wc|head|tail|less|more)\s+([^\s;|&]+)',  # Read commands
        ]
        for pattern in file_patterns:
            for match in re.finditer(pattern, cmd):
                filename = match.group(1)
                if filename not in dep_info['creates_files']:
                    dep_info['reads_files'].append(filename)
        
        # Detect idempotent commands (safe to retry)
        idempotent_cmds = ['mkdir -p', 'touch', 'echo', 'cat', 'grep', 'wc', 'find', 'ls']
        if any(safe_cmd in cmd for safe_cmd in idempotent_cmds):
            dep_info['is_idempotent'] = True
        
        dependencies.append(dep_info)
    
    # Check cross-command dependencies
    for i, dep in enumerate(dependencies):
        for j in range(i + 1, len(dependencies)):
            next_dep = dependencies[j]
            # If next command reads what current command creates
            for created_file in dep['creates_files']:
                if created_file in next_dep['reads_files']:
                    msg = f"  ⚡ Dependency detected: Command {i+1} creates '{created_file}' needed by Command {j+1}"
                    if DEBUG_MODE:
                        console.print(f"[cyan]{msg}[/cyan]")
                    else:
                        console.print(f"[dim]{msg}[/dim]")
                    next_dep['depends_on'] = i
    
    debug_print(f"✓ Dependency analysis complete", "green")
    return dependencies

def execute_sequence(command_str, enable_dependency_check=True):
    """Execute a sequence of bash commands safely with dependency awareness."""
    debug_print("🔧 Parsing command sequence...")
    commands = []
    lines = command_str.split('\n')
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        if not line:
            i += 1
            continue
        
        # Handle heredocs
        if '<<' in line:
            heredoc_match = re.search(r'<<\s*["\']?(\w+)["\']?', line)
            if heredoc_match:
                delimiter = heredoc_match.group(1)
                full_cmd = [line]
                i += 1
                
                while i < len(lines):
                    if lines[i].strip() == delimiter:
                        full_cmd.append(lines[i].strip())
                        break
                    full_cmd.append(lines[i])
                    i += 1
                
                commands.append('\n'.join(full_cmd))
                i += 1
                continue
        
        commands.append(line)
        i += 1
    
    if not commands:
        return {"output": "No commands to execute", "is_error": True}
    
    debug_print(f"✓ Parsed {len(commands)} command(s)", "green")
    
    # Analyze dependencies
    if enable_dependency_check and len(commands) > 1:
        style = "bold cyan" if DEBUG_MODE else "cyan"
        console.print(f"\n[{style}]🔍 Analyzing {len(commands)} command(s){'for dependencies' if DEBUG_MODE else ''}...[/{style}]")
        dependencies = analyze_command_dependencies(commands)
    else:
        dependencies = [{'index': i, 'command': cmd, 'requires_success': True} 
                       for i, cmd in enumerate(commands)]
    
    outputs = []
    
    for i, cmd in enumerate(commands):
        dep_info = dependencies[i] if i < len(dependencies) else {}
        display_cmd = cmd.split('\n')[0] + '...' if '\n' in cmd else cmd
        
        console.print(f"\n[bold yellow]→ Executing Command {i+1}/{len(commands)}:[/bold yellow] {display_cmd}")
        
        # Check if depends on previous command
        if 'depends_on' in dep_info:
            console.print(f"[dim]  ⚠️  This command depends on Command {dep_info['depends_on'] + 1}'s success[/dim]")
        
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                executable="/bin/bash",
                capture_output=True,
                text=True,
                timeout=30
            )
            
            output = (result.stdout + result.stderr).strip()
            outputs.append(f"$ {display_cmd}\n{output or '(success)'}")
            
            if result.returncode != 0:
                console.print(f"[bold red]⚠️ Command {i+1} failed with return code {result.returncode}[/bold red]")
                
                # Check if subsequent commands depend on this
                has_dependents = any(d.get('depends_on') == i for d in dependencies[i+1:])
                
                if has_dependents:
                    console.print(f"[bold red]🛑 Stopping execution: Dependent commands exist and would fail[/bold red]")
                    return {
                        "output": "\n".join(outputs), 
                        "is_error": True,
                        "failed_command": cmd,
                        "error_message": output,
                        "reason": "Dependency chain broken"
                    }
                elif dep_info.get('requires_success', True):
                    console.print(f"[yellow]⚠️  Stopping execution as configured[/yellow]")
                    return {
                        "output": "\n".join(outputs), 
                        "is_error": True,
                        "failed_command": cmd,
                        "error_message": output
                    }
                else:
                    console.print(f"[yellow]⚠️  Continuing despite error (command is optional)[/yellow]")
                    continue
                
        except subprocess.TimeoutExpired:
            return {
                "output": "Error: Command timed out", 
                "is_error": True,
                "failed_command": cmd,
                "error_message": "Timeout after 30 seconds"
            }
        except Exception as e:
            return {
                "output": f"Error: {e}", 
                "is_error": True,
                "failed_command": cmd,
                "error_message": str(e)
            }
    
    return {"output": "\n".join(outputs), "is_error": False}

def explain_error(original_query, failed_cmd, error_msg, rag_examples):
    """Ask Ollama to explain why a command failed."""
    console.print("\n[bold magenta]🔬 Analyzing Error with RAG Context...[/bold magenta]")
    
    debug_print("📚 Examining RAG examples used for generation...")
    
    # Show RAG examples if in debug mode or if there's a problem
    if DEBUG_MODE or not rag_examples:
        console.print(f"\n[cyan]RAG Examples Used for Generation:[/cyan]")
        if rag_examples:
            for i, item in enumerate(rag_examples[:3], 1):
                console.print(f"\n[yellow]Example {i}:[/yellow]")
                console.print(f"  Query: {item['instruction'][:80]}...")
                console.print(f"  Command: {item['response'][:80]}...")
        else:
            console.print("[bold red]⚠️ WARNING: No RAG examples were retrieved![/bold red]")
    
    debug_print("🤖 Generating error explanation...")
    
    prompt = f"""Explain briefly why this command failed and suggest a fix:

User wanted: {original_query}
Failed command: {failed_cmd}
Error message: {error_msg}

Provide a concise 2-3 sentence explanation and the corrected command.
"""
    explanation = ollama_generate(prompt, max_tokens=100, temperature=0.6)
    
    # RAG Quality Check - only show in debug mode or if there are issues
    if DEBUG_MODE:
        console.print("\n[bold blue]RAG Quality Check:[/bold blue]")
        if rag_examples:
            console.print(f"✓ Retrieved {len(rag_examples)} examples from knowledge base")
            
            query_lower = original_query.lower()
            relevant_count = sum(1 for ex in rag_examples[:3] 
                               if any(word in ex['instruction'].lower() for word in query_lower.split()[:3]))
            
            if relevant_count > 0:
                console.print(f"✓ {relevant_count}/3 examples appear relevant to query")
            else:
                console.print("[yellow]⚠️ Retrieved examples may not be highly relevant[/yellow]")
        else:
            console.print("[bold red]✗ RAG retrieval failed or returned no results[/bold red]")
    
    return explanation

# ===============================
# MAIN CLI LOOP
# ===============================
def run_cli():
    global DEBUG_MODE
    
    console.print("[bold magenta]=== Linux Bash CLI Agent (RAG + Ollama) ===[/bold magenta]")
    console.print(f"[bold yellow]Platform:[/bold yellow] {platform.system()}")
    
    # Prompt for mode selection
    console.print("\n[bold cyan]Select mode:[/bold cyan]")
    console.print("  1. [green]Normal mode[/green] - Clean output with minimal details")
    console.print("  2. [yellow]Debug mode[/yellow] - Show all internal workings\n")
    
    mode_choice = Prompt.ask("Choose mode", choices=["1", "2"], default="1")
    DEBUG_MODE = (mode_choice == "2")
    
    if DEBUG_MODE:
        console.print("\n[bold yellow]🔧 DEBUG MODE ENABLED[/bold yellow]")
        console.print("[dim]You will see detailed internal workings of the agent[/dim]\n")
    else:
        console.print("\n[bold green]✓ Normal mode active[/bold green]\n")
    
    console.print("[bold green]Type a natural language instruction (or 'exit' to quit)[/bold green]")
    console.print("[dim]Commands: 'exit'/'quit' to exit, 'toggle debug' to switch modes[/dim]\n")
    
    max_retries = 2
    
    while True:
        query = Prompt.ask("> ").strip()
        
        # Handle special commands
        if query.lower() in ["exit", "quit", "q"]:
            console.print("[bold red]Goodbye![/bold red]")
            break
        
        if query.lower() in ["toggle debug", "debug", "toggle"]:
            DEBUG_MODE = not DEBUG_MODE
            if DEBUG_MODE:
                console.print("[bold yellow]🔧 Debug mode ENABLED - Showing all internal workings[/bold yellow]\n")
            else:
                console.print("[bold green]✓ Debug mode DISABLED - Clean output mode[/bold green]\n")
            continue
        
        if not query:
            continue
        
        # Chain of smart clarifications for specific command types
        clarified_query = query
        was_clarified = False
        
        # Try each specific command clarifier in order
        clarifiers = [
            check_and_clarify_delete_command,
            check_and_clarify_find_command,
            check_and_clarify_copy_command,
            check_and_clarify_create_command,
            check_and_clarify_grep_command,
            check_and_clarify_rename_command,
            check_and_clarify_permission_command,
            check_and_clarify_compress_command,
            check_and_clarify_extract_command,
        ]
        
        for clarifier in clarifiers:
            clarified_query, was_clarified = clarifier(clarified_query)
            if was_clarified:
                query = clarified_query
                break
        
        error_context = None
        rag_examples = None
        
        for attempt in range(max_retries + 1):
            if attempt > 0:
                console.print(f"\n[bold yellow]🔄 Auto-retry attempt {attempt}/{max_retries}[/bold yellow]")
            
            show_rag_debug = (attempt == 2)
            
            bash_cmds, rag_examples = generate_bash(query, error_context, show_rag_debug)
            
            if not bash_cmds or bash_cmds.strip() == "":
                console.print("[bold red]No valid commands generated.[/bold red]\n")
                break
            
            # Display RAG examples - only in debug mode or if explicitly requested
            if DEBUG_MODE or show_rag_debug:
                console.print("\n[bold blue]📚 Retrieved RAG Examples:[/bold blue]")
                if rag_examples and len(rag_examples) > 0:
                    for i, item in enumerate(rag_examples[:3], 1):
                        console.print(f"\n[yellow]Example {i}:[/yellow]")
                        console.print(f"  [dim]Query:[/dim] {item['instruction'][:100]}...")
                        console.print(f"  [dim]Command:[/dim] {item['response'][:100]}...")
                else:
                    console.print("[red]⚠️ No RAG examples retrieved[/red]")
            
            console.print(f"\n[cyan]Generated command(s):[/cyan]\n[white]{bash_cmds}[/white]\n")
            
            # Safety check for dangerous commands
            dangers = check_dangerous_commands(bash_cmds)
            if dangers:
                console.print("[bold red]⚠️  DANGER: Potentially destructive commands detected![/bold red]")
                for danger in dangers:
                    console.print(f"  [red]• {danger}[/red]")
                console.print("\n[yellow]These commands could cause system damage or data loss.[/yellow]")
                
                if not Confirm.ask("[bold]Are you ABSOLUTELY SURE you want to proceed?[/bold]", default=False):
                    console.print("[green]✓ Execution cancelled for safety.[/green]\n")
                    break
            
            # Auto-execute on first attempt, ask for confirmation on retries
            should_execute = True
            if attempt == 0 and not dangers:
                should_execute = Confirm.ask("Execute these commands?", default=True)
            elif attempt > 0:
                style = "yellow" if DEBUG_MODE else "dim"
                console.print(f"[{style}]Auto-executing retry attempt...[/{style}]")
            
            if not should_execute:
                console.print("[grey]Skipped execution.[/grey]\n")
                break
            
            debug_print("\n" + "=" * 60, "bold cyan")
            debug_print("⚙️  Starting command execution", "bold cyan")
            debug_print("=" * 60, "bold cyan")
            
            result = execute_sequence(bash_cmds)
            console.print(f"\n[bold white]Output:[/bold white]\n{result['output']}\n")
            
            if result["is_error"]:
                console.print("[bold red]❌ Command(s) failed![/bold red]")
                
                if attempt == 0:
                    console.print("[yellow]Attempting automatic fix...[/yellow]")
                    error_context = {
                        'command': result.get('failed_command', bash_cmds),
                        'error': result.get('error_message', result['output'])
                    }
                    continue
                
                elif attempt == 1:
                    console.print("[yellow]Analyzing error with RAG verification...[/yellow]")
                    explanation = explain_error(
                        query, 
                        result.get('failed_command', bash_cmds),
                        result.get('error_message', result['output']),
                        rag_examples
                    )
                    console.print(f"\n[bold yellow]🧩 Error Analysis:[/bold yellow]\n{explanation}\n")
                    
                    error_context = {
                        'command': result.get('failed_command', bash_cmds),
                        'error': result.get('error_message', result['output'])
                    }
                    
                    if Confirm.ask("Try one more time with full RAG debug info?"):
                        continue
                    break
                
                else:
                    console.print("[bold red]❌ Maximum retry attempts reached.[/bold red]\n")
                    break
            else:
                console.print("[bold green]✅ All commands executed successfully![/bold green]\n")
                break


if __name__ == "__main__":
    run_cli()