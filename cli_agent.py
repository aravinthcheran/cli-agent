import os
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
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mistral")
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

def extract_filename_from_create_query(query):
    """Extract filename intelligently from create file queries using LLM."""
    # First, try simple regex for explicit filenames
    # Pattern: "create filename.ext" or "file name is X.ext"
    match = re.search(r'(?:create|make|touch)\s+([a-zA-Z0-9_\-]+\.[a-zA-Z0-9]+)', query, re.IGNORECASE)
    if match:
        return match.group(1)
    
    match = re.search(r'file\s+name\s+is\s+([a-zA-Z0-9_\-]+(?:\.[a-zA-Z0-9]+)?)', query, re.IGNORECASE)
    if match:
        filename = match.group(1)
        # Add extension if not present
        if '.' not in filename:
            if 'python' in query.lower():
                filename += '.py'
            elif 'text' in query.lower():
                filename += '.txt'
            elif 'bash' in query.lower() or 'shell' in query.lower():
                filename += '.sh'
        return filename
    
    match = re.search(r'(?:called|named)\s+([a-zA-Z0-9_\-]+(?:\.[a-zA-Z0-9]+)?)', query, re.IGNORECASE)
    if match:
        filename = match.group(1)
        if '.' not in filename:
            if 'python' in query.lower():
                filename += '.py'
            elif 'text' in query.lower():
                filename += '.txt'
        return filename
    
    # If no explicit filename found, use LLM to suggest one
    console.print("[dim]🤖 Using LLM to generate appropriate filename...[/dim]")
    
    prompt = f"""Based on this request, suggest a short, descriptive filename (with extension):

Request: {query}

Rules:
1. Suggest ONLY the filename, nothing else
2. Use lowercase letters, numbers, underscores only
3. Keep it short (1-2 words max)
4. Add appropriate extension (.py for Python, .txt for text, .sh for bash, etc.)
5. Make it descriptive of what the file does
6. If the request mentions checking/testing something, use that as the name

Examples:
- "create a python file to check prime numbers" → prime_checker.py
- "make a file for odd or even" → odd_even.py
- "python file that calculates factorial" → factorial.py
- "script to sort data" → sort.py

Filename:"""
    
    result = ollama_generate(prompt, max_tokens=20, temperature=0.3).strip()
    
    # Clean up the result (remove quotes, extra text, etc.)
    result = result.split('\n')[0].strip()  # Take first line only
    result = result.replace('"', '').replace("'", "").strip()
    
    # Validate and clean filename
    result = re.sub(r'[^a-zA-Z0-9_\-\.]', '', result)
    
    if result and '.' in result:
        console.print(f"[dim]✓ LLM suggested filename: {result}[/dim]")
        return result
    
    return None

def detect_content_requirement(query):
    """Detect if the user wants content in the file based on query context."""
    query_lower = query.lower()
    
    # Strong indicators that content is needed
    content_indicators = [
        'which checks',
        'that checks',
        'which does',
        'that does',
        'which calculates',
        'that calculates',
        'which prints',
        'that prints',
        'that gets',
        'which gets',
        'to check',
        'to calculate',
        'to print',
        'to find',
        'to sort',
        'to search',
        'to get',
        'for checking',
        'for calculating',
        'for printing',
        'for getting',
        'with code',
        'with logic',
        'with function',
        'with a function',
        'function to',
        'script to',
        'script that',
        'script which',
        'program to',
        'program that',
        'program which',
        'gets',
        'and prints',
        'and calculates',
        'and shows',
    ]
    
    if any(indicator in query_lower for indicator in content_indicators):
        return 'yes'
    
    # Empty file indicators
    empty_indicators = ['empty', 'blank', 'just create', 'only create', 'touch']
    if any(indicator in query_lower for indicator in empty_indicators):
        return 'no'
    
    # Default: if it's a code file with description, assume content needed
    if any(lang in query_lower for lang in ['python', 'javascript', 'bash', 'script', 'program', 'c file', 'c program', 'java', 'cpp']):
        # Check if there's a description of what it should do
        if any(word in query_lower for word in ['which', 'that', 'to', 'for', 'checks', 'finds', 'calculates', 'prints', 'gets', 'shows']):
            return 'yes'
    
    return 'no'

def extract_content_description(query):
    """Extract what the file should do from the query using LLM."""
    # First try simple regex patterns
    match = re.search(r'(?:which|that)\s+(checks?|does|calculates?|prints?|finds?|sorts?|searches?)\s+(.+?)(?:\s+or\s+|\s+and\s+|$)', query, re.IGNORECASE)
    if match:
        action = match.group(1)
        target = match.group(2).strip()
        # Clean up common endings
        target = re.sub(r'\s+or\s+not.*$', '', target, flags=re.IGNORECASE)
        return f"{action} {target}"
    
    match = re.search(r'(?:to|for)\s+(check|calculate|print|find|sort|search)\s+(.+?)(?:\s+or\s+|\s+and\s+|$)', query, re.IGNORECASE)
    if match:
        action = match.group(1)
        target = match.group(2).strip()
        target = re.sub(r'\s+or\s+not.*$', '', target, flags=re.IGNORECASE)
        return f"{action} {target}"
    
    # If regex fails, use LLM
    console.print("[dim]🤖 Using LLM to extract purpose...[/dim]")
    
    prompt = f"""Extract a brief description of what this file should do:

Request: {query}

Provide ONLY a short phrase (3-8 words) describing the purpose. Examples:
- "create a python file to check if number is prime" → "check if number is prime"
- "make a file for odd or even numbers" → "determine if number is odd or even"
- "python script that calculates factorial" → "calculate factorial of a number"

Description:"""
    
    result = ollama_generate(prompt, max_tokens=30, temperature=0.3).strip()
    
    # Clean up
    result = result.split('\n')[0].strip()
    result = result.replace('"', '').replace("'", "").strip()
    result = result.lower()
    
    # Remove common prefixes
    result = re.sub(r'^(to|for|that|which)\s+', '', result)
    
    if result:
        return result
    
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

# ===============================
# UNIFIED CLARIFICATION SYSTEM
# ===============================

# Command clarification configuration
CLARIFIER_CONFIGS = {
    'delete': {
        'patterns': [r'\bdelete\b', r'\brm\b', r'\bremove\b'],
        'questions': [
            ('filename', 'File to delete', None, lambda q: extract_filename(q)),
            ('directory', 'Directory path where file is located', '.', 
             lambda q: next((m.group(1) for p, _ in [
                 (r'folder\s+(?:named|called)\s+([a-zA-Z0-9_\-\.]+)', 'folder'),
                 (r'directory\s+(?:named|called)\s+([a-zA-Z0-9_\-\.]+)', 'directory'),
                 (r'in\s+([a-zA-Z0-9_/\-\.]+)(?:\s+folder)?', 'path')
             ] if (m := re.search(p, q, re.IGNORECASE))), None)),
        ],
        'template': "Delete the file '{directory}/{filename}'"
    },
    'find': {
        'patterns': [r'\bfind\s+(?:files?|all)\b', r'\bsearch\s+(?:files?|all)\b', r'\blocate\s+(?:files?|all)\b'],
        'questions': [
            ('search_term', 'What are you looking for? (filename, pattern, or extension)', None, None),
            ('search_location', 'Where should I search? (directory path)', '.', None),
            ('file_type', 'Specific file type? (e.g., txt, py, or \'all\' for any)', 'all', 
             lambda q, st: 'detected' if ('*.' in st or '.' in st) else None),
        ],
        'template': "Find files matching '{search_term}' in '{search_location}' with type '{file_type}'"
    },
    'copy': {
        'patterns': [r'\bcopy\b', r'\bcp\b'],
        'questions': [
            ('source', 'Source file or directory path', None, lambda q: (m.group(1) if (m := re.search(r'\bfrom\s+([^\s]+)', q, re.IGNORECASE)) else None)),
            ('destination', 'Destination path', None, lambda q: (m.group(1) if (m := re.search(r'\bto\s+([^\s]+)', q, re.IGNORECASE)) else None)),
            ('recursive', 'Copy recursively? (yes/no)', 'yes', None, lambda q: 'recursively' if q == 'yes' else ''),
        ],
        'template': "Copy from '{source}' to '{destination}' {recursive}",
        'use_model_check': True
    },
    'create': {
        'patterns': [r'\bcreate\b', r'\btouch\b', r'\bmake.*file\b'],
        'questions': [
            ('filename', 'Filename/path (e.g., test.txt or path/to/file.txt)', None, 
             lambda q: extract_filename_from_create_query(q)),
            ('content', 'Add initial content? (yes/no)', 'no', 
             lambda q: detect_content_requirement(q)),
        ],
        'template': "Create {'file \'{filename}\' with content: {content_text}' if '{content}' == 'yes' else 'empty file \'{filename}\''}",
        'use_model_check': True
    },
    'grep': {
        'patterns': [r'\bgrep\b', r'\bsearch in\b', r'\bfind text\b'],
        'questions': [
            ('search_pattern', 'Text pattern to search for', None, None),
            ('search_location', 'File or directory to search in', '.', 
             lambda q: (m.group(1) if (m := re.search(r'\bin\s+([^\s]+)', q, re.IGNORECASE)) else None)),
            ('case_sensitive', 'Case sensitive? (yes/no)', 'yes', None),
        ],
        'template': "Search for '{search_pattern}' in '{search_location}' {'case sensitive' if '{case_sensitive}' == 'yes' else 'case insensitive'}"
    },
    'rename': {
        'patterns': [r'\brename\b', r'\bmv\b'],
        'questions': [
            ('old_name', 'Current file path', None, lambda q: (m.group(1) if (m := re.search(r'\bfrom\s+([^\s]+)', q, re.IGNORECASE)) else None)),
            ('new_name', 'New file path/name', None, lambda q: (m.group(1) if (m := re.search(r'\bto\s+([^\s]+)', q, re.IGNORECASE)) else None)),
        ],
        'template': "Rename '{old_name}' to '{new_name}'",
        'skip_if_complete': True
    },
    'permission': {
        'patterns': [r'\bchmod\b', r'\bpermission\b', r'\bmake.*executable\b'],
        'questions': [
            ('file_path', 'File or directory path', None, 
             lambda q: (m.group(1) if (m := re.search(r'(?:on|for|to)\s+([^\s]+)', q, re.IGNORECASE)) and '/' in m.group(1) else None)),
            ('permission', 'Permission mode (e.g., 755, 644, +x, u+rwx)', None,
             lambda q: (m.group(1) if (m := re.search(r'(?:to|as)\s+([0-7]{3}|[+\-][rwx]+)', q, re.IGNORECASE)) else None)),
            ('recursive', 'Apply recursively? (yes/no)', 'no', None, 
             lambda ans: 'recursively' if ans == 'yes' else '', 
             lambda q: 'directory' in q.lower() or 'folder' in q.lower() or q.endswith('/')),
        ],
        'template': "Change permissions of '{file_path}' to '{permission}' {recursive}"
    },
    'compress': {
        'patterns': [r'\bcompress\b', r'\bzip\b', r'\btar\b', r'\barchive\b'],
        'questions': [
            ('source', 'Files or directory to compress', None,
             lambda q: (m.group(1) if (m := re.search(r'(?:compress|archive)\s+([^\s]+)', q, re.IGNORECASE)) else None)),
            ('output', 'Output archive name (with extension)', None,
             lambda q: (m.group(1) if (m := re.search(r'(?:to|into|as)\s+([^\s]+)', q, re.IGNORECASE)) else None)),
            ('compression', 'Compression type (tar, zip, tar.gz, 7z)', 'tar.gz', None),
        ],
        'template': "Compress '{source}' to '{output}' using '{compression}' format"
    },
    'extract': {
        'patterns': [r'\bextract\b', r'\bunzip\b', r'\buntar\b'],
        'questions': [
            ('archive', 'Archive file path', None,
             lambda q: (m.group(1) if (m := re.search(r'(?:extract|unzip|untar)\s+([^\s]+)', q, re.IGNORECASE)) else None)),
            ('destination', 'Extract to directory (leave blank for current)', '.',
             lambda q: (m.group(1) if (m := re.search(r'(?:to|into)\s+([^\s]+)', q, re.IGNORECASE)) else None)),
        ],
        'template': "Extract '{archive}' to '{destination}'"
    },
}

def unified_clarify_command(query):
    """Unified clarification system for all command types."""
    for cmd_type, config in CLARIFIER_CONFIGS.items():
        # Check if this command type matches
        if not any(re.search(pattern, query, re.IGNORECASE) for pattern in config['patterns']):
            continue
        
        # Skip if all info already present (for certain command types)
        if config.get('skip_if_complete'):
            all_extracted = True
            for q_name, q_text, q_default, extractor, *_ in config['questions']:
                if extractor and not extractor(query):
                    all_extracted = False
                    break
            if all_extracted:
                return query, False
        
        console.print(f"\n[bold yellow]ℹ️  {cmd_type.title()} command detected[/bold yellow]")
        
        answers = {}
        question_num = 1
        questions_asked = False  # Track if we actually ask any questions
        
        # Process each question
        for question_info in config['questions']:
            q_name, q_text, q_default, extractor = question_info[:4]
            transformer = question_info[4] if len(question_info) > 4 else None
            condition = question_info[5] if len(question_info) > 5 else None
            
            # Check condition if specified
            if condition and not condition(query):
                if q_default:
                    answers[q_name] = q_default
                    console.print(f"[dim]✓ Using default: {q_default}[/dim]")
                continue
            
            # Try to extract from query first
            extracted = None
            if extractor:
                try:
                    extracted = extractor(query) if q_name not in answers else extractor(query, answers.get('search_term', ''))
                except:
                    pass
            
            if extracted:
                answers[q_name] = extracted
                console.print(f"[dim]✓ {q_name.replace('_', ' ').title()}: {extracted}[/dim]")
            else:
                # Ask user
                if not questions_asked:
                    console.print("[yellow]Asking clarifying questions:[/yellow]\n")
                    questions_asked = True
                
                if q_default:
                    answer = Prompt.ask(f"[cyan]Q{question_num}: {q_text}[/cyan]", default=q_default).strip()
                else:
                    answer = Prompt.ask(f"[cyan]Q{question_num}: {q_text}[/cyan]").strip()
                    if not answer:
                        return query, False
                
                # Apply transformer if specified
                if transformer:
                    answer = transformer(answer)
                
                answers[q_name] = answer
                question_num += 1
        
        # Build clarified query from template
        template = config['template']
        
        # Handle complex template logic
        if 'content' in answers and answers['content'] == 'yes':
            # Try to extract content description from query
            content_text = extract_content_description(query)
            if not content_text:
                content_text = Prompt.ask("[cyan]Initial content[/cyan]").strip()
            else:
                console.print(f"[dim]✓ Detected purpose: {content_text}[/dim]")
            answers['content_text'] = content_text
            clarified_query = f"Create file '{answers['filename']}' with content: {content_text}"
        else:
            # Handle templates with simple placeholders
            # Replace template placeholders manually to avoid format string issues
            clarified_query = template
            for key, value in answers.items():
                clarified_query = clarified_query.replace(f"{{{key}}}", str(value))
            
            # Clean up any remaining template syntax
            clarified_query = re.sub(r"\{[^}]+\}", "", clarified_query)
            
            # If it's a create command without content, simplify
            if 'filename' in answers:
                clarified_query = f"Create empty file '{answers['filename']}'"

        
        # Model sufficiency check if configured
        if config.get('use_model_check'):
            is_sufficient, missing_info = evaluate_information_sufficiency(clarified_query)
            if not is_sufficient:
                console.print(f"[yellow]Model says: {missing_info}[/yellow]")
        
        console.print(f"\n[green]✓ {clarified_query}[/green]\n")
        return clarified_query, True
    
    return query, False

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
                timeout=260
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
        
        # Use unified clarification system
        clarified_query, was_clarified = unified_clarify_command(query)
        if was_clarified:
            query = clarified_query
        
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