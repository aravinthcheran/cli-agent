"""Command clarification system for interactive user input."""
import re
from rich.console import Console
from rich.prompt import Prompt
from src.utils import extract_filename
from src.ollama_client import evaluate_information_sufficiency

console = Console()

# Cache for LLM extraction results to avoid duplicate calls
_delete_info_cache = {}


def extract_delete_info_from_query(query):
    """Extract filename and directory from delete queries using LLM."""
    from src.ollama_client import ollama_generate
    
    # Check cache first
    if query in _delete_info_cache:
        return _delete_info_cache[query]
    
    console.print("[dim]🤖 Analyzing delete request with LLM...[/dim]")
    
    prompt = f"""Analyze this file deletion request and extract the filename and directory path.

Request: {query}

Extract and return ONLY in this exact format (one line each):
FILENAME: <filename with extension, or UNKNOWN if not specified>
DIRECTORY: <directory path, or UNKNOWN if not specified>

Rules:
1. Look for explicit filenames mentioned (e.g., "user_greeter.c", "test.txt")
2. Look for directory/path mentions (e.g., "in src/", "from /home/user/", "in current directory")
3. If directory is mentioned as "current" or not specified, use "."
4. If information is not clear or missing, return UNKNOWN for that field
5. Return ONLY the two lines in the format above, nothing else

Examples:
- "delete file named test.txt" → FILENAME: test.txt, DIRECTORY: .
- "remove user_greeter.c from src folder" → FILENAME: user_greeter.c, DIRECTORY: src
- "delete the file" → FILENAME: UNKNOWN, DIRECTORY: UNKNOWN
- "rm temp.log in /var/log" → FILENAME: temp.log, DIRECTORY: /var/log

Now analyze:
Request: {query}

Response:"""
    
    result = ollama_generate(prompt, max_tokens=50, temperature=0.1).strip()
    
    # Parse the LLM response
    filename = None
    directory = None
    
    for line in result.split('\n'):
        line = line.strip()
        if line.startswith('FILENAME:'):
            extracted = line.replace('FILENAME:', '').strip()
            if extracted and extracted.upper() != 'UNKNOWN':
                filename = extracted
                console.print(f"[dim]✓ LLM found filename: {filename}[/dim]")
        elif line.startswith('DIRECTORY:'):
            extracted = line.replace('DIRECTORY:', '').strip()
            if extracted and extracted.upper() != 'UNKNOWN':
                directory = extracted
                console.print(f"[dim]✓ LLM found directory: {directory}[/dim]")
    
    # Cache the results
    _delete_info_cache[query] = (filename, directory)
    
    return filename, directory


def re_clarify_delete_with_error(original_query, error_message):
    """Re-clarify delete operation when file is not found."""
    from src.ollama_client import ollama_generate
    
    console.print("\n[bold yellow]🔍 File not found - Re-clarifying filename/location...[/bold yellow]")
    
    # Clear the cache for this query
    if original_query in _delete_info_cache:
        del _delete_info_cache[original_query]
    
    prompt = f"""The user wanted to delete a file but got an error. Help determine what went wrong.

Original request: {original_query}
Error: {error_message}

The file was not found. This could mean:
1. The filename was extracted incorrectly
2. The directory path is wrong
3. The file doesn't exist

Analyze the error and original request. Return ONLY in this format:
ISSUE: <brief description of the likely problem>
SUGGESTION: <what to ask the user>

Example:
ISSUE: File path may be incorrect or file doesn't exist
SUGGESTION: Verify the filename and directory path

Response:"""
    
    result = ollama_generate(prompt, max_tokens=100, temperature=0.3).strip()
    
    # Parse the response
    issue = None
    suggestion = None
    
    for line in result.split('\n'):
        line = line.strip()
        if line.startswith('ISSUE:'):
            issue = line.replace('ISSUE:', '').strip()
        elif line.startswith('SUGGESTION:'):
            suggestion = line.replace('SUGGESTION:', '').strip()
    
    if issue:
        console.print(f"[yellow]💡 {issue}[/yellow]")
    
    # Ask user to re-confirm filename and directory
    console.print("\n[cyan]Let's verify the file details:[/cyan]\n")
    
    filename = Prompt.ask("[cyan]Exact filename (with extension)[/cyan]").strip()
    if not filename:
        return None
    
    directory = Prompt.ask("[cyan]Directory path (. for current)[/cyan]", default=".").strip()
    
    # Build new clarified query
    clarified_query = f"Delete the file '{directory}/{filename}'"
    
    console.print(f"\n[green]✓ {clarified_query}[/green]\n")
    
    return clarified_query


def handle_file_not_found_error(command, error_message, original_query=None):
    """Generic handler for file not found errors in any command."""
    from src.ollama_client import ollama_generate
    import os
    
    console.print("\n[bold yellow]⚠️  File not found error detected[/bold yellow]")
    
    # Extract filename from command or error message
    prompt = f"""Extract the filename that caused this "file not found" error.

Command: {command}
Error: {error_message}

Return ONLY the filename that was not found, nothing else.
Examples:
- "grep: test.txt: No such file or directory" → test.txt
- "cat: /path/to/file.py: No such file or directory" → /path/to/file.py
- "rm: cannot remove 'data.csv': No such file or directory" → data.csv

Filename:"""
    
    result = ollama_generate(prompt, max_tokens=30, temperature=0.1).strip()
    
    # Clean up the result
    missing_file = result.replace('"', '').replace("'", "").strip()
    
    console.print(f"[yellow]📂 Could not find: {missing_file}[/yellow]")
    
    # Try to find similar files in current directory
    if os.path.exists('.'):
        files_in_dir = []
        try:
            for root, dirs, files in os.walk('.'):
                # Only search in immediate subdirectories and current dir
                depth = root.replace('.', '').count(os.sep)
                if depth < 2:
                    files_in_dir.extend([os.path.join(root, f) for f in files])
        except:
            files_in_dir = []
        
        # Find similar filenames
        base_name = os.path.basename(missing_file)
        similar_files = [f for f in files_in_dir if base_name.lower() in f.lower() or 
                        os.path.basename(f).lower() in base_name.lower()]
        
        if similar_files:
            console.print(f"\n[cyan]💡 Found similar files:[/cyan]")
            for i, f in enumerate(similar_files[:5], 1):
                console.print(f"  {i}. {f}")
            
            choice = Prompt.ask(
                f"\n[cyan]Select file number (1-{min(5, len(similar_files))}) or enter new path[/cyan]",
                default="1"
            ).strip()
            
            # Check if user entered a number
            if choice.isdigit() and 1 <= int(choice) <= len(similar_files):
                correct_file = similar_files[int(choice) - 1]
            else:
                correct_file = choice
        else:
            console.print("\n[yellow]No similar files found in current directory[/yellow]")
            correct_file = Prompt.ask("[cyan]Enter correct file path[/cyan]").strip()
    else:
        correct_file = Prompt.ask("[cyan]Enter correct file path[/cyan]").strip()
    
    if not correct_file:
        return None
    
    # Replace the missing file in the command
    corrected_command = command.replace(missing_file, correct_file)
    
    console.print(f"\n[green]✓ Corrected command: {corrected_command}[/green]\n")
    
    return corrected_command


def extract_filename_from_create_query(query):
    """Extract filename intelligently from create file queries using LLM."""
    from src.ollama_client import ollama_generate
    
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
    from src.ollama_client import ollama_generate
    
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


# Command clarification configuration
CLARIFIER_CONFIGS = {
    'delete': {
        'patterns': [r'\bdelete\b', r'\brm\b', r'\bremove\b'],
        'questions': [
            ('filename', 'File to delete', None, lambda q: extract_delete_info_from_query(q)[0]),
            ('directory', 'Directory path where file is located', '.', lambda q: extract_delete_info_from_query(q)[1]),
        ],
        'template': "Delete the file '{directory}/{filename}'",
        'use_model_check': True
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
        
        # Handle complex template logic for CREATE commands
        if cmd_type == 'create' and 'content' in answers and answers['content'] == 'yes':
            # Try to extract content description from query
            content_text = extract_content_description(query)
            if not content_text:
                content_text = Prompt.ask("[cyan]Initial content[/cyan]").strip()
            else:
                console.print(f"[dim]✓ Detected purpose: {content_text}[/dim]")
            answers['content_text'] = content_text
            clarified_query = f"Create file '{answers['filename']}' with content: {content_text}"
        elif cmd_type == 'create' and 'filename' in answers:
            # Create command without content, simplify
            clarified_query = f"Create empty file '{answers['filename']}'"
        else:
            # Handle templates with simple placeholders for other command types
            # Replace template placeholders manually to avoid format string issues
            clarified_query = template
            for key, value in answers.items():
                clarified_query = clarified_query.replace(f"{{{key}}}", str(value))
            
            # Clean up any remaining template syntax
            clarified_query = re.sub(r"\{[^}]+\}", "", clarified_query)

        
        # Model sufficiency check if configured
        if config.get('use_model_check'):
            is_sufficient, missing_info = evaluate_information_sufficiency(clarified_query)
            if not is_sufficient:
                console.print(f"[yellow]Model says: {missing_info}[/yellow]")
        
        console.print(f"\n[green]✓ {clarified_query}[/green]\n")
        return clarified_query, True
    
    return query, False
