"""Command generation, execution, and dependency analysis."""
import re
import subprocess
from rich.console import Console
from src.config import COMMAND_TIMEOUT, DEBUG_MODE
from src.utils import debug_print, sanitize_bash
from src.rag import retrieve, display_rag_examples
from src.ollama_client import ollama_generate

console = Console()


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
                timeout=COMMAND_TIMEOUT
            )
            
            output = (result.stdout + result.stderr).strip()
            outputs.append(f"$ {display_cmd}\n{output or '(success)'}")
            
            # Check for errors - either non-zero return code OR error messages in stderr
            # (piped commands may return 0 even if early commands fail)
            has_file_error = 'no such file or directory' in result.stderr.lower()
            
            if result.returncode != 0 or has_file_error:
                if has_file_error:
                    console.print(f"[bold red]⚠️ Command {i+1} failed: File not found error detected[/bold red]")
                else:
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
