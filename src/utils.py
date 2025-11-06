"""Utility functions for the CLI agent."""
import re
from rich.console import Console

console = Console()


def debug_print(message: str, style: str = "dim"):
    """Print debug messages only when DEBUG_MODE is enabled."""
    from src.config import DEBUG_MODE
    if DEBUG_MODE:
        console.print(f"[{style}]{message}[/{style}]")


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


def extract_filename(query):
    """Extract filename from query if present."""
    # Look for common filename patterns
    filename_pattern = r'(?:delete|remove|rm|del)\s+(?:the\s+)?(?:file\s+)?([a-zA-Z0-9_\-\.]+(?:\.[a-zA-Z0-9]+)?)'
    match = re.search(filename_pattern, query, re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def check_dangerous_commands(command_str):
    """Check if commands contain dangerous patterns."""
    from src.config import DANGEROUS_PATTERNS
    dangers = []
    for pattern, description in DANGEROUS_PATTERNS:
        if re.search(pattern, command_str, re.IGNORECASE):
            dangers.append(description)
    return dangers
