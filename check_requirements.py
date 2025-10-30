"""
Requirements checker for Gemini Tester
Validates that all dependencies and files are in place
"""

import sys
import os
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

def check_requirement(name, check_func, fix_hint=""):
    """Check a requirement and return status."""
    try:
        result = check_func()
        if result:
            console.print(f"[green]✓[/green] {name}")
            return True
        else:
            console.print(f"[red]✗[/red] {name}")
            if fix_hint:
                console.print(f"  [yellow]Fix:[/yellow] {fix_hint}")
            return False
    except Exception as e:
        console.print(f"[red]✗[/red] {name}")
        console.print(f"  [red]Error:[/red] {e}")
        if fix_hint:
            console.print(f"  [yellow]Fix:[/yellow] {fix_hint}")
        return False

def main():
    console.print(Panel.fit(
        "[bold cyan]Gemini Tester - Requirements Check[/bold cyan]\n"
        "[dim]Validating setup before running tests[/dim]",
        border_style="cyan"
    ))
    
    checks = []
    
    # Python version
    console.print("\n[bold]Python Environment[/bold]")
    checks.append(check_requirement(
        "Python 3.8+",
        lambda: sys.version_info >= (3, 8),
        "Install Python 3.8 or higher"
    ))
    
    # Required files
    console.print("\n[bold]Required Files[/bold]")
    checks.append(check_requirement(
        "cli_agent.py exists",
        lambda: Path("cli_agent.py").exists(),
        "Make sure you're in the project directory"
    ))
    checks.append(check_requirement(
        "gemini_tester.py exists",
        lambda: Path("gemini_tester.py").exists(),
        "Run this from the project root directory"
    ))
    checks.append(check_requirement(
        "datasets/test.csv exists",
        lambda: Path("datasets/test.csv").exists(),
        "Ensure test data file is present"
    ))
    checks.append(check_requirement(
        "FAISS index exists",
        lambda: Path("bash_commands_l2.bin").exists() or Path("metadata_l2.npz").exists(),
        "Build FAISS index first"
    ))
    
    # Python packages
    console.print("\n[bold]Python Packages[/bold]")
    
    def check_package(pkg_name):
        try:
            __import__(pkg_name)
            return True
        except ImportError:
            return False
    
    checks.append(check_requirement(
        "requests",
        lambda: check_package("requests"),
        "pip install requests"
    ))
    checks.append(check_requirement(
        "rich",
        lambda: check_package("rich"),
        "pip install rich"
    ))
    checks.append(check_requirement(
        "sentence-transformers",
        lambda: check_package("sentence_transformers"),
        "pip install sentence-transformers"
    ))
    checks.append(check_requirement(
        "faiss",
        lambda: check_package("faiss"),
        "pip install faiss-cpu"
    ))
    checks.append(check_requirement(
        "numpy",
        lambda: check_package("numpy"),
        "pip install numpy"
    ))
    
    # Environment variables
    console.print("\n[bold]Environment Variables[/bold]")
    checks.append(check_requirement(
        "GEMINI_API_KEY set",
        lambda: bool(os.environ.get("GEMINI_API_KEY")),
        "Set: $env:GEMINI_API_KEY = 'your-key'"
    ))
    
    # External services
    console.print("\n[bold]External Services[/bold]")
    
    def check_ollama():
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    checks.append(check_requirement(
        "Ollama running",
        check_ollama,
        "Start Ollama: ollama serve"
    ))
    
    def check_gemini_api():
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return False
        try:
            import requests
            url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except:
            return False
    
    checks.append(check_requirement(
        "Gemini API accessible",
        check_gemini_api,
        "Check API key and internet connection"
    ))
    
    # Summary
    console.print("\n" + "=" * 80)
    passed = sum(checks)
    total = len(checks)
    
    if passed == total:
        console.print(f"\n[bold green]✓ All checks passed! ({passed}/{total})[/bold green]")
        console.print("\n[cyan]You're ready to run the tester:[/cyan]")
        console.print("  [bold]python gemini_tester.py[/bold]")
        console.print("\nOr use the quick start:")
        console.print("  [bold].\\run_tester.ps1[/bold]")
    elif passed >= total * 0.7:
        console.print(f"\n[yellow]⚠ Most checks passed ({passed}/{total})[/yellow]")
        console.print("\n[yellow]You can proceed, but some features may not work.[/yellow]")
        console.print("Review the failed checks above.")
    else:
        console.print(f"\n[red]✗ Several checks failed ({passed}/{total})[/red]")
        console.print("\n[red]Please fix the issues above before running the tester.[/red]")
    
    console.print("\n" + "=" * 80 + "\n")
    
    # Detailed help
    if passed < total:
        console.print("[bold]Common Fixes:[/bold]\n")
        
        if not os.environ.get("GEMINI_API_KEY"):
            console.print("[yellow]1. Set Gemini API Key:[/yellow]")
            console.print("   PowerShell: $env:GEMINI_API_KEY = 'your-key'")
            console.print("   Get key: https://makersuite.google.com/app/apikey\n")
        
        if not check_ollama():
            console.print("[yellow]2. Start Ollama:[/yellow]")
            console.print("   ollama serve\n")
        
        console.print("[yellow]3. Install Missing Packages:[/yellow]")
        console.print("   pip install -r requirements.txt\n")

if __name__ == "__main__":
    main()
