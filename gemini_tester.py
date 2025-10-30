"""
Gemini API-based CLI Agent Tester

This script tests the CLI agent by:
1. Reading test cases from datasets/test.csv
2. Generating commands using the CLI agent
3. Comparing generated commands with expected commands
4. Using Gemini API to evaluate semantic accuracy
"""

import os
import csv
import json
import time
import requests
from typing import Dict, List, Tuple
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
from rich.panel import Panel

# Import the CLI agent components
from cli_agent import embedder, retrieve, ollama_generate, sanitize_bash, generate_bash

console = Console()

# ===============================
# GEMINI API CONFIGURATION
# ===============================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyC4oPZFMEw18uUSUbXK_ZmWz4CMAbPMZ5Y")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent"

# ===============================
# GEMINI API FUNCTIONS
# ===============================
def call_gemini_api(prompt: str, temperature: float = 0.3, max_tokens: int = 500) -> str:
    """Call Gemini API with the given prompt."""
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable not set!")
    
    headers = {
        "Content-Type": "application/json"
    }
    
    payload = {
        "contents": [{
            "parts": [{
                "text": prompt
            }]
        }],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        }
    }
    
    try:
        response = requests.post(
            f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        
        result = response.json()
        
        # Extract text from Gemini response
        if "candidates" in result and len(result["candidates"]) > 0:
            candidate = result["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                parts = candidate["content"]["parts"]
                if len(parts) > 0 and "text" in parts[0]:
                    return parts[0]["text"].strip()
        
        console.print("[red]⚠️ Unexpected Gemini API response format[/red]")
        return ""
        
    except requests.exceptions.RequestException as e:
        console.print(f"[bold red]Error calling Gemini API: {e}[/bold red]")
        return ""
    except Exception as e:
        console.print(f"[bold red]Unexpected error: {e}[/bold red]")
        return ""

def evaluate_batch_accuracy(test_results: List[Dict]) -> Dict:
    """
    Use Gemini to evaluate all test results in a single batch.
    Provides overall accuracy assessment instead of individual scores.
    
    Returns a dict with:
    - overall_accuracy: str (percentage or description)
    - analysis: str (detailed analysis)
    - correct_count: int
    - total_count: int
    """
    # Build comprehensive prompt with all test cases
    test_summary = []
    for i, result in enumerate(test_results, 1):
        test_summary.append(f"""
Test Case #{i}:
Natural Language Query: {result['nl_query']}
Expected Command: {result['expected']}
{f"Alternative Expected: {result['expected_alt']}" if result.get('expected_alt') else ""}
Generated Command: {result['generated']}
---""")
    
    all_tests = "\n".join(test_summary)
    
    prompt = f"""You are an expert Linux/Bash command evaluator. Analyze these test results from a CLI agent that generates bash commands from natural language.

Below are {len(test_results)} test cases showing:
1. What the user asked for (Natural Language Query)
2. What command was expected (Expected Command)
3. What the CLI agent generated (Generated Command)

{all_tests}

Please evaluate:
1. How accurately does the CLI agent provide the required output?
2. For each test, does the generated command achieve the same goal as expected?
3. What is the overall accuracy percentage?

IMPORTANT:
- Different commands can be equally correct (e.g., 'ls -a' vs 'ls -la')
- Focus on whether commands achieve the GOAL, not exact syntax matching
- Minor flag differences are OK if result is the same
- Consider both expected and alternative commands as correct

Provide your analysis in this EXACT JSON format:
{{
  "overall_accuracy": "<percentage like '85%' or '17/20'>",
  "correct_count": <number of correct commands>,
  "total_count": {len(test_results)},
  "analysis": "<detailed analysis of patterns, strengths, weaknesses>",
  "per_test_results": [
    {{
      "test_num": 1,
      "is_correct": true/false,
      "brief_note": "<one line explanation>"
    }},
    ...
  ]
}}

Output ONLY the JSON, nothing else."""

    response = call_gemini_api(prompt, temperature=0.2, max_tokens=2000)
    
    # Parse JSON response
    try:
        # Extract JSON from response
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            json_str = response[json_start:json_end]
            result = json.loads(json_str)
            
            # Validate required fields
            if "overall_accuracy" in result and "correct_count" in result:
                return result
    except (json.JSONDecodeError, ValueError) as e:
        console.print(f"[yellow]⚠️ Failed to parse Gemini response as JSON: {e}[/yellow]")
        console.print(f"[dim]Response was: {response[:300]}...[/dim]")
    
    # Fallback evaluation
    return {
        "overall_accuracy": "Unable to evaluate",
        "correct_count": 0,
        "total_count": len(test_results),
        "analysis": "Failed to parse Gemini evaluation response",
        "per_test_results": []
    }

# ===============================
# TEST EXECUTION FUNCTIONS
# ===============================
def load_test_cases(csv_path: str, limit: int = None) -> List[Dict]:
    """Load test cases from CSV file."""
    test_cases = []
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if limit and i >= limit:
                    break
                test_cases.append({
                    'nl': row['nl'],
                    'bash': row['bash'],
                    'bash2': row.get('bash2', ''),
                    'difficulty': int(row.get('difficulty', 0))
                })
    except FileNotFoundError:
        console.print(f"[bold red]Error: Test file not found: {csv_path}[/bold red]")
        return []
    except Exception as e:
        console.print(f"[bold red]Error loading test cases: {e}[/bold red]")
        return []
    
    return test_cases

def generate_command_from_nl(nl_query: str) -> Tuple[str, bool]:
    """
    Generate command using the CLI agent.
    
    Returns:
    - (generated_command, success)
    """
    try:
        # Use the CLI agent's generate_bash function
        bash_cmds, rag_examples = generate_bash(nl_query, error_context=None, show_rag_debug=False)
        
        if bash_cmds and bash_cmds.strip():
            return bash_cmds.strip(), True
        else:
            return "", False
            
    except Exception as e:
        console.print(f"[red]Error generating command: {e}[/red]")
        return "", False

def run_test_suite(test_cases: List[Dict], detailed: bool = False) -> Dict:
    """
    Run the complete test suite and return results.
    First generates all commands, then evaluates in batch with Gemini.
    
    Returns dict with:
    - total: int
    - passed: int
    - failed: int
    - overall_accuracy: str
    - gemini_evaluation: dict
    - results: list of detailed results
    """
    # Step 1: Generate all commands
    console.print("\n[bold cyan]Step 1: Generating commands...[/bold cyan]")
    test_results = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Generating...", total=len(test_cases))
        
        for i, test_case in enumerate(test_cases, 1):
            progress.update(task, description=f"[cyan]Generating {i}/{len(test_cases)}: {test_case['nl'][:50]}...")
            
            # Generate command
            generated_cmd, gen_success = generate_command_from_nl(test_case['nl'])
            
            test_results.append({
                'test_num': i,
                'nl_query': test_case['nl'],
                'expected': test_case['bash'],
                'expected_alt': test_case.get('bash2', ''),
                'generated': generated_cmd if gen_success else "(generation failed)",
                'generation_success': gen_success,
                'difficulty': test_case['difficulty']
            })
            
            progress.advance(task)
    
    console.print(f"[green]✓ Generated {len(test_results)} commands[/green]")
    
    # Step 2: Batch evaluate with Gemini
    console.print("\n[bold cyan]Step 2: Evaluating with Gemini AI...[/bold cyan]")
    console.print("[dim]Sending all test results to Gemini for accuracy analysis...[/dim]\n")
    
    gemini_evaluation = evaluate_batch_accuracy(test_results)
    
    # Step 3: Combine results
    results = {
        'total': len(test_cases),
        'passed': gemini_evaluation.get('correct_count', 0),
        'failed': len(test_cases) - gemini_evaluation.get('correct_count', 0),
        'generation_failed': sum(1 for r in test_results if not r['generation_success']),
        'overall_accuracy': gemini_evaluation.get('overall_accuracy', 'N/A'),
        'gemini_analysis': gemini_evaluation.get('analysis', ''),
        'results': []
    }
    
    # Merge Gemini per-test results with our test results
    per_test_eval = gemini_evaluation.get('per_test_results', [])
    
    for i, test_result in enumerate(test_results):
        # Find corresponding Gemini evaluation
        gemini_result = None
        for eval_item in per_test_eval:
            if eval_item.get('test_num') == i + 1:
                gemini_result = eval_item
                break
        
        results['results'].append({
            **test_result,
            'is_correct': gemini_result.get('is_correct', False) if gemini_result else False,
            'gemini_note': gemini_result.get('brief_note', '') if gemini_result else 'No evaluation'
        })
    
    return results

# ===============================
# RESULT DISPLAY FUNCTIONS
# ===============================
def display_summary(results: Dict):
    """Display test summary."""
    console.print("\n" + "=" * 80)
    console.print("[bold cyan]TEST SUITE SUMMARY[/bold cyan]")
    console.print("=" * 80 + "\n")
    
    # Create summary table
    summary_table = Table(show_header=True, header_style="bold magenta")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", justify="right", style="yellow")
    
    summary_table.add_row("Total Tests", str(results['total']))
    summary_table.add_row("Passed", f"[green]{results['passed']}[/green]")
    summary_table.add_row("Failed", f"[red]{results['failed']}[/red]")
    summary_table.add_row("Generation Failed", f"[red]{results['generation_failed']}[/red]")
    summary_table.add_row("Overall Accuracy", f"[bold]{results.get('overall_accuracy', 'N/A')}[/bold]")
    
    # Calculate pass rate
    if results['total'] > 0:
        pass_rate = (results['passed'] / results['total']) * 100
        summary_table.add_row("Pass Rate", f"{pass_rate:.1f}%")
    
    console.print(summary_table)
    
    # Display Gemini's analysis
    if results.get('gemini_analysis'):
        console.print("\n[bold yellow]Gemini AI Analysis:[/bold yellow]")
        console.print(Panel(results['gemini_analysis'], border_style="yellow", expand=False))
    
    console.print()

def display_detailed_results(results: Dict, show_all: bool = False, show_failed_only: bool = False):
    """Display detailed test results."""
    console.print("\n[bold cyan]DETAILED RESULTS[/bold cyan]\n")
    
    for result in results['results']:
        # Filter based on options
        if show_failed_only and result.get('is_correct', False):
            continue
        
        # Color based on result
        if result.get('is_correct', False):
            border_style = "green"
            status_icon = "✅"
        else:
            border_style = "red"
            status_icon = "❌"
        
        # Build panel content
        # Handle alternative command separately to avoid f-string backslash issue
        alt_cmd_section = ""
        if result.get('expected_alt'):
            alt_cmd_section = f"\n[cyan]Alternative:[/cyan]\n{result['expected_alt']}"
        
        content = f"""[bold]Test #{result['test_num']}[/bold] {status_icon}
[yellow]Status:[/yellow] {'✓ Correct' if result.get('is_correct') else '✗ Incorrect'}
[yellow]Difficulty:[/yellow] {result['difficulty']}

[cyan]Natural Language Query:[/cyan]
{result['nl_query']}

[cyan]Expected Command:[/cyan]
{result['expected']}{alt_cmd_section}

[cyan]Generated Command:[/cyan]
{result['generated']}

[cyan]Gemini Note:[/cyan]
{result.get('gemini_note', 'No evaluation available')}"""
        
        panel = Panel(content, border_style=border_style, expand=False)
        console.print(panel)
        
        if not show_all:
            # Ask to continue for each result
            if result != results['results'][-1]:  # Not last result
                if not Confirm.ask("\n[dim]Show next result?[/dim]", default=True):
                    break

def save_results_to_file(results: Dict, output_path: str):
    """Save test results to a JSON file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        console.print(f"\n[green]✅ Results saved to: {output_path}[/green]")
    except Exception as e:
        console.print(f"[red]❌ Error saving results: {e}[/red]")

# ===============================
# MAIN FUNCTION
# ===============================
def main():
    """Main function to run the tester."""
    console.print(Panel.fit(
        "[bold cyan]Gemini API-based CLI Agent Tester[/bold cyan]\n"
        "[dim]Tests CLI agent command generation using Gemini for evaluation[/dim]",
        border_style="cyan"
    ))
    
    # Check for API key
    if not GEMINI_API_KEY:
        console.print("\n[bold red]❌ ERROR: GEMINI_API_KEY environment variable not set![/bold red]")
        console.print("\n[yellow]Please set your Gemini API key:[/yellow]")
        console.print("  Windows (PowerShell): $env:GEMINI_API_KEY = 'your-key-here'")
        console.print("  Linux/Mac: export GEMINI_API_KEY='your-key-here'")
        return
    
    console.print("\n[green]✅ Gemini API key detected[/green]")
    
    # Configuration
    csv_path = "datasets/test.csv"
    
    console.print(f"\n[cyan]Test file:[/cyan] {csv_path}")
    
    # Ask how many tests to run
    console.print("\n[bold yellow]Test Configuration:[/bold yellow]")
    num_tests = Prompt.ask(
        "How many test cases to run?",
        default="10"
    )
    
    try:
        num_tests = int(num_tests)
    except ValueError:
        num_tests = 10
    
    # Load test cases
    console.print(f"\n[cyan]Loading test cases...[/cyan]")
    test_cases = load_test_cases(csv_path, limit=num_tests)
    
    if not test_cases:
        console.print("[bold red]No test cases loaded. Exiting.[/bold red]")
        return
    
    console.print(f"[green]✅ Loaded {len(test_cases)} test cases[/green]")
    
    # Confirm to proceed
    if not Confirm.ask(f"\n[bold]Run {len(test_cases)} tests?[/bold]", default=True):
        console.print("[yellow]Cancelled.[/yellow]")
        return
    
    # Run tests
    console.print("\n[bold cyan]Starting test suite...[/bold cyan]\n")
    results = run_test_suite(test_cases, detailed=False)
    
    # Display summary
    display_summary(results)
    
    # Ask about detailed results
    show_details = Prompt.ask(
        "\n[bold]Show detailed results?[/bold]",
        choices=["all", "failed", "no"],
        default="failed"
    )
    
    if show_details == "all":
        display_detailed_results(results, show_all=True, show_failed_only=False)
    elif show_details == "failed":
        display_detailed_results(results, show_all=True, show_failed_only=True)
    
    # Save results
    if Confirm.ask("\n[bold]Save results to JSON file?[/bold]", default=True):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = f"test_results_{timestamp}.json"
        save_results_to_file(results, output_path)
    
    console.print("\n[bold green]✅ Testing complete![/bold green]\n")

if __name__ == "__main__":
    main()
