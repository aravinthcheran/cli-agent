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
import re
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
            timeout=120  # Increased to 120 seconds for large batches
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

def evaluate_batch_accuracy(test_results: List[Dict], batch_size: int = 50) -> Dict:
    """
    Use Gemini to evaluate test results in batches to avoid timeouts.
    Processes tests in chunks and combines results.
    
    Args:
        test_results: List of test result dictionaries
        batch_size: Number of tests to evaluate per API call (default: 50)
    
    Returns a dict with:
    - overall_accuracy: str (percentage or description)
    - analysis: str (detailed analysis)
    - correct_count: int
    - total_count: int
    - per_test_results: list of per-test evaluations
    """
    total_tests = len(test_results)
    
    # If tests fit in one batch, process normally
    if total_tests <= batch_size:
        return _evaluate_single_batch(test_results, start_index=0)
    
    # Process in batches
    console.print(f"[yellow]Processing {total_tests} tests in batches of {batch_size}...[/yellow]")
    
    all_per_test_results = []
    total_correct = 0
    batch_analyses = []
    
    num_batches = (total_tests + batch_size - 1) // batch_size  # Ceiling division
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Evaluating batches...", total=num_batches)
        
        for batch_num in range(num_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, total_tests)
            batch = test_results[start_idx:end_idx]
            
            progress.update(task, description=f"[cyan]Evaluating batch {batch_num + 1}/{num_batches} (tests {start_idx + 1}-{end_idx})...")
            
            # Evaluate this batch
            batch_result = _evaluate_single_batch(batch, start_index=start_idx)
            
            # Accumulate results
            total_correct += batch_result.get('correct_count', 0)
            all_per_test_results.extend(batch_result.get('per_test_results', []))
            
            if batch_result.get('analysis'):
                batch_analyses.append(f"Batch {batch_num + 1}: {batch_result['analysis']}")
            
            progress.advance(task)
            
            # Small delay between batches to avoid rate limiting
            if batch_num < num_batches - 1:
                time.sleep(1)
    
    # Combine all results
    overall_accuracy = f"{total_correct}/{total_tests} ({(total_correct/total_tests)*100:.1f}%)"
    combined_analysis = "\n\n".join(batch_analyses) if batch_analyses else "Batch evaluation completed"
    
    return {
        "overall_accuracy": overall_accuracy,
        "correct_count": total_correct,
        "total_count": total_tests,
        "analysis": combined_analysis,
        "per_test_results": all_per_test_results
    }

def _evaluate_single_batch(test_results: List[Dict], start_index: int = 0) -> Dict:
    """
    Evaluate a single batch of test results with Gemini.
    
    Args:
        test_results: List of test result dictionaries for this batch
        start_index: Starting test number for proper numbering
    
    Returns evaluation dict for this batch
    """
    # Build comprehensive prompt with all test cases
    test_summary = []
    for i, result in enumerate(test_results, 1):
        actual_test_num = start_index + i
        test_summary.append(f"""
Test Case #{actual_test_num}:
Natural Language Query: {result['nl_query']}
Expected Command: {result['expected']}
{f"Alternative Expected: {result['expected_alt']}" if result.get('expected_alt') else ""}
Generated Command: {result['generated']}
---""")
    
    all_tests = "\n".join(test_summary)
    
    prompt = f"""You are an expert Linux/Bash command evaluator. Analyze these test results from a CLI agent that generates bash commands from natural language.

Below are {len(test_results)} test cases (starting from test #{start_index + 1}) showing:
1. What the user asked for (Natural Language Query)
2. What command was expected (Expected Command)
3. What the CLI agent generated (Generated Command)

{all_tests}

CRITICAL EVALUATION CRITERIA:
Your PRIMARY focus is: **Does the generated command accomplish the task described in the Natural Language Query?**

Evaluation Guidelines:
1. **Functional Equivalence is KEY**: If the generated command achieves the same END RESULT as the expected command, mark it CORRECT
   - Example: 'uptime' and 'w' both show system load averages → BOTH CORRECT
   - Example: 'rm -rf dir' and 'rmdir dir' both remove directories → BOTH CORRECT
   - Example: Different flags that produce same output → CORRECT

2. **Multiple Valid Approaches**: Many Linux tasks have multiple correct solutions
   - Different tools can achieve the same goal (grep vs awk, find vs ls, etc.)
   - More robust/forceful commands are acceptable (rm -rf vs rmdir)
   - Different syntax that produces identical output is CORRECT

3. **When to Mark INCORRECT**:
   - Command does NOT accomplish the stated task
   - Command operates on wrong target (e.g., $HOME instead of $PATH)
   - Command produces fundamentally different output than requested
   - Command would fail or error for the given task

4. **Verify Your Understanding**: 
   - Read the Natural Language Query carefully
   - Ask: "Would this command give me what was asked for?"
   - Don't penalize for being "different" - only penalize for being "wrong"

5. Consider both expected AND alternative commands as reference points

Provide your analysis in this EXACT JSON format:
{{
  "overall_accuracy": "<percentage like '85%' or '17/20'>",
  "correct_count": <number of correct commands in THIS batch>,
  "total_count": {len(test_results)},
  "analysis": "<detailed analysis of patterns, strengths, weaknesses for THIS batch>",
  "per_test_results": [
    {{
      "test_num": <actual test number from the test case header>,
      "is_correct": true/false,
      "brief_note": "<one line explanation>"
    }},
    ...
  ]
}}

EXAMPLE EVALUATIONS FOR REFERENCE:
- Query: "print system load averages" | Expected: "w" | Generated: "uptime" → CORRECT (both show load averages)
- Query: "remove directory fake_dir" | Expected: "rmdir fake_dir" | Generated: "rm -rf fake_dir" → CORRECT (both remove the directory)
- Query: "print current user's path" | Expected: "echo $PATH" | Generated: "echo $HOME" → INCORRECT (PATH ≠ HOME)
- Query: "show hidden files" | Expected: "ls -a" | Generated: "ls -la" → CORRECT (both show hidden files)

Output ONLY the JSON, nothing else."""

    # Calculate appropriate max_tokens based on number of tests
    # Each test result needs roughly 50-100 tokens, plus overhead
    estimated_tokens = len(test_results) * 100 + 1000
    max_tokens = max(2000, min(estimated_tokens, 8000))
    
    response = call_gemini_api(prompt, temperature=0.2, max_tokens=max_tokens)
    
    # Parse JSON response
    try:
        # Remove markdown code blocks if present
        cleaned_response = response.strip()
        
        # Try multiple strategies to extract JSON
        if "```json" in cleaned_response:
            # Extract content between ```json and ```
            start_marker = "```json"
            end_marker = "```"
            start_idx = cleaned_response.find(start_marker)
            if start_idx >= 0:
                start_idx += len(start_marker)
                end_idx = cleaned_response.find(end_marker, start_idx)
                if end_idx >= 0:
                    cleaned_response = cleaned_response[start_idx:end_idx].strip()
                else:
                    # No closing marker found, take everything after opening
                    cleaned_response = cleaned_response[start_idx:].strip()
        elif "```" in cleaned_response:
            # Extract content between ``` and ```
            parts = cleaned_response.split("```")
            if len(parts) >= 3:
                cleaned_response = parts[1].strip()
                # Remove language identifier if present (e.g., "json")
                if cleaned_response.startswith(('json', 'JSON')):
                    cleaned_response = cleaned_response[4:].strip()
        
        # Find JSON object boundaries
        json_start = cleaned_response.find('{')
        json_end = cleaned_response.rfind('}')
        
        if json_start >= 0 and json_end >= 0 and json_end > json_start:
            json_str = cleaned_response[json_start:json_end + 1]
            
            # Try to parse the JSON
            result = json.loads(json_str)
            
            # Validate required fields
            if "overall_accuracy" in result and "correct_count" in result:
                console.print(f"[green]✓ Successfully parsed Gemini evaluation[/green]")
                return result
            else:
                console.print(f"[yellow]⚠️ JSON missing required fields[/yellow]")
        else:
            console.print(f"[yellow]⚠️ Could not find valid JSON boundaries[/yellow]")
            
    except json.JSONDecodeError as e:
        console.print(f"[yellow]⚠️ Failed to parse Gemini response as JSON: {e}[/yellow]")
        console.print(f"[dim]Response preview: {response[:500]}...[/dim]")
        
        # Try to salvage partial results
        try:
            # Attempt to extract just the summary fields if per_test_results is broken
            if "overall_accuracy" in response and "correct_count" in response:
                console.print("[yellow]Attempting to extract partial results...[/yellow]")
                # Use regex or simple parsing to extract key fields
                import re
                
                # Try to find overall_accuracy
                acc_match = re.search(r'"overall_accuracy"\s*:\s*"([^"]+)"', response)
                count_match = re.search(r'"correct_count"\s*:\s*(\d+)', response)
                total_match = re.search(r'"total_count"\s*:\s*(\d+)', response)
                
                if acc_match and count_match:
                    return {
                        "overall_accuracy": acc_match.group(1),
                        "correct_count": int(count_match.group(1)),
                        "total_count": int(total_match.group(1)) if total_match else len(test_results),
                        "analysis": "Partial results extracted from malformed response",
                        "per_test_results": []
                    }
        except Exception as extract_error:
            console.print(f"[dim]Could not extract partial results: {extract_error}[/dim]")
    except Exception as e:
        console.print(f"[yellow]⚠️ Unexpected error parsing response: {e}[/yellow]")
    
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
    For testing purposes, we instruct the LLM to:
    1. Generate ONLY the single best command
    2. Prioritize patterns from RAG-retrieved examples
    
    Returns:
    - (generated_command, success)
    """
    try:
        # Enhance the query to instruct LLM to:
        # 1. Generate only ONE best command
        # 2. Prioritize RAG-retrieved examples
        enhanced_query = (
            f"{nl_query}\n"
            "[TESTING MODE: Generate ONLY the single BEST command, no alternatives. "
            "PRIORITIZE using the same command patterns and syntax from the RAG examples above. "
            "If a retrieved example closely matches this query, use that exact command pattern.]"
        )
        
        # Use the CLI agent's generate_bash function
        bash_cmds, rag_examples = generate_bash(enhanced_query, error_context=None, show_rag_debug=False)
        
        if bash_cmds and bash_cmds.strip():
            # The LLM may still generate multiple commands
            # Extract only the FIRST command for testing
            first_command = bash_cmds.strip().split('\n')[0].strip()
            return first_command, True
        else:
            return "", False
            
    except Exception as e:
        console.print(f"[red]Error generating command: {e}[/red]")
        return "", False

def run_test_suite(test_cases: List[Dict], detailed: bool = False, batch_size: int = 50) -> Dict:
    """
    Run the complete test suite and return results.
    First generates all commands, then evaluates in batches with Gemini.
    
    Args:
        test_cases: List of test case dictionaries
        detailed: Whether to show detailed output (legacy param)
        batch_size: Number of tests to evaluate per Gemini API call
    
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
    console.print(f"[dim]Evaluating {len(test_results)} test results (batch size: {batch_size})...[/dim]\n")
    
    gemini_evaluation = evaluate_batch_accuracy(test_results, batch_size=batch_size)
    
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
    
    # Determine batch size based on number of tests
    # For large test sets, use smaller batches to avoid timeouts
    if len(test_cases) > 200:
        batch_size = 30
    elif len(test_cases) > 100:
        batch_size = 40
    else:
        batch_size = 50
    
    console.print(f"[dim]Using batch size: {batch_size} tests per API call[/dim]\n")
    
    results = run_test_suite(test_cases, detailed=False, batch_size=batch_size)
    
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
