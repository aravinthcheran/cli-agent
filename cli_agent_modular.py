#!/usr/bin/env python3
"""
Linux Bash CLI Agent - Main Entry Point
A RAG-powered CLI agent that generates and executes bash commands from natural language.
"""

import platform
from rich.console import Console
from rich.prompt import Prompt, Confirm

# Import modular components
from src import config
from src.rag import initialize_rag
from src.utils import check_dangerous_commands
from src.clarification import unified_clarify_command
from src.command_processor import generate_bash, execute_sequence, explain_error
from src.rag import display_rag_examples

console = Console()


def run_cli():
    """Main CLI loop for the bash agent."""
    console.print("[bold magenta]=== Linux Bash CLI Agent (RAG + Ollama) ===[/bold magenta]")
    console.print(f"[bold yellow]Platform:[/bold yellow] {platform.system()}")
    
    # Initialize RAG resources
    initialize_rag()
    
    # Prompt for mode selection
    console.print("\n[bold cyan]Select mode:[/bold cyan]")
    console.print("  1. [green]Normal mode[/green] - Clean output with minimal details")
    console.print("  2. [yellow]Debug mode[/yellow] - Show all internal workings\n")
    
    mode_choice = Prompt.ask("Choose mode", choices=["1", "2"], default="1")
    config.DEBUG_MODE = (mode_choice == "2")
    
    if config.DEBUG_MODE:
        console.print("\n[bold yellow]🔧 DEBUG MODE ENABLED[/bold yellow]")
        console.print("[dim]You will see detailed internal workings of the agent[/dim]\n")
    else:
        console.print("\n[bold green]✓ Normal mode active[/bold green]\n")
    
    console.print("[bold green]Type a natural language instruction (or 'exit' to quit)[/bold green]")
    console.print("[dim]Commands: 'exit'/'quit' to exit, 'toggle debug' to switch modes[/dim]\n")
    
    while True:
        query = Prompt.ask("> ").strip()
        
        # Handle special commands
        if query.lower() in ["exit", "quit", "q"]:
            console.print("[bold red]Goodbye![/bold red]")
            break
        
        if query.lower() in ["toggle debug", "debug", "toggle"]:
            config.DEBUG_MODE = not config.DEBUG_MODE
            if config.DEBUG_MODE:
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
        
        for attempt in range(config.MAX_RETRIES + 1):
            if attempt > 0:
                console.print(f"\n[bold yellow]🔄 Auto-retry attempt {attempt}/{config.MAX_RETRIES}[/bold yellow]")
            
            show_rag_debug = (attempt == 2)
            
            bash_cmds, rag_examples = generate_bash(query, error_context, show_rag_debug)
            
            if not bash_cmds or bash_cmds.strip() == "":
                console.print("[bold red]No valid commands generated.[/bold red]\n")
                break
            
            # Display RAG examples - only in debug mode or if explicitly requested
            if config.DEBUG_MODE or show_rag_debug:
                display_rag_examples(rag_examples)
            
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
                style = "yellow" if config.DEBUG_MODE else "dim"
                console.print(f"[{style}]Auto-executing retry attempt...[/{style}]")
            
            if not should_execute:
                console.print("[grey]Skipped execution.[/grey]\n")
                break
            
            result = execute_sequence(bash_cmds)
            console.print(f"\n[bold white]Output:[/bold white]\n{result['output']}\n")
            
            if result["is_error"]:
                console.print("[bold red]❌ Command(s) failed![/bold red]")
                
                # Generic handling for "No such file or directory" errors in ANY command
                is_not_found = 'no such file or directory' in result.get('error_message', '').lower()
                
                if is_not_found and attempt == 0:
                    from src.clarification import handle_file_not_found_error
                    
                    failed_command = result.get('failed_command', bash_cmds.split('\n')[0])
                    corrected_command = handle_file_not_found_error(
                        command=failed_command,
                        error_message=result.get('error_message', ''),
                        original_query=query
                    )
                    
                    if corrected_command:
                        # Re-execute with the corrected command
                        console.print("\n[bold green]♻️  Re-executing with corrected path...[/bold green]")
                        result = execute_sequence(corrected_command)
                        console.print(f"\n[bold white]Output:[/bold white]\n{result['output']}\n")
                        
                        if not result["is_error"]:
                            console.print("[bold green]✅ All commands executed successfully![/bold green]\n")
                            break
                        else:
                            console.print("[yellow]Still encountering errors after correction.[/yellow]")
                    else:
                        console.print("[red]Cancelled - could not determine correct file path.[/red]\n")
                        break
                
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
