import pickle
import re
import subprocess
import platform
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ===============================
# Configuration
# ===============================
INDEX_FILE = "bash_commands.index"
META_FILE = "metadata.pkl"
TOP_K = 5  # number of FAISS neighbors
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
PLATFORM = "windows"  # Detect platform for command generation

# ===============================
# Load FAISS index and metadata
# ===============================
print("Loading FAISS index and metadata...")
index = faiss.read_index(INDEX_FILE)
with open(META_FILE, "rb") as f:
    data = pickle.load(f)

# ===============================
# Load embedder and LLM
# ===============================
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("Loading Qwen model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map={"": "cpu"},
    low_cpu_mem_usage=True,
    torch_dtype="auto"
)

# Configure generation pipeline with better parameters for reducing hallucinations
gen = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer,
    return_full_text=False,  # Only return generated text, not the input
    clean_up_tokenization_spaces=True
)

# ===============================
# Helper functions
# ===============================
def retrieve(query, top_k=TOP_K):
    """Retrieve top-k similar examples from FAISS"""
    vec = embedder.encode([query], convert_to_numpy=True)
    _, I = index.search(vec, top_k)
    neighbors = []
    for idx in I[0]:
        neighbors.append(data[idx])
    return neighbors

def extract_filename(query):
    """Detect if user mentions a specific file"""
    match = re.search(r'\b[\w\-/]+\.\w+\b', query)
    return match.group(0) if match else None

def generate_bash(query, error_context=None):
    neighbors = retrieve(query)
    file_name = extract_filename(query)
    
    # Detect platform
    is_windows = platform.system() == "Windows"

    # Enhanced prompt with better context understanding
    if is_windows:
        prompt = """You are an expert Windows PowerShell/CMD command generator. Convert natural language to precise Windows commands.

IMPORTANT RULES:
- Generate ONLY the exact command for Windows, no explanations
- Use PowerShell/CMD syntax (not bash)
- For creating files: use 'New-Item' or 'echo. > filename'
- For listing: use 'dir' or 'Get-ChildItem'
- For parent directory: use '..'
- Be precise and avoid hallucinations

"""
    else:
        prompt = """You are an expert Bash command generator. Convert natural language to precise bash commands.

IMPORTANT RULES:
- Generate ONLY the exact bash command, no explanations
- "parent directory" means "../" (not literal "parent")
- "current directory" means "." or "./"
- Use proper bash syntax and paths
- Be precise and avoid hallucinations

"""
    
    # If there was an error, add context to fix it
    if error_context:
        prompt += f"\nPREVIOUS ERROR: {error_context['error']}\n"
        prompt += f"FAILED COMMAND: {error_context['command']}\n"
        prompt += "Generate a corrected command that will work.\n\n"
    
    prompt += "Examples:\n"

    # Add common examples based on platform
    if is_windows:
        prompt += """Input: list files in parent directory
Output: dir ..

Input: show contents of parent directory
Output: Get-ChildItem ..

Input: list all files in current directory
Output: dir

Input: create a text file
Output: New-Item -ItemType File -Name file.txt

"""
        # Prepend file creation template for Windows
        create_file_keywords = any(word in query.lower() for word in ["create", "file", "touch", "make", "new"])
        if file_name and create_file_keywords:
            prompt += f"Input: create a file named {file_name}\nOutput: New-Item -ItemType File -Name {file_name}\n\n"
    else:
        prompt += """Input: list files in parent directory
Output: ls ../

Input: show contents of parent directory
Output: ls ../

Input: list all files in current directory
Output: ls

"""
        # Prepend file creation template for Unix/Linux
        create_file_keywords = any(word in query.lower() for word in ["create", "file", "touch", "make", "new"])
        if file_name and create_file_keywords:
            prompt += f"Input: create a file named {file_name}\nOutput: touch {file_name}\n\n"

    # Add FAISS examples with better formatting, but filter out confusing ones
    for item in neighbors:
        # Skip examples that might confuse the model about "parent" directory
        if "parent" in item['bash'] and not "../" in item['bash']:
            continue
        prompt += f"Input: {item['nl']}\nOutput: {item['bash']}\n\n"

    prompt += f"""Generate the bash command for this input:
Input: {query}
Output:"""

    # Generate with very constrained parameters to prevent hallucinations
    output = gen(
        prompt, 
        max_new_tokens=15,  # Very short to prevent hallucinations
        temperature=0.0,    # Completely deterministic
        do_sample=False,    # Use greedy decoding
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.0,
        return_full_text=False
    )[0]["generated_text"]
    
    # More aggressive output cleaning
    if "Output:" in output:
        parts = output.split("Output:")
        if len(parts) > 1:
            output = parts[-1].strip()
    
    # Take only the first line and remove everything after the first complete command
    lines = output.split("\n")
    output = lines[0].strip()
    
    # Remove any text that appears after common command patterns
    # Stop at first space followed by uppercase letter (often indicates hallucination)
    import re
    match = re.search(r'^([^A-Z]*?)(?:\s+[A-Z]|$)', output)
    if match:
        output = match.group(1).strip()
    
    # Clean up artifacts
    output = output.replace("```", "").replace("`", "").replace('"', "")
    output = re.sub(r'[.!?]+$', '', output)
    
    # Fix spacing issues
    output = re.sub(r'([a-z])([A-Z])', r'\1 \2', output)  # Add space before capital letters
    output = re.sub(r'(\.\./?)([a-z])', r'\1 \2', output)  # Add space after ../
    
    # Post-process common issues
    if "parent" in output and "../" not in output:
        output = output.replace("parent", "../")
    
    # Basic validation
    if not output or len(output.strip()) == 0:
        return "echo 'Error: Could not generate command'"
    
    # Additional validation for common directory operations
    if "parent directory" in query.lower() and not any(x in output for x in ["../", "cd .."]):
        return "ls ../"
    
    return output

# ===============================
# CLI Loop
# ===============================
def explain_error(original_query, first_cmd, second_cmd, first_error, second_error):
    """Generate an explanation for why commands failed"""
    prompt = f"""You are a helpful technical assistant. Explain why the commands failed in simple terms.

User's request: {original_query}

First command attempted: {first_cmd}
First error: {first_error}

Second command attempted: {second_cmd}
Second error: {second_error}

Provide a brief, clear explanation of:
1. Why both commands failed
2. What the actual issue is
3. A suggestion for what command might work (if you know)

Keep it concise (2-3 sentences)."""

    try:
        output = gen(
            prompt,
            max_new_tokens=100,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_full_text=False
        )[0]["generated_text"]
        
        # Clean up the output
        output = output.strip()
        lines = output.split('\n')
        # Take first few meaningful lines
        explanation = '\n'.join([line for line in lines if line.strip()][:5])
        
        return explanation if explanation else "The commands failed due to syntax or permission issues. Please check the command syntax for your platform."
    except Exception as e:
        return f"Unable to generate explanation. Both commands failed. Consider checking: command syntax, file permissions, or whether the required tools are installed on your system."

def execute_command(command):
    """Execute a bash/shell command and return the output and status"""
    try:
        # On Windows, use PowerShell to execute commands
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30  # 30 second timeout
        )
        
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += result.stderr
        
        # Check if command failed
        is_error = result.returncode != 0 or "not recognized" in output.lower() or "error" in output.lower()
            
        return {
            "output": output.strip() if output else "Command executed successfully (no output)",
            "is_error": is_error,
            "return_code": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {
            "output": "Error: Command timed out (exceeded 30 seconds)",
            "is_error": True,
            "return_code": -1
        }
    except Exception as e:
        return {
            "output": f"Error executing command: {str(e)}",
            "is_error": True,
            "return_code": -1
        }

def run_cli():
    """Run the interactive CLI loop with agent-like self-correction"""
    print("=== Bash CLI Agent ===")
    print(f"Platform: {platform.system()}")
    print("Type your command in natural language (type 'exit' to quit)\n")

    while True:
        query = input("> ")
        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        # Initial command generation
        bash_cmd = generate_bash(query)
        print(f"\nGenerated command:\n{bash_cmd}\n")
        
        # Ask for user permission to execute
        permission = input("Execute this command? (y/n): ").strip().lower()
        
        if permission in ['y', 'yes']:
            print("\nExecuting command...")
            result = execute_command(bash_cmd)
            print(f"\nOutput:\n{result['output']}\n")
            
            # If error occurred, offer to self-correct
            if result['is_error']:
                print("⚠️  Command failed!")
                retry_permission = input("\nShould I try to fix this and generate a corrected command? (y/n): ").strip().lower()
                
                if retry_permission in ['y', 'yes']:
                    print("\n🤖 Analyzing error and generating corrected command...")
                    
                    # Create error context for the model
                    error_context = {
                        "error": result['output'],
                        "command": bash_cmd
                    }
                    
                    # Generate corrected command
                    corrected_cmd = generate_bash(query, error_context=error_context)
                    print(f"\nCorrected command:\n{corrected_cmd}\n")
                    
                    # Ask permission to execute corrected command
                    execute_corrected = input("Execute the corrected command? (y/n): ").strip().lower()
                    
                    if execute_corrected in ['y', 'yes']:
                        print("\nExecuting corrected command...")
                        corrected_result = execute_command(corrected_cmd)
                        print(f"\nOutput:\n{corrected_result['output']}\n")
                        
                        if not corrected_result['is_error']:
                            print("✅ Success!\n")
                        else:
                            print("❌ Still failed. Let me explain the issue...\n")
                            explanation = explain_error(query, bash_cmd, corrected_cmd, result['output'], corrected_result['output'])
                            print(f"📝 Explanation:\n{explanation}\n")
                    else:
                        print("Corrected command not executed.\n")
                else:
                    print("Self-correction skipped.\n")
        else:
            print("Command not executed.\n")

if __name__ == "__main__":
    run_cli()
