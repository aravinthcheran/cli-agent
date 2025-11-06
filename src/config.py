"""Configuration constants for the CLI agent."""
import os

# ===============================
# FAISS & RAG CONFIGURATION
# ===============================
INDEX_FILE = "cache/bash_commands_l2.bin"
META_FILE = "cache/metadata_l2.npz"
TOP_K = 5

# ===============================
# OLLAMA CONFIGURATION
# ===============================
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mistral")
OLLAMA_URL = "http://localhost:11434/api/generate"

# ===============================
# EXECUTION CONFIGURATION
# ===============================
MAX_RETRIES = 2
COMMAND_TIMEOUT = 260

# ===============================
# DANGEROUS COMMAND PATTERNS
# ===============================
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

# Global debug mode flag
DEBUG_MODE = False
