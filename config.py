"""Central config — edit here, not scattered across modules.

Models + repo-relative paths (portable: no hardcoded /Users/...) + key constants.
All runtime files resolve relative to the repo root via __file__.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# --- models ---
BASE_MODEL = "mlx-community/Qwen3.5-4B-MLX-4bit"     # phone brain (4-bit, MLX)
E5_MODEL = "intfloat/multilingual-e5-small"          # embedder (ru/en)
E5_DIM = 384

# --- LoRA adapters (hot-swappable skills) ---
ADAPTERS = {
    "analytical": str(ROOT / "adapters_qwen4b_v2"),
    "creative": str(ROOT / "adapters_qwen4b_creative"),
}

# --- runtime state (git-ignored, created on use) ---
NOTES_FILE = str(ROOT / "notes.json")
REMINDERS_FILE = str(ROOT / "reminders.json")
PREFS_FILE = str(ROOT / "preferences.json")
NUDGE_LOG = str(ROOT / "nudges.json")
DOC_STORE = str(ROOT / "mp_docs.bin")               # DocumentMemory (RAG vectors)
FACTS_STORE = str(ROOT / "mp_facts.json")           # structured facts
PACKS_DIR = ROOT / "packs"                           # hobby RAG packs

# --- behaviour knobs ---
MEM_USER = "demo"
THINK_MAX_TOKENS = 700                                # token budget on reasoning turns
