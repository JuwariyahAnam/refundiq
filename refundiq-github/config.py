"""Central settings for the Refund Desk agent."""
import os
from datetime import date
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / os.getenv("REFUNDIQ_DB", "refund_desk.db")  # eval.py uses its own eval.db
POLICY_PATH = BASE_DIR / "data" / "policy.md"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# If this model name errors, change it in .env (e.g. GEMINI_MODEL=gemini-2.5-flash)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
# Backup models, tried in order when one is overloaded (503) or unavailable (404).
GEMINI_MODELS = [m.strip() for m in os.getenv(
    "GEMINI_MODELS", GEMINI_MODEL + ",gemini-3.8-flash-lite,gemini-3.5-flash,gemini-3-flash,gemini-flash-latest"
).split(",") if m.strip()]

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
# Groq models tried in order (first one that exists on your account is used)
GROQ_MODELS = [m.strip() for m in os.getenv(
    "GROQ_MODELS", "openai/gpt-oss-120b,openai/gpt-oss-20b,llama-3.3-70b-versatile,qwen/qwen3-32b"
).split(",") if m.strip()]
# Which LLM to use: "groq" or "gemini". Default: Groq if a Groq key is set, else Gemini.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq" if GROQ_API_KEY else "gemini").lower()

# Fixed "today" so the demo and eval are reproducible
TODAY = date(2026, 10, 31)

AUTO_REFUND_LIMIT = 2000      # P9: refunds above this need a human
REPEAT_RETURN_THRESHOLD = 3   # P10: requests in the last 60 days
REPEAT_RETURN_WINDOW_DAYS = 60
MAX_AGENT_STEPS = 10          # safety cap on the agent loop

COURIER_HOST = "127.0.0.1"
COURIER_PORT = 8765
COURIER_TIMEOUT_SECS = 2
