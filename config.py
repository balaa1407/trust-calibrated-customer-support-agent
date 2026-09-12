"""
Central configuration for the Hiver Trust-Calibrated Support Agent.
All tunable parameters in one place — no magic numbers scattered in code.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
PROJECT_ROOT = Path(__file__).parent
load_dotenv(PROJECT_ROOT / ".env")

# ── API Keys ──────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# ── Model Configuration ──────────────────────────────────────────────────────
# Default to gemini-3.6-flash for speed and cost efficiency
LLM_MODEL = "gemini-3.6-flash"
EMBEDDING_MODEL = "text-embedding-004"

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INDEX_DIR = PROJECT_ROOT / "retrieval" / "index_store"
EVAL_DIR = PROJECT_ROOT / "eval"
GOLDEN_SET_DIR = EVAL_DIR / "golden_set"
RESULTS_DIR = EVAL_DIR / "results"
REPORT_DIR = PROJECT_ROOT / "report"

# ── Data Configuration ────────────────────────────────────────────────────────
KAGGLE_DATASET = "thoughtvector/customer-support-on-twitter"
CSV_FILENAME = "twcs.csv"
SELECTED_BRAND = ""  # Set after brand_selection.py

# ── Intent Classification ─────────────────────────────────────────────────────
MAX_INTENTS = 10
CLASSIFIER_TEMPERATURE = 0.0
CLASSIFIER_FEW_SHOT_K = 3

# ── Retrieval ─────────────────────────────────────────────────────────────────
RETRIEVAL_TOP_K = 3
EMBEDDING_BATCH_SIZE = 50
EMBEDDING_DIMENSION = 768

# ── Generation ────────────────────────────────────────────────────────────────
GENERATION_TEMPERATURE = 0.3
GENERATION_MAX_TOKENS = 300

# ── Escalation / Confidence ──────────────────────────────────────────────────
CONFIDENCE_WEIGHTS = {
    "intent_confidence": 0.3,
    "retrieval_strength": 0.4,
    "self_consistency": 0.3,
}
DEFAULT_ESCALATION_THRESHOLD = 0.6
SELF_CONSISTENCY_SAMPLES = 2

# ── Evaluation ────────────────────────────────────────────────────────────────
GOLDEN_SET_SIZE = 200
GOLDEN_SET_SUBSAMPLE = 50
LLM_JUDGE_TEMPERATURE = 0.0
HUMAN_AGREEMENT_SAMPLE = 80

# ── Rate Limiting ─────────────────────────────────────────────────────────────
API_CALLS_PER_MINUTE = 15
API_RETRY_ATTEMPTS = 3
API_RETRY_DELAY = 2.0

# ── Ensure directories exist ─────────────────────────────────────────────────
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, INDEX_DIR, GOLDEN_SET_DIR, RESULTS_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)
