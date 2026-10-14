"""
Central settings for the whole project.

Keeping paths and tunable numbers in ONE file means later phases never
hard-code them, and you can explain every "magic number" in your viva.

NOTE: values marked PLACEHOLDER are starting guesses. They are tuned on the
labelled test pairs in Phase 5 and must not be quoted as results before that.
"""
from pathlib import Path

# ---- Paths (computed from this file's location, so they work on any OS) ----
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLES_DIR = DATA_DIR / "samples"
LABELS_DIR = DATA_DIR / "labels"
SKILLS_FILE = DATA_DIR / "skills.json"          # created in Phase 3

# ---- Matching model names (used from Phase 4) ----
TFIDF_MAX_FEATURES = 5000                        # PLACEHOLDER
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"            # small, CPU-friendly

# ---- Scoring (tuned in Phase 5) ----
WEIGHT_SEMANTIC = 0.5                            # PLACEHOLDER
WEIGHT_SKILL_COVERAGE = 0.5                      # PLACEHOLDER
FIT_THRESHOLD = 0.70                             # PLACEHOLDER
PARTIAL_THRESHOLD = 0.45                         # PLACEHOLDER

# ---- Input guards (used from Phase 2) ----
MIN_TEXT_CHARS = 200                             # shorter than this = suspicious