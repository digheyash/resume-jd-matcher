"""
Central settings for the whole project.

Keeping paths and tunable numbers in ONE file means later phases never
hard-code them, and you can explain every "magic number" in your viva.

NOTE: values marked PLACEHOLDER are starting guesses and must not be quoted as
results. Values marked TUNED were produced by analyzer/tuning.py in Phase 5 on
the 24 rated (synthetic) pairs and carry the caveats printed in that report.
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

# ---- Scoring (TUNED in Phase 5 by analyzer/tuning.py; see its report) ----
# Fitted on 24 SYNTHETIC rated pairs: the best current values, not proven
# constants. Re-run `python -m resume_matcher.analyzer.tuning` if the data changes.
SEMANTIC_ANCHOR_LOW = 0.2078                     # TUNED: raw cosine that maps to 0
SEMANTIC_ANCHOR_HIGH = 0.5901                    # TUNED: raw cosine that maps to 1
WEIGHT_SEMANTIC = 0.8                            # TUNED (upper limit of the 0.2-0.8 search)
WEIGHT_SKILL_COVERAGE = 0.2                      # TUNED (= 1 - WEIGHT_SEMANTIC)
FIT_THRESHOLD = 0.5926                           # TUNED: score >= this -> Fit
PARTIAL_THRESHOLD = 0.4669                       # TUNED: score >= this -> Partial Fit

# ---- Input guards (used from Phase 2) ----
MIN_TEXT_CHARS = 200                             # shorter than this = suspicious
