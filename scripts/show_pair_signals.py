"""
Phase 5, Step 0b: print one table with every signal we have for the 24 pairs.

Columns: your rating and verdict (human), skill coverage from Phase 3, and the
three similarity scores from Phase 4 (read from the CSV that compare_methods
already saved, so nothing slow is recomputed).

This script only LOOKS at data. It changes no file.

Run from the project root (venv active):
    PYTHONPATH=src python scripts/show_pair_signals.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

from resume_matcher import config
from resume_matcher.matching.pairs import load_pairs
from resume_matcher.parsing.parser import parse_text
from resume_matcher.skills.comparison import compare_documents
from resume_matcher.skills.extractor import get_default_extractor

RESULTS_CSV = config.PROCESSED_DIR / "comparison_results.csv"


def _read(path: Path) -> str:
    """Read a pair file; relative paths are taken from the project root."""
    path = Path(path)
    if not path.is_absolute():
        path = config.PROJECT_ROOT / path
    return path.read_text(encoding="utf-8")


def _load_similarity() -> dict[int, dict[str, str]]:
    """Map pair_id -> the saved Phase 4 row."""
    if not RESULTS_CSV.exists():
        sys.exit(
            f"Missing {RESULTS_CSV}. Create it with:\n"
            "  PYTHONPATH=src python -m resume_matcher.matching.compare_methods"
        )
    with RESULTS_CSV.open(newline="", encoding="utf-8") as handle:
        return {int(row["pair_id"]): row for row in csv.DictReader(handle)}


def main() -> None:
    pairs = load_pairs()
    sims = _load_similarity()
    extractor = get_default_extractor()

    header = (
        f"{'id':>2} {'human':>5} {'verdict':<11} "
        f"{'cov':>5} {'match':>5} {'miss':>4} {'jd_n':>4}  "
        f"{'tfidf':>6} {'sbert':>6} {'sect':>6}"
    )
    print(header)
    print("-" * len(header))

    for pair in pairs:
        resume_doc = parse_text(_read(pair.resume_file), kind="resume")
        jd_doc = parse_text(_read(pair.jd_file), kind="jd")
        comp = compare_documents(resume_doc, jd_doc, extractor)
        sim = sims.get(pair.pair_id, {})
        print(
            f"{pair.pair_id:>2} {pair.human_score:>5.0f} {pair.human_verdict:<11} "
            f"{comp.coverage:>5.2f} {len(comp.matched):>5} {len(comp.missing):>4} "
            f"{comp.jd_total:>4}  "
            f"{float(sim.get('tfidf', 'nan')):>6.3f} "
            f"{float(sim.get('sbert', 'nan')):>6.3f} "
            f"{float(sim.get('sbert_sections', 'nan')):>6.3f}"
        )

    print()
    print("cov = matched / jd_n (Phase 3). jd_n = distinct skills found in the JD.")
    print("tfidf / sbert / sect = Phase 4 cosine scores (raw, not percentages).")


if __name__ == "__main__":
    main()
