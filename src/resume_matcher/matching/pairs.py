"""
Load and check the hand-labelled resume / job-description pairs.

THE FILE
    data/labels/pairs.csv   (written empty by scripts/make_synthetic_pairs.py)
    columns: pair_id, resume_file, jd_file, human_score_0_to_100,
             human_verdict, notes
    File paths are relative to the PROJECT ROOT.

WHAT THE RATINGS MEAN
    human_score_0_to_100   YOUR judgement of how well the resume fits the job.
    human_verdict          Fit | Partial Fit | No-Fit  (case does not matter).
    The numbers are never changed by this code; they are the ground truth that
    the TF-IDF and Sentence-BERT scores are compared against.

WHAT IS CHECKED (all problems are collected and reported together)
    * the header has the six expected columns;
    * pair_id is a whole number and unique;
    * both text files exist;
    * the score is a number from 0 to 100, the verdict is one of the three;
    * a row has BOTH a score and a verdict, or NEITHER (half-filled is wrong);
    * with require_ratings=True (the default) every row must be rated.

CONSISTENCY WARNINGS (not errors)
    check_consistency() lists "inversions": a pair with a better verdict but a
    lower score than a pair with a worse verdict (order: No-Fit < Partial Fit
    < Fit). Two rows tied on score are never an inversion. An inversion is
    usually a typo, or a sign that the pair deserves a note.

STATUS COMMAND (safe to run while you are still rating)
    PYTHONPATH=src python -m resume_matcher.matching.pairs
"""
from __future__ import annotations

import csv
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from resume_matcher import config

EXPECTED_COLUMNS = (
    "pair_id",
    "resume_file",
    "jd_file",
    "human_score_0_to_100",
    "human_verdict",
    "notes",
)

VALID_VERDICTS = ("Fit", "Partial Fit", "No-Fit")
_VERDICT_BY_LOWER = {v.lower(): v for v in VALID_VERDICTS}
_VERDICT_RANK = {"No-Fit": 0, "Partial Fit": 1, "Fit": 2}


class PairsError(ValueError):
    """The pairs file has problems. `.problems` lists every one of them."""

    def __init__(self, problems: list[str]):
        self.problems = list(problems)
        super().__init__("\n".join(self.problems))


@dataclass(frozen=True)
class LabelledPair:
    """One resume-JD pair with the human rating (None while not rated yet)."""

    pair_id: int
    resume_file: Path
    jd_file: Path
    human_score: float | None
    human_verdict: str | None
    notes: str

    @property
    def is_rated(self) -> bool:
        return self.human_score is not None and self.human_verdict is not None

    def read_resume(self) -> str:
        return self.resume_file.read_text(encoding="utf-8")

    def read_jd(self) -> str:
        return self.jd_file.read_text(encoding="utf-8")


def default_pairs_path() -> Path:
    return config.LABELS_DIR / "pairs.csv"


def _cell(row: dict, name: str) -> str:
    """A cell as stripped text; a missing cell counts as empty."""
    value = row.get(name)
    return "" if value is None else value.strip()


def _parse_row(row: dict, line: int, root: Path, require_ratings: bool):
    """Return (LabelledPair or None, [problems]) for one CSV row."""
    problems: list[str] = []
    where = f"row {line}"

    # pair_id
    raw_id = _cell(row, "pair_id")
    pair_id = None
    try:
        pair_id = int(raw_id)
    except ValueError:
        problems.append(f"{where}: pair_id {raw_id!r} is not a whole number")

    # files
    files: dict[str, Path | None] = {}
    for column in ("resume_file", "jd_file"):
        text = _cell(row, column)
        if not text:
            problems.append(f"{where}: {column} is empty")
            files[column] = None
            continue
        path = Path(text)
        path = path if path.is_absolute() else root / path
        if not path.is_file():
            problems.append(f"{where}: {column} not found: {text}")
        files[column] = path

    # extra cells beyond the six columns (for example a stray comma)
    extra = row.get(None)
    if extra and any(str(cell).strip() for cell in extra):
        problems.append(
            f"{where}: too many columns; check for an extra comma "
            "(a note containing a comma must be wrapped in double quotes)"
        )

    # rating
    raw_score = _cell(row, "human_score_0_to_100")
    raw_verdict = _cell(row, "human_verdict")
    score: float | None = None
    verdict: str | None = None

    if raw_score:
        try:
            score = float(raw_score)
        except ValueError:
            problems.append(f"{where}: score {raw_score!r} is not a number")
        else:
            if not math.isfinite(score) or not 0.0 <= score <= 100.0:
                problems.append(f"{where}: score {raw_score!r} is not between 0 and 100")
                score = None
    if raw_verdict:
        verdict = _VERDICT_BY_LOWER.get(raw_verdict.lower())
        if verdict is None:
            problems.append(
                f"{where}: verdict {raw_verdict!r} must be one of "
                + ", ".join(VALID_VERDICTS)
            )

    if bool(raw_score) != bool(raw_verdict):
        missing = "verdict" if raw_score else "score"
        problems.append(f"{where}: has a {'score' if raw_score else 'verdict'} but no {missing}")
    elif not raw_score and require_ratings:
        problems.append(f"{where}: not rated yet (score and verdict are empty)")

    if problems:
        return None, problems
    return (
        LabelledPair(
            pair_id=pair_id,
            resume_file=files["resume_file"],
            jd_file=files["jd_file"],
            human_score=score,
            human_verdict=verdict,
            notes=_cell(row, "notes"),
        ),
        [],
    )


def load_pairs(
    path: str | Path | None = None,
    root: str | Path | None = None,
    require_ratings: bool = True,
) -> list[LabelledPair]:
    """Read and check the pairs file.

    Args:
        path: the CSV file. Defaults to data/labels/pairs.csv.
        root: folder that the file paths in the CSV are relative to.
            Defaults to the project root.
        require_ratings: if True (default) every row must be rated. Use False
            to look at a half-finished file.

    Raises:
        PairsError: with every problem found, one per line.
        FileNotFoundError: if the CSV itself does not exist.
    """
    csv_path = Path(path) if path is not None else default_pairs_path()
    base = Path(root) if root is not None else config.PROJECT_ROOT

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        header = tuple(name.strip() for name in (reader.fieldnames or ()))
        if header != EXPECTED_COLUMNS:
            raise PairsError(
                [
                    "header must be exactly: " + ",".join(EXPECTED_COLUMNS),
                    "found: " + ",".join(header),
                ]
            )
        rows = list(reader)

    pairs: list[LabelledPair] = []
    problems: list[str] = []
    for offset, row in enumerate(rows):
        pair, row_problems = _parse_row(row, offset + 2, base, require_ratings)
        problems.extend(row_problems)
        if pair is not None:
            pairs.append(pair)

    seen: dict[int, int] = {}
    for offset, row in enumerate(rows):
        try:
            number = int(_cell(row, "pair_id"))
        except ValueError:
            continue
        if number in seen:
            problems.append(
                f"row {offset + 2}: pair_id {number} is already used in row {seen[number]}"
            )
        else:
            seen[number] = offset + 2

    if problems:
        raise PairsError(problems)
    return pairs


def check_consistency(pairs: list[LabelledPair]) -> list[str]:
    """Warnings for rating inversions among the RATED pairs (see module doc)."""
    rated = [p for p in pairs if p.is_rated]
    warnings: list[str] = []
    for better in rated:
        for worse in rated:
            if (
                _VERDICT_RANK[better.human_verdict] > _VERDICT_RANK[worse.human_verdict]
                and better.human_score < worse.human_score
            ):
                warnings.append(
                    f"pair {better.pair_id} is '{better.human_verdict}' with score "
                    f"{better.human_score:g}, but pair {worse.pair_id} is "
                    f"'{worse.human_verdict}' with the higher score {worse.human_score:g}"
                )
    return warnings


def main(argv: list[str] | None = None) -> int:
    """Print a status report on the pairs file. Safe while rating is unfinished."""
    path = Path(argv[0]) if argv else default_pairs_path()
    try:
        pairs = load_pairs(path, require_ratings=False)
    except FileNotFoundError:
        print(f"File not found: {path}")
        return 1
    except PairsError as error:
        print(f"{len(error.problems)} problem(s) in {path}:")
        for problem in error.problems:
            print("  -", problem)
        return 1

    rated = [p for p in pairs if p.is_rated]
    print(f"{len(pairs)} pairs, {len(rated)} rated, {len(pairs) - len(rated)} still to rate")
    if rated:
        counts = Counter(p.human_verdict for p in rated)
        print("verdicts:", ", ".join(f"{v}: {counts.get(v, 0)}" for v in VALID_VERDICTS))
        scores = [p.human_score for p in rated]
        print(f"scores: min {min(scores):g}, max {max(scores):g}")
    warnings = check_consistency(pairs)
    if warnings:
        print(f"{len(warnings)} consistency warning(s):")
        for warning in warnings:
            print("  -", warning)
    elif rated:
        print("no consistency warnings")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
