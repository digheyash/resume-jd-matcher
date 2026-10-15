"""Score the skill extractor against hand-labelled cases (Phase 3).

Each labelled case is a short text, the section it belongs to, and the set of
skills a human says it contains. For every case we compare the extractor's
answer with the human answer:

    TP (true positive)   found and correct
    FP (false positive)  found but wrong
    FN (false negative)  real but missed

    precision = TP / (TP + FP)   of what we found, how much was right
    recall    = TP / (TP + FN)   of what existed, how much we found
    F1        = 2 * P * R / (P + R)

The overall numbers are MICRO averages: TP, FP and FN are added up over all
cases first, then the formulas are applied once. A case with many skills
therefore counts for more than a case with one.

Run it from the project root (Git Bash):

    PYTHONPATH=src python -m resume_matcher.skills.evaluation
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from resume_matcher.skills.extractor import SkillExtractor, get_default_extractor

VALID_KINDS = ("resume", "jd")


class LabelError(Exception):
    """Raised when the labels file is missing or malformed."""


@dataclass(frozen=True)
class LabelledCase:
    """One hand-labelled snippet."""

    id: str
    kind: str  # "resume" or "jd"
    section: str  # section key, for example "skills" or "requirements"
    text: str
    skills: frozenset[str]  # canonical names a human sees in the text
    note: str = ""


@dataclass(frozen=True)
class Scores:
    """Counts and the metrics derived from them."""

    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        found = self.tp + self.fp
        return self.tp / found if found else 0.0

    @property
    def recall(self) -> float:
        real = self.tp + self.fn
        return self.tp / real if real else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def __add__(self, other: "Scores") -> "Scores":
        return Scores(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn)


@dataclass(frozen=True)
class CaseResult:
    """The extractor's answer for one case, compared with the label."""

    case: LabelledCase
    predicted: frozenset[str]

    @property
    def true_positives(self) -> frozenset[str]:
        return self.predicted & self.case.skills

    @property
    def false_positives(self) -> frozenset[str]:
        return self.predicted - self.case.skills

    @property
    def false_negatives(self) -> frozenset[str]:
        return self.case.skills - self.predicted

    @property
    def is_exact(self) -> bool:
        return self.predicted == self.case.skills

    @property
    def scores(self) -> Scores:
        return Scores(
            len(self.true_positives),
            len(self.false_positives),
            len(self.false_negatives),
        )


@dataclass(frozen=True)
class EvaluationReport:
    """All case results plus overall and per-kind scores."""

    results: tuple[CaseResult, ...]
    overall: Scores
    by_kind: dict[str, Scores] = field(default_factory=dict)

    @property
    def exact_matches(self) -> int:
        """Number of cases where the extractor's set equals the label exactly."""
        return sum(1 for r in self.results if r.is_exact)

    @property
    def mistakes(self) -> tuple[CaseResult, ...]:
        """Cases with at least one false positive or false negative."""
        return tuple(r for r in self.results if not r.is_exact)


def default_labels_path() -> Path:
    """data/labels/skill_labels.json, found relative to the project root."""
    return Path(__file__).resolve().parents[3] / "data" / "labels" / "skill_labels.json"


def parse_cases(data: object) -> list[LabelledCase]:
    """Validate parsed JSON and build LabelledCase objects."""
    if not isinstance(data, dict) or not isinstance(data.get("cases"), list):
        raise LabelError("labels file must be an object with a 'cases' list")

    cases: list[LabelledCase] = []
    seen: set[str] = set()
    for position, raw in enumerate(data["cases"], start=1):
        where = f"cases[{position}]"
        if not isinstance(raw, dict):
            raise LabelError(f"{where} must be an object")
        for key in ("id", "kind", "section", "text", "skills"):
            if key not in raw:
                raise LabelError(f"{where} is missing '{key}'")
        case_id = raw["id"]
        if not isinstance(case_id, str) or not case_id.strip():
            raise LabelError(f"{where}.id must be a non-empty string")
        if case_id in seen:
            raise LabelError(f"duplicate case id: {case_id!r}")
        seen.add(case_id)
        if raw["kind"] not in VALID_KINDS:
            raise LabelError(f"{case_id}: kind must be one of {VALID_KINDS}")
        if not isinstance(raw["section"], str) or not isinstance(raw["text"], str):
            raise LabelError(f"{case_id}: section and text must be strings")
        skills = raw["skills"]
        if not isinstance(skills, list) or not all(isinstance(s, str) for s in skills):
            raise LabelError(f"{case_id}: skills must be a list of strings")
        if len(set(skills)) != len(skills):
            raise LabelError(f"{case_id}: skills contains duplicates")
        cases.append(
            LabelledCase(
                id=case_id,
                kind=raw["kind"],
                section=raw["section"],
                text=raw["text"],
                skills=frozenset(skills),
                note=str(raw.get("note", "")),
            )
        )
    if not cases:
        raise LabelError("labels file contains no cases")
    return cases


def load_cases(path: str | Path | None = None) -> list[LabelledCase]:
    """Load labelled cases (default: data/labels/skill_labels.json)."""
    file_path = Path(path) if path is not None else default_labels_path()
    if not file_path.exists():
        raise LabelError(f"labels file not found: {file_path}")
    try:
        with open(file_path, encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise LabelError(f"{file_path} is not valid JSON: {exc}") from exc
    return parse_cases(data)


def evaluate(
    cases: Iterable[LabelledCase], extractor: SkillExtractor | None = None
) -> EvaluationReport:
    """Run the extractor on every case and collect the scores."""
    extractor = extractor if extractor is not None else get_default_extractor()
    results: list[CaseResult] = []
    for case in cases:
        found = extractor.extract_from_text(case.text, case.section).skills()
        results.append(CaseResult(case=case, predicted=frozenset(found)))

    overall = Scores()
    by_kind: dict[str, Scores] = {}
    for result in results:
        overall = overall + result.scores
        kind = result.case.kind
        by_kind[kind] = by_kind.get(kind, Scores()) + result.scores
    return EvaluationReport(results=tuple(results), overall=overall, by_kind=by_kind)


def _metric_line(label: str, scores: Scores) -> str:
    return (
        f"{label:<10} precision {scores.precision:.3f}   recall {scores.recall:.3f}"
        f"   F1 {scores.f1:.3f}   (TP {scores.tp}, FP {scores.fp}, FN {scores.fn})"
    )


def format_report(report: EvaluationReport) -> str:
    """Plain-text report: overall scores, per-kind scores and every mistake."""
    lines = [
        f"Cases: {len(report.results)}   "
        f"exactly right: {report.exact_matches}   "
        f"with mistakes: {len(report.mistakes)}",
        "",
        _metric_line("OVERALL", report.overall),
    ]
    for kind in sorted(report.by_kind):
        lines.append(_metric_line(kind, report.by_kind[kind]))

    lines.append("")
    if not report.mistakes:
        lines.append("No mistakes.")
    else:
        lines.append("MISTAKES")
        for result in report.mistakes:
            case = result.case
            lines.append(f"  {case.id} [{case.kind}/{case.section}] {case.note}")
            shown = case.text.replace("\n", " / ")
            lines.append(f"    text: {shown}")
            if result.false_positives:
                lines.append(
                    f"    false positive (found, wrong): "
                    f"{', '.join(sorted(result.false_positives))}"
                )
            if result.false_negatives:
                lines.append(
                    f"    false negative (missed):       "
                    f"{', '.join(sorted(result.false_negatives))}"
                )
    return "\n".join(lines)


def main() -> None:
    """Evaluate the default extractor on the default labels and print a report."""
    print(format_report(evaluate(load_cases())))


if __name__ == "__main__":
    main()
