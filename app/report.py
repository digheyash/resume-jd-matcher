"""Plain-text report for the "Download report" button, and the page notes.

    build_report(result, summary, warnings, generated_at=None) -> str

The report repeats what the page shows: score, verdict, the two raw numbers,
the summary paragraph, the skill lists and the warnings. It ends with the same
two notes that the page prints under the results (HOW_TO_READ and
GUIDE_NOT_VERDICT), so a report read on its own is not mistaken for more than
it is.

If the scoring is ever retuned, or the evaluation set changes, revisit the two
note texts below: they describe the Phase 5 state of the project.
"""

from __future__ import annotations

import textwrap
from datetime import datetime

from resume_matcher.schemas import AnalysisResult
from summary import MUST_HAVE, NICE_TO_HAVE, UNKNOWN, group_missing
from ui_helpers import (
    GROUP_TITLES,
    coverage_line,
    format_score,
    meaning_match_line,
)

# Width of the text report, in characters.
WIDTH = 78

HOW_TO_READ = (
    "The overall score blends two things: how closely the resume text matches "
    "the job requirements in meaning, and how many of the required skills it "
    "shows. Must-have skills count more than nice-to-have ones, and the meaning "
    "match counts for more than the skill count. The two small numbers beside "
    "the score are the raw values before this blending."
)

GUIDE_NOT_VERDICT = (
    "The scoring was tuned on only 24 rated resume and job pairs, all with "
    "synthetic text, so treat the score and verdict as a rough guide. Skills "
    "are found by matching a dictionary of known skills, so a skill described "
    "in other words may be reported as missing."
)


def _wrap(text: str, indent: int = 2) -> str:
    """Wrap a paragraph to WIDTH with a left indent."""
    pad = " " * indent
    # break_on_hyphens=False keeps words such as "nice-to-have" on one line.
    return textwrap.fill(
        text,
        width=WIDTH,
        initial_indent=pad,
        subsequent_indent=pad,
        break_on_hyphens=False,
    )


def _bullet_lines(names: list[str], indent: int = 4) -> list[str]:
    return [f"{' ' * indent}- {name}" for name in names]


def _skill_block(title: str, note: str, names: list[str]) -> list[str]:
    """A heading with a count, a one-line meaning, then a bullet per skill."""
    lines = [f"{title} ({len(names)})", f"  {note}"]
    lines += _bullet_lines(names, indent=4) if names else ["    None."]
    return lines


def build_report(
    result: AnalysisResult,
    summary: str,
    warnings: list[str],
    generated_at: datetime | None = None,
) -> str:
    """Return the whole report as plain text (UTF-8 safe, ends with a newline)."""
    lines = ["RESUME MATCH REPORT"]
    if generated_at is not None:
        lines.append(f"Generated: {generated_at:%Y-%m-%d %H:%M}")

    lines += [
        "",
        "OVERALL",
        f"  Score:           {format_score(result.overall_score)} / 100",
        f"  Verdict:         {result.verdict}",
        f"  Meaning match:   {meaning_match_line(result)}",
        f"  Skill coverage:  {coverage_line(result)}",
        "",
        "SUMMARY",
        _wrap(summary),
        "",
        "SKILLS TO ADD (most important first)",
    ]

    grouped = group_missing(result.missing_skills)
    listed_any = False
    for key in (MUST_HAVE, NICE_TO_HAVE, UNKNOWN):
        names = grouped[key]
        if not names:
            continue
        listed_any = True
        lines.append(f"  {GROUP_TITLES[key]} ({len(names)})")
        lines += _bullet_lines(names, indent=4)
    if not listed_any:
        lines.append("  No required skills are missing.")

    lines.append("")
    lines += _skill_block(
        "SKILLS YOU ALREADY HAVE",
        "Asked for by the job description and found on the resume.",
        list(dict.fromkeys(result.matched_skills)),
    )
    lines.append("")
    lines += _skill_block(
        "EXTRA SKILLS",
        "On the resume, but not asked for in this job description.",
        list(dict.fromkeys(result.extra_skills)),
    )

    if warnings:
        lines += ["", "THINGS TO CHECK"]
        for warning in warnings:
            lines.append(
                textwrap.fill(
                    warning,
                    width=WIDTH,
                    initial_indent="  - ",
                    subsequent_indent="    ",
                    break_on_hyphens=False,
                )
            )

    lines += [
        "",
        "ABOUT THIS REPORT",
        _wrap(HOW_TO_READ),
        "",
        _wrap(GUIDE_NOT_VERDICT),
    ]
    return "\n".join(lines) + "\n"
