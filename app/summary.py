"""One-paragraph, plain-language summary of an analysis result.

The paragraph is built by fixed rules from the fields that analyze() already
returns. There is no language model and no new data: every sentence can be
traced to a number or a skill name shown elsewhere on the page.

    build_summary(result) -> str

Sentence plan (a sentence is left out when it does not apply):
    1. Verdict and score (whole number).
    2. How many of the job's skills the resume shows.
    2b. (only when the job description had no requirement lines) a note that
       the score rests on the skill match alone.
    3. The biggest gaps: must-have skills first, then nice-to-have.
    4. A short pointer to the must-have skills (they count for more).
    5. A pointer to the warnings, when there are any.
"""

from __future__ import annotations

from resume_matcher.schemas import AnalysisResult, SkillGap

# How many skill names to spell out in one sentence before "and N more".
MAX_NAMES = 4

# The three importance labels produced by the analyzer (see scoring.py).
MUST_HAVE = "must-have"
NICE_TO_HAVE = "nice-to-have"
UNKNOWN = "unknown"

# Start of the warning that analyze() adds when the job description has no
# requirement lines. In that case analyze() reports the meaning match as a
# placeholder 0.0 and scores on skill coverage alone. This text is copied from
# resume_matcher/analyzer/analyzer.py; a test reads that file and fails if the
# two ever drift apart.
NO_REQUIREMENTS_MARK = "No requirement lines were found"

# Words used for each verdict in the first sentence.
VERDICT_PHRASES = {
    "Fit": "a good fit",
    "Partial Fit": "a partial fit",
    "No-Fit": "not a fit",
}


def semantic_not_used(result: AnalysisResult) -> bool:
    """True when the meaning match was not measured for this job description.

    analyze() still returns semantic_score = 0.0 then, so the number alone
    cannot tell "measured as zero" from "not measured". The warning can.
    """
    return any(NO_REQUIREMENTS_MARK in warning for warning in result.warnings)


def _join_names(names: list[str], limit: int = MAX_NAMES) -> str:
    """Join skill names for a sentence: 'A, B and C' or 'A, B, C, D and 2 more'."""
    shown = names[:limit]
    hidden = len(names) - len(shown)
    if hidden > 0:
        return ", ".join(shown) + f" and {hidden} more"
    if len(shown) <= 1:
        return "".join(shown)
    return ", ".join(shown[:-1]) + " and " + shown[-1]


def _skill_word(count: int) -> str:
    """'skill' for one, 'skills' otherwise."""
    return "skill" if count == 1 else "skills"


def group_missing(missing: list[SkillGap]) -> dict[str, list[str]]:
    """Split missing skills by importance, keeping their order, listing each once.

    Any importance label that is not must-have or nice-to-have is treated as
    "unknown", so a new label can never make the page crash.
    """
    groups: dict[str, list[str]] = {MUST_HAVE: [], NICE_TO_HAVE: [], UNKNOWN: []}
    for gap in missing:
        key = gap.importance if gap.importance in groups else UNKNOWN
        if gap.skill not in groups[key]:
            groups[key].append(gap.skill)
    return groups


def _opening_sentence(result: AnalysisResult) -> str:
    """Sentence 1: the verdict and the whole-number score."""
    score = max(0, min(100, round(result.overall_score)))
    phrase = VERDICT_PHRASES.get(result.verdict)
    if phrase is None:
        return f"This resume scored {score} out of 100 (verdict: {result.verdict})."
    return f"This resume is {phrase} for the role, with a score of {score} out of 100."


def _coverage_sentence(matched_count: int, required_count: int) -> str:
    """Sentence 2: how many of the job's skills the resume shows."""
    if required_count == 0:
        return (
            "No skills from the job description were recognised, so the score "
            "reflects the wording match only."
        )
    if matched_count == 0:
        if required_count == 1:
            return "It does not show the one skill the job description asks for."
        return (
            f"It shows none of the {required_count} skills "
            "the job description asks for."
        )
    if matched_count == required_count:
        if required_count == 1:
            return "It shows the one skill the job description asks for."
        if required_count == 2:
            return "It shows both skills the job description asks for."
        return f"It shows all {required_count} skills the job description asks for."
    return (
        f"It shows {matched_count} of the {required_count} skills "
        "the job description asks for."
    )


def _gap_sentences(groups: dict[str, list[str]]) -> str:
    """Sentence 3 (and a short follow-up): the biggest gaps, by importance."""
    must = groups[MUST_HAVE]
    nice = groups[NICE_TO_HAVE]
    unknown = groups[UNKNOWN]

    if not (must or nice or unknown):
        return "No required skills are missing."

    if must:
        if len(must) == 1:
            text = f"The biggest gap is the must-have skill {must[0]}"
        else:
            text = f"The biggest gaps are the must-have skills {_join_names(must)}"
        if nice:
            verb = "is" if len(nice) == 1 else "are"
            text += f", while {_join_names(nice)} {verb} only nice-to-have"
        text += "."
    elif nice:
        if unknown:
            if len(nice) == 1:
                text = f"The missing nice-to-have skill is {nice[0]}."
            else:
                text = f"The missing nice-to-have skills are {_join_names(nice)}."
        elif len(nice) == 1:
            text = f"The only missing skill, {nice[0]}, is nice-to-have."
        else:
            text = f"The missing skills are all nice-to-have: {_join_names(nice)}."
    else:
        if len(unknown) == 1:
            return (
                f"The missing skill is {unknown[0]}; the job description "
                "does not say how important it is."
            )
        return (
            f"The missing skills are {_join_names(unknown)}; the job description "
            "does not say how important they are."
        )

    if unknown:
        text += (
            f" {len(unknown)} other missing {_skill_word(len(unknown))} "
            "could not be rated for importance."
        )
    return text


def build_summary(result: AnalysisResult) -> str:
    """Return the summary paragraph for one analysis result (plain text)."""
    groups = group_missing(result.missing_skills)
    matched_count = len(dict.fromkeys(result.matched_skills))
    missing_count = sum(len(names) for names in groups.values())

    required_count = matched_count + missing_count

    sentences = [
        _opening_sentence(result),
        _coverage_sentence(matched_count, required_count),
    ]
    if semantic_not_used(result):
        sentences.insert(
            2,
            "The job description has no clear list of requirements, so this "
            "score rests on the skill match alone.",
        )
    # With no recognised JD skills there is nothing to be missing, and the
    # coverage sentence has already said so.
    if required_count > 0:
        sentences.append(_gap_sentences(groups))
    if groups[MUST_HAVE]:
        sentences.append(
            "If you really have any of the must-have skills, make sure they "
            "appear clearly on your resume; they count for more than the "
            "nice-to-have ones."
        )
    if result.warnings:
        sentences.append(
            "Please read the notes under the results, as they may affect how "
            "reliable this is."
        )
    return " ".join(sentences)
