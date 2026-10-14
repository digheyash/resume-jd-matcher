"""Section detection: split a resume or a job description into sections.

How it works
------------
A line is treated as a HEADING when it is short and its whole text matches
a known heading name (for example "SKILLS", "Work Experience",
"Technical Skills:"). Every line after a heading belongs to that section
until the next heading appears. The inline form "Skills: Python, SQL" also
works: the part before the colon is the heading, the rest is content.

Run this on cleaned text (cleaner.clean_text), so bullets and contact
details are already gone.

Main entry point
----------------
detect_sections(text, kind="resume" | "jd") -> dict[str, str]

Special keys in the result
--------------------------
"header"    text before the first heading (name, contact line, job title)
"full_text" used alone when NO heading was found at all
"""

from __future__ import annotations

import re

HEADER_KEY = "header"
FULL_TEXT_KEY = "full_text"

# Headings longer than this (in characters or words) are never headings.
_MAX_HEADING_CHARS = 40
_MAX_HEADING_WORDS = 5

# canonical section name -> accepted heading spellings. Spellings are written
# in normalized form: lowercase letters and digits, single spaces, "&" -> "and",
# no punctuation (apostrophes become spaces: "you'll" -> "you ll").
RESUME_HEADINGS: dict[str, list[str]] = {
    "summary": [
        "summary", "professional summary", "career summary", "executive summary",
        "objective", "career objective", "profile", "professional profile",
        "about me", "about",
    ],
    "skills": [
        "skills", "technical skills", "key skills", "core skills",
        "core competencies", "skills summary", "technical proficiency",
        "technical expertise", "technologies", "tech stack", "skills and tools",
        "tools and technologies", "areas of expertise", "it skills",
        "computer skills",
    ],
    "experience": [
        "experience", "work experience", "professional experience",
        "employment history", "work history", "internship experience",
        "internships", "professional background", "career history",
        "relevant experience", "industry experience",
    ],
    "education": [
        "education", "academic background", "academic qualifications",
        "educational qualifications", "education and training", "academics",
    ],
    "projects": [
        "projects", "academic projects", "personal projects", "key projects",
        "project experience", "selected projects", "major projects",
        "technical projects",
    ],
    "certifications": [
        "certifications", "certificates", "licenses and certifications",
        "certifications and courses", "courses", "online courses", "training",
    ],
    "achievements": [
        "achievements", "awards", "honors and awards", "honours and awards",
        "accomplishments", "awards and achievements", "achievements and awards",
    ],
    "other": [
        "interests", "hobbies", "hobbies and interests", "languages",
        "references", "publications", "volunteering", "volunteer experience",
        "extracurricular activities", "extra curricular activities",
        "activities", "personal details", "personal information", "declaration",
    ],
}

JD_HEADINGS: dict[str, list[str]] = {
    "about": [
        "about the role", "about the job", "about us", "about the company",
        "overview", "role overview", "job description", "job summary",
        "position summary",
    ],
    "responsibilities": [
        "responsibilities", "key responsibilities", "roles and responsibilities",
        "role and responsibilities", "what you will do", "what you ll do",
        "duties",
    ],
    "requirements": [
        "requirements", "key requirements", "technical requirements",
        "required qualifications", "required skills", "skills required",
        "qualifications", "minimum qualifications", "basic qualifications",
        "must have", "must haves", "what we are looking for", "what you need",
        "what you bring", "who you are", "your profile",
    ],
    "nice_to_have": [
        "nice to have", "nice to haves", "good to have", "preferred qualifications",
        "preferred skills", "bonus", "bonus points", "desirable", "desired skills",
    ],
    "benefits": [
        "benefits", "what we offer", "perks", "perks and benefits", "compensation",
    ],
}


def _heading_key(text: str) -> str:
    """Normalize a possible heading so it can be looked up in the tables."""
    key = text.lower().replace("&", " and ")
    key = re.sub(r"[^a-z0-9 ]+", " ", key)  # drop punctuation
    key = re.sub(r"\s+", " ", key).strip()
    # Some PDFs space out letters: "S K I L L S" -> "skills".
    if re.fullmatch(r"(?:[a-z] )+[a-z]", key):
        key = key.replace(" ", "")
    return key


def _build_lookup(table: dict[str, list[str]]) -> dict[str, str]:
    """Turn {canonical: [spellings]} into {spelling: canonical}."""
    return {
        _heading_key(spelling): canonical
        for canonical, spellings in table.items()
        for spelling in spellings
    }


_LOOKUPS = {
    "resume": _build_lookup(RESUME_HEADINGS),
    "jd": _build_lookup(JD_HEADINGS),
}


def _match_heading(line: str, lookup: dict[str, str]) -> tuple[str, str] | None:
    """If the line is a heading, return (canonical_name, rest_of_line).

    rest_of_line is the content after a colon ("Skills: Python" -> "Python"),
    or an empty string for a plain heading line.
    """
    stripped = line.strip()
    if not stripped:
        return None

    head, colon, rest = stripped.partition(":")
    candidate = head if colon else stripped
    if len(candidate) > _MAX_HEADING_CHARS:
        return None

    # Count words AFTER normalizing, so "S K I L L S" counts as one word.
    key = _heading_key(candidate)
    if len(key.split()) > _MAX_HEADING_WORDS:
        return None

    canonical = lookup.get(key)
    if canonical is None:
        return None
    return canonical, (rest.strip() if colon else "")


def detect_sections(text: str, kind: str = "resume") -> dict[str, str]:
    """Split text into named sections.

    Args:
        text: cleaned text of a resume or a job description.
        kind: "resume" or "jd" (chooses which heading names are known).

    Returns:
        Ordered dict of section name -> section text. Sections that end up
        empty are left out. Repeated headings are merged. If no heading is
        found, returns {"full_text": text}. Empty input returns {}.

    Raises:
        ValueError: if kind is not "resume" or "jd".
    """
    if kind not in _LOOKUPS:
        raise ValueError(f"kind must be 'resume' or 'jd', not {kind!r}")
    lookup = _LOOKUPS[kind]

    buckets: dict[str, list[str]] = {}
    current = HEADER_KEY
    found_heading = False

    for line in text.split("\n"):
        match = _match_heading(line, lookup)
        if match is None:
            buckets.setdefault(current, []).append(line)
            continue
        found_heading = True
        current, rest = match
        buckets.setdefault(current, [])
        if rest:
            buckets[current].append(rest)

    if not found_heading:
        stripped = text.strip()
        return {FULL_TEXT_KEY: stripped} if stripped else {}

    sections: dict[str, str] = {}
    for name, lines in buckets.items():
        body = "\n".join(lines).strip()
        if body:
            sections[name] = body
    return sections
