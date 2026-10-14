# Resume-to-JD Matcher & Skill Gap Analyzer

Compares one resume with one job description, gives a match score and a
Fit / Partial Fit / No-Fit verdict, and lists matched and missing skills.

## Setup

```bash
python -m venv .venv

# Windows (Git Bash):   source .venv/Scripts/activate
# Windows (cmd):        .venv\Scripts\activate
# Mac / Linux:          source .venv/bin/activate

pip install -r requirements.txt
python -m pytest
```

## Project layout

- `src/resume_matcher/` - the engine (parsing, skills, matching, analyzer)
- `app/` - the web dashboard (added in Phase 6)
- `data/` - sample files, labelled test pairs, raw and processed data
- `tests/` - automated tests
- `docs/` - architecture and project documentation

## Status

Phase 1 (project skeleton) in progress. See `docs/ARCHITECTURE.md`.