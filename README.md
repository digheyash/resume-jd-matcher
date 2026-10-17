# Resume-to-JD Matcher & Skill Gap Analyzer

Compares one resume with one job description, gives a match score and a
Fit / Partial Fit / No-Fit verdict, and lists matched and missing skills.

**Status: work in progress.** Phases 1-4 are built: parsing, skill
extraction, and vectorization / matching. The final score, the verdict and the
ranked skill-gap report (Phase 5) and the web dashboard (Phase 6) are not
built yet, and `pipeline.analyze()` is still a stub.

## Setup

```bash
python -m venv .venv

# Windows (Git Bash):   source .venv/Scripts/activate
# Windows (cmd):        .venv\Scripts\activate
# Mac / Linux:          source .venv/bin/activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m pytest
```

The first use of Sentence-BERT downloads the model `all-MiniLM-L6-v2`
(about 90 MB), so it needs internet once; after that it is cached.
Developed and tested on Windows with Python 3.14.6.

## Project layout

```
src/resume_matcher/
    config.py, schemas.py, pipeline.py
    parsing/    load PDF / text, clean, lemmatise, detect sections   (Phase 2)
    skills/     skill ontology, extractor, set comparison, evaluation (Phase 3)
    matching/   chunking, TF-IDF, Sentence-BERT, requirement-level matching,
                labelled pairs, method comparison                     (Phase 4)
    analyzer/   final score and verdict                               (Phase 5)
app/            web dashboard                                         (Phase 6)
data/
    samples/    one sample resume and one sample JD
    skills.json the skill ontology (85 skills)
    labels/     skill_labels.json (37 snippets), pairs.csv (24 rated pairs)
    synthetic/  10 fictional resumes and 6 fictional job descriptions
    processed/  generated outputs
scripts/        make_synthetic_pairs.py
tests/          automated tests
docs/           architecture and project documentation (ARCHITECTURE.md)
```

## Phases

| Phase | Content | Status |
|---|---|---|
| 1 | Skeleton, config, schemas, smoke tests | done |
| 2 | PDF / text loading, cleaning, lemmatising, section detection | done |
| 3 | Skill ontology, regex extractor, set logic, precision / recall / F1 | done |
| 4 | TF-IDF and Sentence-BERT similarity, requirement-level matching, comparison on rated pairs | done |
| 5 | Final score, Fit / Partial Fit / No-Fit thresholds, ranked missing skills, calibration | next |
| 6 | Streamlit dashboard | planned |
| 7 | Testing, edge cases, evaluation chapter, report | planned |

## Commands

```bash
python -m pytest                                          # all tests

# Phase 3: skill-extraction quality on the 37 labelled snippets
PYTHONPATH=src python -m resume_matcher.skills.evaluation

# Phase 4: check the rated pairs file, then compare the three methods
PYTHONPATH=src python -m resume_matcher.matching.pairs
PYTHONPATH=src python -m resume_matcher.matching.compare_methods

# Rebuild the synthetic texts and an EMPTY pairs.csv (refuses if pairs.csv exists)
python scripts/make_synthetic_pairs.py
```

## Phase 4: how similarity is measured

Three methods are scored against the same 24 rated resume-JD pairs:

- **TF-IDF**: cosine similarity of word-weight vectors on the lemmatised text.
  The vocabulary and IDF weights are fitted once on all the texts in the
  pairs file; no ratings are used in the fit.
- **Sentence-BERT, whole document**: the model reads at most 256 tokens, so a
  text is cut into chunks of at most 200 tokens (a placeholder size, not
  tuned), each chunk is embedded, and the chunk vectors are averaged weighted
  by length. Two documents are compared by cosine similarity.
- **Sentence-BERT, requirement lines**: for each line of the JD's
  "Requirements" section, the most similar resume line is found; the score is
  the mean of those best scores. This also shows which resume line answered
  which requirement.

Cosine values are **not probabilities** and are not percentages. Turning them
into a score and a verdict is Phase 5.

## Phase 4 results

24 resume-JD pairs, rated 0-100 with a Fit / Partial Fit / No-Fit verdict by
the project author (13 No-Fit, 3 Partial Fit, 8 Fit). Agreement between each
method's score and the rating, measured on one run:

| Method | Pearson r | Spearman rho | Seconds per pair |
|---|---|---|---|
| TF-IDF | 0.833 | 0.712 | 0.0204 |
| Sentence-BERT, whole document | 0.681 | 0.630 | 0.0726 |
| Sentence-BERT, requirement lines | 0.853 | 0.793 | 0.0695 |

One-time costs: model load 21.65 s, TF-IDF fit 0.011 s. Times are single-run
laptop CPU measurements; TF-IDF's time includes the spaCy lemmatisation it
needs.

How to read this:

- All three methods agree with the ratings clearly.
- Whole-document Sentence-BERT scored lowest. The more modern method did
  **not** beat TF-IDF here, and TF-IDF and requirement-line Sentence-BERT are
  too close to separate on 24 pairs (an exploratory bootstrap during
  development found no reliable difference between them; it is not yet part of
  this repository).
- No method separates Fit from the other verdicts cleanly: for each, the
  lowest Fit score is below the highest non-Fit score. The raw scales also
  differ between methods, so no common threshold exists. This is the main
  problem for Phase 5.
- A resume that describes ML work without naming tools (pair 9, rated 70) was
  ranked low by all three methods, and a resume that lists every skill but has
  little substance (pairs 1 and 13) was ranked too high by TF-IDF and
  whole-document Sentence-BERT.

## Known limits

- **The evaluation set is small and synthetic.** The 10 resumes and 6 JDs are
  fictional and were written by an AI assistant; each JD and its matching
  resume share much of the same vocabulary, which probably favours TF-IDF.
  Only 3 pairs are Partial Fit, so verdict-level results will rest on few
  borderline examples. Treat all numbers as indicative.
- The skill-extraction scores (precision 0.991, recall 0.955, F1 0.973) come
  from a 37-snippet development set written by the assistant and are
  optimistic.
- Scanned PDFs give no text (no OCR); multi-column layouts may scramble order.
- Section detection knows common English headings only.
- The skill ontology has only 85 skills and cannot detect skills outside it.
- The lemmatiser turns "data" (as a noun in some phrases) into "datum" and
  "AWS" into "aw"; matching still works because both sides change alike, but
  TF-IDF treats "data" and "datum" as different words.
- Chunk size (200 tokens), the length-weighted average and the choice of
  requirement sections are untuned defaults.
