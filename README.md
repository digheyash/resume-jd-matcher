# Resume-to-JD Matcher & Skill Gap Analyzer

Compares one resume with one job description, gives a match score out of 100 and
a Fit / Partial Fit / No-Fit verdict, and lists the matched, missing and extra
skills. Missing skills are ranked by importance.

**Status: work in progress.** Phases 1-5 are built: parsing, skill extraction,
similarity measurement, and the scoring and verdict stage. `pipeline.analyze()`
now returns a full result. The web dashboard (Phase 6) is not built yet.

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

## How it works

```
resume + JD -> parse and clean -> skill extraction  -> skill coverage  \
                                -> Sentence-BERT     -> semantic score  -> score -> verdict
                                   (requirement lines)                       |
                                                            ranked missing skills
```

1. **Parse** both texts: clean them and split them into sections.
2. **Skills:** find skills from a 85-skill ontology and compare the two sets
   (matched, missing, extra). A skill asked for under "Requirements" counts as
   a must-have; one under "Nice to have" counts less.
3. **Semantic score:** for each requirement line of the JD, find the most
   similar resume line with Sentence-BERT; the score is the mean of those best
   matches. This also shows which resume line answered which requirement.
4. **Final score:** the raw cosine is rescaled to 0-1 and blended with the
   weighted skill coverage (80 % semantic, 20 % coverage). Two thresholds turn
   the score into Fit / Partial Fit / No-Fit.
5. **Gap report:** missing skills are listed must-have first, then
   nice-to-have, then unknown.

If the JD has no requirement lines, the score uses skill coverage only; if it
has no recognised skills, it uses the semantic part only. Each case adds a
warning. If neither signal exists, `analyze()` raises `AnalysisError`.

## Project layout

```
src/resume_matcher/
    config.py, schemas.py, pipeline.py
    parsing/    load PDF / text, clean, lemmatise, detect sections (Phase 2)
    skills/     skill ontology, extractor, set comparison, evaluation (Phase 3)
    matching/   chunking, TF-IDF, Sentence-BERT, requirement-level matching,
                labelled pairs, method comparison  (Phase 4)
    analyzer/   scoring formula, ranked gaps, analyzer, tuning  (Phase 5)
app/            web dashboard  (Phase 6)
data/
    samples/    one sample resume and one sample JD
    skills.json the skill ontology (85 skills)
    labels/     skill_labels.json (37 snippets), pairs.csv (24 rated pairs)
    synthetic/  10 fictional resumes and 6 fictional job descriptions
    processed/  generated outputs
scripts/        make_synthetic_pairs.py, show_pair_signals.py
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
| 5 | Final score, verdict thresholds, ranked missing skills, tuning and evaluation | done |
| 6 | Streamlit dashboard | planned |
| 7 | Testing, edge cases, evaluation chapter, report | planned |

## Commands

```bash
python -m pytest                                          # all tests

# Analyse one resume against one JD (defaults to the two sample files)
PYTHONPATH=src python -m resume_matcher.pipeline
PYTHONPATH=src python -m resume_matcher.pipeline --resume my_resume.txt --jd job.txt

# Phase 5: tune the score on the rated pairs and print the evaluation
PYTHONPATH=src python -m resume_matcher.analyzer.tuning

# Phase 3: skill-extraction quality on the 37 labelled snippets
PYTHONPATH=src python -m resume_matcher.skills.evaluation

# Phase 4: check the rated pairs file, then compare the three methods
PYTHONPATH=src python -m resume_matcher.matching.pairs
PYTHONPATH=src python -m resume_matcher.matching.compare_methods

# Table of every signal for the 24 pairs (needs the Phase 4 results file)
PYTHONPATH=src python scripts/show_pair_signals.py

# Rebuild the synthetic texts and an EMPTY pairs.csv (refuses if pairs.csv exists)
python scripts/make_synthetic_pairs.py
```

## Phase 4: how similarity is measured

Three methods were scored against the same 24 rated resume-JD pairs:

- **TF-IDF**: cosine similarity of word-weight vectors on the lemmatised text.
  The vocabulary and IDF weights are fitted once on all the texts in the
  pairs file; no ratings are used in the fit.
- **Sentence-BERT, whole document**: the model reads at most 256 tokens, so a
  text is cut into chunks of at most 200 tokens (a placeholder size, not
  tuned), each chunk is embedded, and the chunk vectors are averaged weighted
  by length. Two documents are compared by cosine similarity.
- **Sentence-BERT, requirement lines**: for each line of the JD's
  "Requirements" section, the most similar resume line is found; the score is
  the mean of those best scores.

Cosine values are **not probabilities** and are not percentages; each method
has its own scale. Phase 5 maps one of them onto a 0-100 score.

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
  too close to separate on 24 pairs.
- No method separates Fit from the other verdicts cleanly, and the raw scales
  differ, so no common threshold exists. Phase 5 addresses this.
- A resume that describes ML work without naming tools (pair 9, rated 70) was
  ranked low by all three methods, and a resume that lists every skill but has
  little substance (pairs 1 and 13) was ranked too high by TF-IDF and
  whole-document Sentence-BERT.

## Phase 5: score and verdict

The semantic signal is the requirement-line method, chosen because it agreed
best with the ratings and because it is explainable. Five numbers turn the two
signals into a score and a verdict. They are fitted by
`analyzer/tuning.py` and stored in `config.py`:

| Number | Value | How it was set |
|---|---|---|
| `SEMANTIC_ANCHOR_LOW` | 0.2078 | lowest raw requirement-line cosine among the pairs (maps to 0) |
| `SEMANTIC_ANCHOR_HIGH` | 0.5901 | highest raw cosine among the pairs (maps to 1) |
| `WEIGHT_SEMANTIC` | 0.8 | best Spearman correlation on a grid of 0.2 to 0.8 (the best value is the upper limit of that range) |
| `PARTIAL_THRESHOLD` | 0.4669 | cut-off with the best 3-class accuracy |
| `FIT_THRESHOLD` | 0.5926 | cut-off with the best 3-class accuracy |

Design choices that were **not** tuned: the weight search is limited to 0.2-0.8
so both signals always count, and a must-have skill counts 1.0 while a
nice-to-have or unclassified skill counts 0.5.

### Phase 5 results (24 pairs; output of `analyzer/tuning.py`)

Each signal on its own, against the ratings:

| Signal | Pearson r | Spearman rho |
|---|---|---|
| Raw requirement-line cosine | 0.853 | 0.793 |
| Plain skill coverage | 0.639 | 0.584 |
| Weighted skill coverage | 0.651 | 0.593 |

The final score, in two views. **In-sample** means the numbers were fitted on
the same pairs and are optimistic. **Leave-one-out** refits all five numbers on
23 pairs and scores the 24th, once per pair, so every pair is scored by a fit
that never saw it.

| | Pearson r | Spearman rho | Verdict accuracy |
|---|---|---|---|
| In-sample | 0.841 | 0.789 | 0.833 |
| Leave-one-out (held out) | 0.832 | 0.774 | 0.792 |
| Always answering "No-Fit" | - | - | 0.542 |

Held-out confusion matrix (rows = rating, columns = prediction):

| | Fit | Partial Fit | No-Fit |
|---|---|---|---|
| **Fit** (8) | 8 | 0 | 0 |
| **Partial Fit** (3) | 0 | 1 | 2 |
| **No-Fit** (13) | 1 | 2 | 10 |

Across the 24 leave-one-out refits the tuned weight stayed between 0.7 and 0.8,
the Partial Fit threshold moved between 0.3745 and 0.4878, and the Fit
threshold between 0.5221 and 0.6141.

How to read this:

- Fit versus No-Fit is separated reasonably well: all 8 Fit pairs are predicted
  Fit, and held-out accuracy (19 of 24) is clearly above the 13 of 24 that
  always answering "No-Fit" would give.
- **Partial Fit is weak.** Only 1 of the 3 Partial Fit pairs is right. Pair 9
  (rated 70, ML work without tool names) scores 25.0 and pair 21 (rated 55)
  scores 38.3, so both come out as No-Fit. With 3 examples nothing firmer can
  be said, and the Partial Fit threshold is the least stable number.
- **The score over-rates some pairs.** Pair 12 (rated 40) scores 66.0 and is
  predicted Fit; pair 1 (rated 5, a keyword-heavy resume) scores 53.6 and is
  predicted Partial Fit.
- **Skill coverage adds little on this data.** Alone it correlates less with
  the ratings than the semantic score does, and the blended score's in-sample
  Pearson (0.841) is slightly below the semantic score alone (0.853). It is
  kept because it keeps the score consistent with the gap report. The weighting
  of must-have skills raised its correlation only from 0.639 to 0.651, too
  little to call real on 24 pairs.

Example run on the sample resume and sample JD (not a measurement of
accuracy): overall score 52.8 / 100, verdict Partial Fit, raw semantic match
0.424, skill coverage 36 % (5 of 14 JD skills). Six must-have skills are
reported missing (Deep Learning, Docker, Model Evaluation, PyTorch, REST API,
TensorFlow). The resume does say its project was "evaluated with precision,
recall and F1", but the extractor does not map that wording to Model
Evaluation, so that gap is a false alarm.

## Known limits

- **The evaluation set is small and synthetic.** The 10 resumes and 6 JDs are
  fictional and were written by an AI assistant; each JD and its matching
  resume share much of the same vocabulary, which probably favours TF-IDF.
  The ratings are the project author's. Only 3 pairs are Partial Fit, so
  verdict-level results rest on few borderline examples. Treat all numbers as
  indicative.
- **The tuned numbers were fitted on all 24 pairs.** The semantic method and
  the must-have weighting were also chosen after looking at these same pairs,
  which leave-one-out does not account for, so even the held-out numbers are
  somewhat optimistic.
- The semantic weight was only searched between 0.2 and 0.8; the data preferred
  the upper limit.
- The skill-extraction scores (precision 0.991, recall 0.955, F1 0.973) come
  from a 37-snippet development set written by the assistant and are
  optimistic.
- The extractor is dictionary-based: a skill described in other words (for
  example "evaluated with precision, recall and F1" for Model Evaluation) is
  reported as missing.
- Learning suggestions for missing skills are not included; the result lists
  the gaps and their importance only.
- Scanned PDFs give no text (no OCR); multi-column layouts may scramble order.
- Section detection knows common English headings only.
- The skill ontology has only 85 skills and cannot detect skills outside it.
- The lemmatiser turns "data" (as a noun in some phrases) into "datum" and
  "AWS" into "aw"; matching still works because both sides change alike, but
  TF-IDF treats "data" and "datum" as different words.
- Chunk size (200 tokens), the length-weighted average and the choice of
  requirement sections are untuned defaults.
