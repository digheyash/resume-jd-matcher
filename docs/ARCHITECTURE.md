# Architecture (Phase 1 draft)

```mermaid
flowchart LR
  A[Resume PDF/Text] --> B[Parser + Preprocessor]
  C[Job Description] --> B
  B --> D[Skill Extractor]
  B --> E[Vectorizer + Cosine]
  D --> F[Matched / Missing / Extra]
  E --> G[Semantic score]
  F --> H[Analyzer + Calibration]
  G --> H
  H --> I[AnalysisResult]
  I --> J[Streamlit UI]
```

## Rules

1. The UI calls only `pipeline.analyze()`. Modules never import the UI.
2. Every stage passes data forward in the shapes defined in `schemas.py`.
3. All paths and tunable numbers live in `config.py`, never hard-coded.