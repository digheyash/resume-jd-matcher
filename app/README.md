# app/ - the Streamlit dashboard (Phase 6)

Start it from the project root, with the virtual environment active:

```bash
streamlit run app/streamlit_app.py
```

| File | What it does |
|---|---|
| `streamlit_app.py` | The page. Reads the two texts, calls `pipeline.analyze()` and draws the result. It is the only file that imports Streamlit. |
| `ui_helpers.py` | Plain functions behind the page: reading uploads, friendly error messages, number formatting, the score ring, skill chips and cards. |
| `summary.py` | The one-paragraph conclusion, built by fixed rules from the fields of the result (no language model). |
| `report.py` | The plain-text report for the "Download report (.txt)" button, and the two notes printed under the results. |
| `report_pdf.py` | The same report as a PDF ("Download report (PDF)"), made with `fpdf2`. |

Theme and server settings (light theme, 5 MB upload limit, no usage statistics,
no file watcher) are in `.streamlit/config.toml` at the project root. Because the
file watcher is off, restart the app after editing any `.py` file.

Everything except `streamlit_app.py` can be tested without a browser. The page
itself is tested with Streamlit's `AppTest` and a fake `analyze()`, so those
tests do not load the Sentence-BERT model. See the "Phase 6" section of the
main `README.md` for what the page shows and its limits.
