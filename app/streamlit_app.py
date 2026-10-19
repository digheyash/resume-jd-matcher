"""Streamlit dashboard for the Resume-to-JD Matcher (Phase 6).

Run it from the project root:

    streamlit run app/streamlit_app.py

Streamlit re-runs this whole script from top to bottom every time the user
clicks or types. Two things follow from that:
  * values that must survive a click (the latest result, error messages) live
    in st.session_state;
  * the slow part (loading the Sentence-BERT model) is cached inside
    resume_matcher.pipeline, which keeps ONE analyzer for the whole server.

The page itself is thin. The real work is done by plain, tested functions:
    ui_helpers.py  reading uploads, friendly errors, chips, ring, cards
    summary.py     the one-paragraph conclusion
    report.py      the downloadable text report and the two notes
    report_pdf.py  the same report as a PDF file
and by pipeline.analyze(), the only entry point into the matching engine.
"""

from __future__ import annotations

import html
import logging
import sys
from datetime import datetime
from pathlib import Path

# Make "resume_matcher" (in src/) and the helper modules (in app/) importable
# when the app is started with "streamlit run" from the project root.
ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT / "app"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import streamlit as st  # noqa: E402

import ui_helpers as ui  # noqa: E402
from report import GUIDE_NOT_VERDICT, HOW_TO_READ, build_report  # noqa: E402
from report_pdf import build_pdf  # noqa: E402
from resume_matcher import pipeline  # noqa: E402
from resume_matcher.analyzer.analyzer import AnalysisError  # noqa: E402
from resume_matcher.parsing.loader import (  # noqa: E402
    DocumentLoadError,
    load_text_file,
)
from resume_matcher.schemas import AnalysisResult  # noqa: E402
from summary import build_summary  # noqa: E402

LOGGER = logging.getLogger("resume_matcher.app")

SAMPLES_DIR = ROOT / "data" / "samples"
SAMPLE_RESUME = SAMPLES_DIR / "sample_resume.txt"
SAMPLE_JD = SAMPLES_DIR / "sample_jd.txt"

MODE_UPLOAD = "Upload file"
MODE_PASTE = "Paste text"

LOADING_MESSAGE = (
    "Loading the language model. This takes about 20 seconds the first time."
)
ANALYZING_MESSAGE = "Analyzing..."

# ----------------------------------------------------------------------
# Page styling (one <style> block). Streamlit's own widgets are restyled here;
# the cards, chips and ring carry their own inline styles (see ui_helpers.py).
# ----------------------------------------------------------------------
PAGE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

.stApp, .stApp h1, .stApp p, .stApp label, .stApp textarea, .stApp input,
.stApp button, .stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stCaptionContainer"] {
  font-family: Inter, system-ui, -apple-system, "Segoe UI", sans-serif;
}

/* Hide Streamlit's own top bar, menu and footer for a clean page. */
#MainMenu, footer, header[data-testid="stHeader"],
[data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }

/* A centred column of readable width. */
.block-container, [data-testid="stMainBlockContainer"] {
  max-width: 1120px !important;
  padding: 2rem 2rem 4rem 2rem !important;
}

/* The two named containers (see render_inputs) become the white input cards.
   Streamlit adds the class "st-key-<name>" to a container that has a key. */
.st-key-resume_card, .st-key-jd_card {
  background: #FFFFFF !important;
  border: 1px solid #ECE8E1 !important;
  border-radius: 16px !important;
  box-shadow: 0 1px 2px rgba(31,36,48,0.04), 0 8px 24px rgba(31,36,48,0.04);
  padding: 0.75rem 1rem 1rem 1rem !important;
}

.stTextArea textarea {
  background: #FCFBF9;
  border: 1px solid #DDD8CF;
  border-radius: 12px;
  color: #1F2430;
  font-size: 15px;
  line-height: 1.55;
}
.stTextArea textarea:focus { border-color: #4F46E5; box-shadow: 0 0 0 1px #4F46E5; }
[data-testid="InputInstructions"] { display: none; }

[data-testid="stFileUploaderDropzone"] {
  background: #FCFBF9;
  border: 1.5px dashed #CFC9BE;
  border-radius: 12px;
  padding: 28px 16px;
}

.stButton > button, .stDownloadButton > button {
  border-radius: 10px;
  font-weight: 550;
  min-height: 44px;
  padding: 0 22px;
}
[data-testid="stBaseButton-primary"] {
  background: #4F46E5;
  border: 0;
  color: #FFFFFF;
  box-shadow: 0 6px 16px rgba(79,70,229,0.28);
}
[data-testid="stBaseButton-primary"]:hover { background: #4338CA; color: #FFFFFF; }
[data-testid="stBaseButton-secondary"] {
  background: #FFFFFF;
  border: 1px solid #DDD8CF;
  color: #1F2430;
}
[data-testid="stBaseButton-secondary"]:hover { border-color: #4F46E5; color: #4F46E5; }
</style>
"""

HEADER_HTML = (
    '<div style="display:flex;align-items:center;gap:12px">'
    '<div style="width:36px;height:36px;border-radius:10px;background:#4F46E5;'
    'display:flex;align-items:center;justify-content:center">'
    '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<path d="M5 12.5l4.5 4.5L19 7.5"></path></svg></div>'
    "<div>"
    f'<div style="font-size:17px;font-weight:650;letter-spacing:-0.01em;color:{ui.INK}">'
    "Resume Matcher</div>"
    f'<div style="font-size:13px;color:{ui.MUTED}">Resume vs job description, '
    "with the skill gap</div></div></div>"
)

HERO_HTML = (
    '<div style="max-width:760px;margin:28px 0 12px 0">'
    f'<h1 style="margin:0 0 10px 0;padding:0;font-size:40px;line-height:1.12;'
    f'font-weight:700;letter-spacing:-0.025em;color:{ui.INK}">'
    "How well does your resume fit the job?</h1>"
    f'<p style="margin:0;font-size:17px;line-height:1.55;color:{ui.MUTED}">'
    "Add a resume and a job description. You get a match score, a verdict, "
    "and the skills you are missing, ranked by importance.</p></div>"
)


def section_title_html(text: str) -> str:
    """Heading used inside the input cards."""
    return (
        f'<div style="font-size:18px;font-weight:650;letter-spacing:-0.01em;'
        f'color:{ui.INK};margin:4px 0 8px 0">{html.escape(text)}</div>'
    )


def note_html(title: str, text: str) -> str:
    """A quiet grey note box (used for 'How to read this' and the honesty note)."""
    return (
        '<div style="background:#F5F3EE;border-radius:14px;padding:22px 24px;'
        'margin-bottom:16px">'
        f'<div style="font-size:14px;font-weight:650;color:{ui.INK};margin-bottom:8px">'
        f"{html.escape(title)}</div>"
        f'<div style="font-size:14px;line-height:1.6;color:{ui.MUTED}">'
        f"{html.escape(text)}</div></div>"
    )


# ----------------------------------------------------------------------
# State and actions
# ----------------------------------------------------------------------
def init_state() -> None:
    """Give every session-state key a starting value (runs on every rerun)."""
    defaults = {
        "resume_mode": MODE_UPLOAD,
        "resume_text": "",
        "jd_text": "",
        "result": None,
        "file_warnings": [],
        "error": None,       # (title, message) shown in a red box
        "problems": [],      # missing-input messages
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def failure_message(exc: Exception) -> tuple[str, str]:
    """Friendly (title, message) for an error; unexpected ones go to the log."""
    if not isinstance(exc, (DocumentLoadError, AnalysisError)):
        LOGGER.exception("Unexpected error while analysing")
    return ui.friendly_error(exc)


def load_sample() -> None:
    """Button callback: put the sample resume and JD into the two text boxes."""
    st.session_state.update(result=None, file_warnings=[], error=None, problems=[])
    try:
        resume = load_text_file(SAMPLE_RESUME).text
        jd = load_text_file(SAMPLE_JD).text
    except Exception as exc:  # noqa: BLE001 - shown as a friendly message
        st.session_state["error"] = failure_message(exc)
        return
    st.session_state["resume_mode"] = MODE_PASTE
    st.session_state["resume_text"] = resume
    st.session_state["jd_text"] = jd


def collect_resume() -> tuple[str, list[str], tuple[str, str] | None]:
    """Return (resume_text, file_warnings, error) for the current resume input."""
    if st.session_state["resume_mode"] == MODE_PASTE:
        return st.session_state.get("resume_text", ""), [], None
    uploaded = st.session_state.get("resume_file")
    if uploaded is None:
        return "", [], None
    try:
        document = ui.read_uploaded_file(uploaded.name, uploaded.getvalue())
    except Exception as exc:  # noqa: BLE001 - shown as a friendly message
        return "", [], failure_message(exc)
    return document.text, list(document.warnings), None


def model_is_cold() -> bool:
    """True when the first analysis will still have to load the language model."""
    cache_info = getattr(pipeline.get_default_analyzer, "cache_info", None)
    return cache_info is not None and cache_info().currsize == 0


def run_analysis() -> None:
    """Button action: read the inputs, call pipeline.analyze(), store the outcome."""
    st.session_state.update(result=None, file_warnings=[], error=None, problems=[])

    resume_text, file_warnings, error = collect_resume()
    if error is not None:
        st.session_state["error"] = error
        return

    jd_text = st.session_state.get("jd_text", "")
    problems = ui.input_problems(resume_text, jd_text)
    if problems:
        st.session_state["problems"] = problems
        return

    message = LOADING_MESSAGE if model_is_cold() else ANALYZING_MESSAGE
    with st.spinner(message):
        try:
            result = pipeline.analyze(resume_text, jd_text)
        except Exception as exc:  # noqa: BLE001 - shown as a friendly message
            st.session_state["error"] = failure_message(exc)
            return
    st.session_state["result"] = result
    st.session_state["file_warnings"] = file_warnings


# ----------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------
def show(markup: str) -> None:
    """Draw one HTML piece."""
    if markup:
        st.markdown(markup, unsafe_allow_html=True)


def render_header() -> None:
    left, right = st.columns([4, 1], vertical_alignment="center")
    with left:
        show(HEADER_HTML)
    with right:
        st.button("Load sample pair", key="load_sample", on_click=load_sample,
                  use_container_width=True)
    show(HERO_HTML)


def render_inputs() -> None:
    resume_col, jd_col = st.columns(2, gap="medium")

    with resume_col, st.container(border=True, key="resume_card"):
        show(section_title_html("Resume"))
        st.radio("Resume input", [MODE_UPLOAD, MODE_PASTE], key="resume_mode",
                 horizontal=True, label_visibility="collapsed")
        if st.session_state["resume_mode"] == MODE_UPLOAD:
            st.file_uploader("Resume file", type=["pdf", "txt", "md"],
                             key="resume_file", label_visibility="collapsed")
            st.caption(
                f"Text-based PDF, .txt or .md, up to {ui.MAX_UPLOAD_MB} MB. "
                "A scanned PDF has no readable text; paste the text instead."
            )
        else:
            st.text_area("Resume text", key="resume_text", height=210,
                         placeholder="Paste your resume text here.",
                         label_visibility="collapsed")

    with jd_col, st.container(border=True, key="jd_card"):
        show(section_title_html("Job description"))
        st.text_area("Job description text", key="jd_text", height=215,
                     placeholder="Paste the full job description here, "
                                 "including the requirements section.",
                     label_visibility="collapsed")
        st.caption("Text that lists requirements under a clear heading gives "
                   "the most reliable result.")


def render_messages() -> None:
    """Show missing-input problems or a friendly error, if there are any."""
    problems = st.session_state["problems"]
    if problems:
        show(ui.error_box_html("Something is missing", " ".join(problems)))
    error = st.session_state["error"]
    if error:
        show(ui.error_box_html(*error))


def render_results(result: AnalysisResult, file_warnings: list[str]) -> None:
    summary = build_summary(result)
    warnings = ui.combine_warnings(file_warnings, result.warnings)

    summary_body = (
        f'<p style="margin:0;font-size:17px;line-height:1.65;color:{ui.INK}">'
        f"{html.escape(summary)}</p>"
    )
    show(ui.card_html("Summary", summary_body))

    score_col, gap_col = st.columns([1, 2], gap="medium")
    with score_col:
        body = (
            ui.score_ring_html(result.overall_score, result.verdict)
            + '<div style="text-align:center;margin:18px 0">'
            + ui.verdict_badge_html(result.verdict)
            + "</div>"
            + '<div style="display:flex;gap:12px">'
            + ui.stat_html("Meaning match", ui.meaning_match_text(result),
                           ui.meaning_match_detail(result))
            + ui.stat_html("Skill coverage", ui.coverage_text(result),
                           ui.coverage_detail(result))
            + "</div>"
        )
        show(ui.card_html("Overall match", body))
    with gap_col:
        show(ui.card_html("Skills to add", ui.missing_section_html(result.missing_skills),
                          "Listed by importance, most important first."))

    have_col, extra_col = st.columns(2, gap="medium")
    with have_col:
        show(ui.card_html(
            "Skills you already have",
            ui.chips_html(result.matched_skills, "matched",
                          empty_text="No skills from the job description were found."),
            "Asked for by the job description and found on your resume.",
        ))
    with extra_col:
        show(ui.card_html(
            "Extra skills",
            ui.chips_html(result.extra_skills, "extra", empty_text="None."),
            "On your resume, but not asked for in this job description.",
        ))

    show(ui.warnings_box_html(warnings))

    how_col, guide_col = st.columns(2, gap="medium")
    with how_col:
        show(note_html("How to read this", HOW_TO_READ))
    with guide_col:
        show(note_html("A guide, not a verdict", GUIDE_NOT_VERDICT))

    render_downloads(result, summary, warnings)


def render_downloads(result: AnalysisResult, summary: str, warnings: list[str]) -> None:
    """Two download buttons: the plain-text report and the PDF report."""
    generated_at = datetime.now()
    text_col, pdf_col, _ = st.columns([1, 1, 3], gap="small")
    with text_col:
        st.download_button(
            "Download report (.txt)",
            data=build_report(result, summary, warnings, generated_at),
            file_name="resume_match_report.txt",
            mime="text/plain",
            key="download_txt",
            on_click="ignore",
            use_container_width=True,
        )
    with pdf_col:
        try:
            pdf_bytes = build_pdf(result, summary, warnings, generated_at)
        except Exception:  # noqa: BLE001 - the page must not break over the PDF
            LOGGER.exception("Could not build the PDF report")
            st.caption("The PDF report could not be created. The text report still works.")
            return
        st.download_button(
            "Download report (PDF)",
            data=pdf_bytes,
            file_name="resume_match_report.pdf",
            mime="application/pdf",
            key="download_pdf",
            on_click="ignore",
            use_container_width=True,
        )


def main() -> None:
    st.set_page_config(page_title="Resume Matcher", layout="wide",
                       initial_sidebar_state="collapsed")
    show(PAGE_CSS)
    init_state()

    render_header()
    render_inputs()

    _, middle, _ = st.columns([2, 1, 2])
    with middle:
        analyze_clicked = st.button("Analyze match", key="analyze", type="primary",
                                    use_container_width=True)
    if analyze_clicked:
        run_analysis()

    render_messages()
    result = st.session_state["result"]
    if result is not None:
        render_results(result, st.session_state["file_warnings"])


main()
