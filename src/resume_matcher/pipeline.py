"""
Single entry point of the system: analyze(resume_text, jd_text).

Right now it is only a stub. Each phase fills in one stage:
    Phase 2 -> parse + clean      Phase 3 -> skills + set logic
    Phase 4 -> similarity         Phase 5 -> score + verdict
The UI (Phase 6) will call ONLY this function.
"""
from .schemas import AnalysisResult


def analyze(resume_text: str, jd_text: str) -> AnalysisResult:
    """Compare one resume against one job description."""
    raise NotImplementedError("analyze() is built across Phases 2-5.")