"""
Data contract shared by every phase.

Defining the OUTPUT shape now means the parser, extractor, matcher and UI can
be built separately and still plug together at the end.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class SkillGap:
    """One missing skill, with how important it is for this job."""
    skill: str
    importance: str = "unknown"      # "must-have" | "nice-to-have" | "unknown"


@dataclass
class AnalysisResult:
    """Everything the user sees after one resume-vs-JD analysis."""
    overall_score: float                      # 0-100, final calibrated score
    verdict: str                              # "Fit" | "Partial Fit" | "No-Fit"
    semantic_score: float                     # 0-1, cosine similarity
    skill_coverage: float                     # 0-1, matched / required
    matched_skills: List[str] = field(default_factory=list)
    missing_skills: List[SkillGap] = field(default_factory=list)
    extra_skills: List[str] = field(default_factory=list)   # in resume, not asked by JD
    warnings: List[str] = field(default_factory=list)       # e.g. "resume text very short"