"""
Tests for resume_matcher.analyzer.gaps (Phase 5).

Small hand-made inputs only. The expected order is worked out by hand in the
comments, following the ranking rules in gaps.py:
  1. must-have, then nice-to-have, then unknown
  2. more distinct JD sections first
  3. alphabetical, ignoring case
"""
import itertools

from resume_matcher.analyzer.gaps import rank_missing_skills
from resume_matcher.schemas import SkillGap


def names(gaps):
    return [g.skill for g in gaps]


class TestRankMissingSkills:
    def test_importance_order(self):
        src = {
            "Docker": ("nice_to_have",),
            "SQL": ("requirements",),
            "Git": ("header",),
        }
        # must-have (SQL) -> nice-to-have (Docker) -> unknown (Git)
        gaps = rank_missing_skills(["Git", "Docker", "SQL"], src)
        assert names(gaps) == ["SQL", "Docker", "Git"]
        assert [g.importance for g in gaps] == ["must-have", "nice-to-have", "unknown"]

    def test_more_sections_first_within_same_importance(self):
        src = {
            "Airflow": ("requirements",),
            "Python": ("requirements", "responsibilities"),
            "SQL": ("requirements",),
        }
        # Python is in 2 sections -> first; Airflow, SQL tie -> alphabetical.
        gaps = rank_missing_skills(["SQL", "Airflow", "Python"], src)
        assert names(gaps) == ["Python", "Airflow", "SQL"]

    def test_alphabetical_ignores_case(self):
        src = {"Docker": ("nice_to_have",), "aws": ("nice_to_have",)}
        # "aws" < "docker" when case is ignored (plain sort would put "Docker" first).
        assert names(rank_missing_skills(["Docker", "aws"], src)) == ["aws", "Docker"]

    def test_skill_without_sources_is_unknown(self):
        gaps = rank_missing_skills(["Spark"], {})
        assert gaps == [SkillGap(skill="Spark", importance="unknown")]

    def test_empty_input_gives_empty_list(self):
        assert rank_missing_skills([], {}) == []

    def test_duplicates_are_listed_once(self):
        src = {"SQL": ("requirements",)}
        assert names(rank_missing_skills(["SQL", "SQL", "SQL"], src)) == ["SQL"]

    def test_returns_skillgap_objects(self):
        gaps = rank_missing_skills(["SQL"], {"SQL": ("requirements",)})
        assert len(gaps) == 1
        assert isinstance(gaps[0], SkillGap)
        assert gaps[0].skill == "SQL"
        assert gaps[0].importance == "must-have"

    def test_accepts_tuple_and_generator(self):
        src = {"A": ("requirements",), "B": ("nice_to_have",)}
        assert names(rank_missing_skills(("B", "A"), src)) == ["A", "B"]
        assert names(rank_missing_skills((s for s in ("B", "A")), src)) == ["A", "B"]

    def test_order_does_not_depend_on_input_order(self):
        src = {
            "Python": ("requirements", "responsibilities"),
            "SQL": ("requirements",),
            "Docker": ("nice_to_have",),
            "Git": ("header",),
        }
        expected = ["Python", "SQL", "Docker", "Git"]
        for perm in itertools.permutations(["Python", "SQL", "Docker", "Git"]):
            assert names(rank_missing_skills(perm, src)) == expected

    def test_inputs_are_not_modified(self):
        missing = ["B", "A"]
        src = {"A": ("requirements",), "B": ("nice_to_have",)}
        rank_missing_skills(missing, src)
        assert missing == ["B", "A"]
        assert src == {"A": ("requirements",), "B": ("nice_to_have",)}

    def test_skill_in_both_requirements_and_nice_to_have_is_must_have(self):
        src = {"Docker": ("nice_to_have", "requirements")}
        gaps = rank_missing_skills(["Docker"], src)
        assert gaps[0].importance == "must-have"

    def test_full_example(self):
        src = {
            "Docker": ("nice_to_have",),
            "SQL": ("requirements",),
            "Python": ("requirements", "responsibilities"),
            "Git": ("header",),
            "aws": ("nice_to_have",),
            "Airflow": ("requirements",),
        }
        missing = ["Docker", "SQL", "Python", "Git", "aws", "Airflow", "SQL", "Spark"]
        # must-have: Python (2 sections), then Airflow, SQL
        # nice-to-have: aws, Docker
        # unknown: Git (header), Spark (no sources) -> alphabetical
        assert names(rank_missing_skills(missing, src)) == [
            "Python", "Airflow", "SQL", "aws", "Docker", "Git", "Spark",
        ]
