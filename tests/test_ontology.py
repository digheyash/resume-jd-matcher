"""Tests for resume_matcher.skills.ontology (Phase 3)."""

import json

import pytest

from resume_matcher.skills.ontology import (
    Ontology,
    OntologyError,
    get_default_ontology,
    load_ontology,
    parse_ontology,
)


def make_data(**overrides):
    """A tiny valid skills.json structure for invented-data tests."""
    data = {
        "context_hints": {"words": ["Programming"], "sections": ["Skills"]},
        "skills": [
            {
                "canonical": "JavaScript",
                "category": "programming_language",
                "aliases": ["JS", "ECMAScript"],
                "ambiguous_aliases": ["JS"],
            },
            {"canonical": "Python", "category": "programming_language"},
        ],
    }
    data.update(overrides)
    return data


# ---------------------------------------------------------------- real file


def test_default_ontology_loads_from_project_file():
    onto = get_default_ontology()
    assert isinstance(onto, Ontology)
    assert len(onto) > 0


def test_default_ontology_is_cached():
    assert get_default_ontology() is get_default_ontology()


@pytest.mark.parametrize(
    "spelling, expected",
    [
        ("sklearn", "Scikit-learn"),
        ("Scikit learn", "Scikit-learn"),
        ("JS", "JavaScript"),
        ("golang", "Go"),
        ("NLP", "Natural Language Processing"),
        ("k8s", "Kubernetes"),
        ("C#", "C#"),
        ("C++", "C++"),
        (".NET", ".NET"),
    ],
)
def test_real_ontology_resolves_aliases(spelling, expected):
    assert get_default_ontology().resolve(spelling) == expected


def test_real_ontology_tricky_names_exist():
    onto = get_default_ontology()
    for name in ["C++", "C#", ".NET", "Node.js", "Scikit-learn", "R", "Go"]:
        assert name in onto


def test_real_ontology_ambiguous_flags():
    flags = {s.text: s.ambiguous for s in get_default_ontology().spellings()}
    for risky in ["R", "Go", "C", "Rust", "Swift", "Spark"]:
        assert flags[risky] is True
    for safe in ["Python", "C++", "Scikit-learn", "Docker"]:
        assert flags[safe] is False


def test_real_ontology_spellings_are_longest_first():
    lengths = [len(s.text) for s in get_default_ontology().spellings()]
    assert lengths == sorted(lengths, reverse=True)


def test_real_ontology_has_context_hints():
    onto = get_default_ontology()
    assert "skills" in onto.context_sections
    assert "requirements" in onto.context_sections
    assert "programming" in onto.context_words


# ------------------------------------------------------------ small examples


def test_resolve_is_case_and_space_insensitive():
    onto = parse_ontology(make_data())
    assert onto.resolve("  js ") == "JavaScript"
    assert onto.resolve("ECMASCRIPT") == "JavaScript"
    assert onto.resolve("python") == "Python"
    assert onto.resolve("cobol") is None


def test_missing_optional_fields_default_to_empty():
    onto = parse_ontology(make_data())
    python = onto.get("Python")
    assert python.aliases == ()
    assert python.ambiguous == frozenset()


def test_category_of():
    onto = parse_ontology(make_data())
    assert onto.category_of("Python") == "programming_language"
    assert onto.category_of("Unknown") is None


def test_ambiguous_spelling_is_flagged():
    onto = parse_ontology(make_data())
    flags = {s.text: s.ambiguous for s in onto.spellings()}
    assert flags["JS"] is True
    assert flags["ECMAScript"] is False
    assert flags["JavaScript"] is False


def test_canonical_names_keep_file_order():
    onto = parse_ontology(make_data())
    assert onto.canonical_names() == ["JavaScript", "Python"]


def test_context_hints_are_lowercased():
    onto = parse_ontology(make_data())
    assert onto.context_words == frozenset({"programming"})
    assert onto.context_sections == frozenset({"skills"})


def test_missing_context_hints_are_allowed():
    data = make_data()
    del data["context_hints"]
    onto = parse_ontology(data)
    assert onto.context_words == frozenset()


def test_spellings_have_no_case_duplicates():
    data = make_data()
    data["skills"][1]["aliases"] = ["python", "PYTHON"]
    onto = parse_ontology(data)
    texts = [s.text.lower() for s in onto.spellings() if s.canonical == "Python"]
    assert texts == ["python"]


# ------------------------------------------------------------- bad input


def test_duplicate_canonical_is_rejected():
    data = make_data()
    data["skills"].append({"canonical": "Python", "category": "x"})
    with pytest.raises(OntologyError, match="duplicate"):
        parse_ontology(data)


def test_spelling_claimed_by_two_skills_is_rejected():
    data = make_data()
    data["skills"][1]["aliases"] = ["js"]
    with pytest.raises(OntologyError, match="claimed by both"):
        parse_ontology(data)


def test_ambiguous_alias_must_be_a_real_spelling():
    data = make_data()
    data["skills"][1]["ambiguous_aliases"] = ["Nope"]
    with pytest.raises(OntologyError, match="ambiguous_aliases"):
        parse_ontology(data)


@pytest.mark.parametrize(
    "bad",
    [
        [],
        {},
        {"skills": []},
        {"skills": "Python"},
        {"skills": ["Python"]},
        {"skills": [{"category": "x"}]},
        {"skills": [{"canonical": "A"}]},
        {"skills": [{"canonical": "A", "category": "x", "aliases": "B"}]},
    ],
)
def test_malformed_structures_are_rejected(bad):
    with pytest.raises(OntologyError):
        parse_ontology(bad)


# ------------------------------------------------------------------ files


def test_load_from_file(tmp_path):
    path = tmp_path / "skills.json"
    path.write_text(json.dumps(make_data()), encoding="utf-8")
    assert len(load_ontology(path)) == 2


def test_missing_file_raises(tmp_path):
    with pytest.raises(OntologyError, match="not found"):
        load_ontology(tmp_path / "nope.json")


def test_invalid_json_raises(tmp_path):
    path = tmp_path / "skills.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(OntologyError, match="not valid JSON"):
        load_ontology(path)
