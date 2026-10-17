"""
Tests for resume_matcher.matching.compare_methods.

A fake parser and a fake Sentence-BERT model are injected, so these tests need
no spaCy, no PyTorch and no download, and they never read the real
data/labels/pairs.csv. The fake model "embeds" a text as word counts over a
small vocabulary, so expected scores can be worked out by hand:
    "python pandas" vs "python sql"            -> 1 / 2          = 0.5
    "python sql docker" vs "python sql"        -> 2 / (sqrt(3) * sqrt(2)) = 0.8165
"""
import csv
import math
from types import SimpleNamespace

import numpy as np
import pytest

from resume_matcher import config
from resume_matcher.matching import compare_methods as cm
from resume_matcher.matching.compare_methods import (
    CAVEATS,
    METHODS,
    ComparisonResult,
    Correlation,
    MethodSummary,
    PairResult,
    correlate,
    default_output_path,
    format_report,
    main,
    mean_by_verdict,
    run_comparison,
    save_csv,
)
from resume_matcher.matching.pairs import EXPECTED_COLUMNS, VALID_VERDICTS, LabelledPair
from resume_matcher.matching.sbert import SbertEmbedder

VOCAB = ["python", "pandas", "java", "spring", "docker", "sql"]


class FakeModel:
    max_seq_length = 256

    def tokenizer(self, text):
        return {"input_ids": list(range(len(text.split()) + 2))}

    def encode(self, texts, **kwargs):
        return np.array(
            [[t.lower().split().count(w) for w in VOCAB] for t in texts], dtype=float
        )


class FakeParser:
    """Stands in for the Phase 2 parser; remembers every call."""

    def __init__(self, jd_section="requirements"):
        self.jd_section = jd_section
        self.calls = []

    def __call__(self, text, kind):
        self.calls.append((kind, text))
        section = "skills" if kind == "resume" else self.jd_section
        return SimpleNamespace(
            text=text, sections={section: text}, normalized_text=lambda: text.lower()
        )


def make_pair(tmp_path, pair_id, resume_text, jd_text, score, verdict, resume_name=None):
    resume_file = tmp_path / (resume_name or f"r{pair_id}.txt")
    jd_file = tmp_path / f"j{pair_id}.txt"
    resume_file.write_text(resume_text, encoding="utf-8")
    jd_file.write_text(jd_text, encoding="utf-8")
    return LabelledPair(pair_id, resume_file, jd_file, score, verdict, "")


@pytest.fixture
def pairs(tmp_path):
    return [
        make_pair(tmp_path, 1, "python pandas", "python sql", 80, "Fit", "shared_r.txt"),
        make_pair(tmp_path, 2, "java spring", "python sql", 10, "No-Fit"),
        make_pair(tmp_path, 3, "python sql docker", "python sql", 60, "Partial Fit"),
        make_pair(tmp_path, 4, "python pandas", "docker", 30, "No-Fit", "shared_r.txt"),
    ]


@pytest.fixture
def embedder():
    return SbertEmbedder(model=FakeModel(), max_tokens=20)


def run(pairs, embedder, parser=None):
    return run_comparison(pairs, embedder=embedder, parse_fn=parser or FakeParser())


# -------------------------------------------------------------- correlate
class TestCorrelate:
    def test_perfect_positive(self):
        for method in ("pearson", "spearman"):
            c = correlate([1, 2, 3, 4], [10, 20, 30, 40], method)
            assert c.r == pytest.approx(1.0)
            assert c.n == 4

    def test_perfect_negative(self):
        assert correlate([1, 2, 3], [9, 5, 1], "pearson").r == pytest.approx(-1.0)

    def test_hand_calculated_value(self):
        # x = 1,2,3,4 and y = 1,3,2,4: sum of products of deviations = 4,
        # sum of squared deviations = 5 for both, so r = 4 / 5 = 0.8.
        assert correlate([1, 2, 3, 4], [1, 3, 2, 4], "pearson").r == pytest.approx(0.8)
        assert correlate([1, 2, 3, 4], [1, 3, 2, 4], "spearman").r == pytest.approx(0.8)

    def test_spearman_only_needs_the_same_order(self):
        x, y = [1, 2, 3, 4, 5], [1, 4, 9, 16, 25]
        assert correlate(x, y, "spearman").r == pytest.approx(1.0)
        assert correlate(x, y, "pearson").r < 1.0

    def test_p_value_is_a_probability(self):
        c = correlate([1, 2, 3, 4, 5], [2, 1, 4, 3, 5])
        assert 0.0 <= c.p <= 1.0

    def test_default_method_is_pearson(self):
        x, y = [1, 2, 3, 4, 5], [1, 4, 9, 16, 25]
        assert correlate(x, y).r == pytest.approx(correlate(x, y, "pearson").r)

    @pytest.mark.parametrize("n", [0, 1, 2])
    def test_fewer_than_three_pairs_is_undefined(self, n):
        c = correlate(list(range(n)), list(range(n)))
        assert c == Correlation(None, None, n)

    def test_constant_scores_are_undefined(self):
        assert correlate([1, 2, 3], [5, 5, 5]).r is None

    def test_constant_human_scores_are_undefined(self):
        assert correlate([4, 4, 4], [1, 2, 3], "spearman").r is None

    def test_length_mismatch_is_rejected(self):
        with pytest.raises(ValueError):
            correlate([1, 2, 3], [1, 2])

    def test_unknown_method_is_rejected(self):
        with pytest.raises(ValueError):
            correlate([1, 2, 3], [1, 2, 3], "kendall")

    def test_r_and_p_are_plain_floats(self):
        c = correlate([1, 2, 3, 4], [1, 3, 2, 4])
        assert type(c.r) is float and type(c.p) is float


# ---------------------------------------------------------- mean_by_verdict
class TestMeanByVerdict:
    def test_averages_per_verdict(self):
        result = mean_by_verdict(
            ["Fit", "Fit", "No-Fit", "Partial Fit"], [0.8, 0.6, 0.1, 0.4]
        )
        assert result["Fit"] == pytest.approx(0.7)
        assert result["Partial Fit"] == pytest.approx(0.4)
        assert result["No-Fit"] == pytest.approx(0.1)

    def test_a_verdict_without_pairs_is_none(self):
        result = mean_by_verdict(["Fit"], [0.9])
        assert result["Partial Fit"] is None and result["No-Fit"] is None

    def test_keys_follow_the_verdict_order(self):
        assert list(mean_by_verdict([], [])) == list(VALID_VERDICTS)


# ----------------------------------------------------------- run_comparison
class TestRunComparison:
    def test_one_result_per_pair_in_order(self, pairs, embedder):
        result = run(pairs, embedder)
        assert [r.pair_id for r in result.pair_results] == [1, 2, 3, 4]
        assert [r.human_score for r in result.pair_results] == [80, 10, 60, 30]
        assert [r.human_verdict for r in result.pair_results] == [
            "Fit", "No-Fit", "Partial Fit", "No-Fit",
        ]

    def test_sbert_scores_match_hand_calculations(self, pairs, embedder):
        first, second, third, _ = run(pairs, embedder).pair_results
        assert first.sbert == pytest.approx(0.5)
        assert second.sbert == pytest.approx(0.0)
        assert third.sbert == pytest.approx(2 / (math.sqrt(3) * math.sqrt(2)))

    def test_section_scores_match_hand_calculations(self, pairs, embedder):
        first, second, third, _ = run(pairs, embedder).pair_results
        assert first.sections == pytest.approx(0.5)
        assert second.sections == pytest.approx(0.0)
        assert third.sections == pytest.approx(2 / (math.sqrt(3) * math.sqrt(2)))

    def test_tfidf_scores_are_between_zero_and_one(self, pairs, embedder):
        for r in run(pairs, embedder).pair_results:
            assert 0.0 <= r.tfidf <= 1.0

    def test_a_pair_with_no_shared_words_scores_zero_everywhere(self, pairs, embedder):
        second = run(pairs, embedder).pair_results[1]
        assert second.tfidf == 0.0 and second.sbert == 0.0 and second.sections == 0.0

    def test_every_distinct_text_is_parsed_once(self, pairs, embedder):
        parser = FakeParser()
        result = run(pairs, embedder, parser)
        distinct_files = {p.resume_file for p in pairs} | {p.jd_file for p in pairs}
        assert len(parser.calls) == len(distinct_files) == 7
        assert result.corpus_size == 7

    def test_texts_are_parsed_with_the_right_kind(self, pairs, embedder):
        parser = FakeParser()
        run(pairs, embedder, parser)
        kinds = {kind for kind, _ in parser.calls}
        assert kinds == {"resume", "jd"}

    def test_tfidf_vocabulary_is_reported(self, pairs, embedder):
        result = run(pairs, embedder)
        # python pandas java spring sql docker
        assert result.tfidf_vocabulary_size == 6

    def test_timings_are_not_negative(self, pairs, embedder):
        result = run(pairs, embedder)
        assert result.tfidf_fit_seconds >= 0.0
        assert result.sbert_load_seconds >= 0.0
        assert all(s.seconds_per_pair >= 0.0 for s in result.summaries)

    def test_three_summaries_in_method_order(self, pairs, embedder):
        result = run(pairs, embedder)
        assert [s.key for s in result.summaries] == [key for key, _ in METHODS]
        assert [s.name for s in result.summaries] == [name for _, name in METHODS]

    def test_summaries_hold_correlations_and_verdict_means(self, pairs, embedder):
        sbert = run(pairs, embedder).summaries[1]
        assert isinstance(sbert, MethodSummary)
        assert sbert.pearson.n == 4 and sbert.spearman.n == 4
        assert sbert.pearson.r is not None
        assert set(sbert.mean_by_verdict) == set(VALID_VERDICTS)
        assert sbert.mean_by_verdict["No-Fit"] == pytest.approx((0.0 + sbert_of(pairs, embedder, 4)) / 2)

    def test_a_pair_without_requirement_lines_has_no_section_score(self, pairs, embedder):
        parser = FakeParser(jd_section="about")  # no "requirements" section
        result = run(pairs, embedder, parser)
        assert all(r.sections is None for r in result.pair_results)
        assert result.summaries[2].pearson == Correlation(None, None, 0)

    def test_pairs_without_a_section_score_are_left_out_of_that_correlation(
        self, pairs, embedder
    ):
        class OneJdWithoutRequirements(FakeParser):
            def __call__(self, text, kind):
                parsed = super().__call__(text, kind)
                if kind == "jd" and text == "docker":
                    parsed.sections = {"about": text}
                return parsed

        result = run(pairs, embedder, OneJdWithoutRequirements())
        assert result.pair_results[3].sections is None
        assert result.summaries[2].pearson.n == 3  # pair 4 left out
        assert result.summaries[1].pearson.n == 4  # other methods use all four

    def test_no_pairs_is_rejected(self, embedder):
        with pytest.raises(ValueError, match="no pairs"):
            run([], embedder)

    def test_an_unrated_pair_is_rejected_and_named(self, tmp_path, embedder):
        unrated = make_pair(tmp_path, 7, "python", "python", None, None)
        with pytest.raises(ValueError, match=r"\[7\]"):
            run([unrated], embedder)

    def test_results_are_immutable(self, pairs, embedder):
        result = run(pairs, embedder)
        assert isinstance(result, ComparisonResult)
        assert isinstance(result.pair_results[0], PairResult)
        with pytest.raises(Exception):
            result.pair_results[0].tfidf = 0.0


def sbert_of(pairs, embedder, pair_id):
    """The whole-document SBERT score of one pair, computed directly."""
    pair = next(p for p in pairs if p.pair_id == pair_id)
    return embedder.similarity(pair.read_resume(), pair.read_jd())


# -------------------------------------------------------------- reporting
@pytest.fixture
def result(pairs, embedder):
    return run(pairs, embedder)


class TestFormatReport:
    def test_has_all_the_parts(self, result):
        text = format_report(result)
        assert "Compared 4 rated pairs." in text
        assert "TF-IDF corpus: 7 texts" in text
        assert "SBERT model load" in text
        assert "Agreement with the human score" in text
        assert "Pearson r" in text and "Spearman rho" in text
        assert "Average method score per human verdict" in text
        assert "Per pair" in text
        assert "Caveats" in text

    def test_every_method_and_verdict_is_named(self, result):
        text = format_report(result)
        for _, name in METHODS:
            assert name in text
        for verdict in VALID_VERDICTS:
            assert verdict in text

    def test_every_pair_has_a_row(self, result):
        lines = format_report(result).splitlines()
        start = lines.index("Per pair") + 2
        rows = lines[start : start + 4]
        assert [row.split()[0] for row in rows] == ["1", "2", "3", "4"]

    def test_caveats_are_printed(self, result):
        text = format_report(result)
        for caveat in CAVEATS:
            assert caveat in text

    def test_caveats_say_the_scores_are_not_probabilities(self):
        assert any("not probabilities" in c for c in CAVEATS)

    def test_undefined_values_print_as_na(self):
        undefined = ComparisonResult(
            pair_results=(PairResult(1, 50.0, "Fit", 0.2, 0.3, None),),
            summaries=tuple(
                MethodSummary(key, name, Correlation(None, None, 1),
                              Correlation(None, None, 1), 0.001,
                              {v: None for v in VALID_VERDICTS})
                for key, name in METHODS
            ),
            corpus_size=2, tfidf_vocabulary_size=5,
            tfidf_fit_seconds=0.01, sbert_load_seconds=1.0,
        )
        text = format_report(undefined)
        assert "n/a" in text
        assert "Compared 1 rated pairs." in text


# ----------------------------------------------------------------- save_csv
class TestSaveCsv:
    def read(self, path):
        with open(path, encoding="utf-8", newline="") as handle:
            return list(csv.reader(handle))

    def test_header_and_rows(self, result, tmp_path):
        rows = self.read(save_csv(result, tmp_path / "out.csv"))
        assert rows[0] == [
            "pair_id", "human_score", "human_verdict", "tfidf", "sbert", "sbert_sections",
        ]
        assert len(rows) == 1 + 4
        assert rows[1][:3] == ["1", "80", "Fit"]
        assert float(rows[1][4]) == pytest.approx(0.5)

    def test_missing_section_score_is_an_empty_cell(self, tmp_path):
        only = ComparisonResult(
            pair_results=(PairResult(1, 50.0, "Fit", 0.2, 0.3, None),),
            summaries=(), corpus_size=2, tfidf_vocabulary_size=5,
            tfidf_fit_seconds=0.0, sbert_load_seconds=0.0,
        )
        rows = self.read(save_csv(only, tmp_path / "out.csv"))
        assert rows[1][5] == ""

    def test_missing_folders_are_created(self, result, tmp_path):
        target = tmp_path / "a" / "b" / "out.csv"
        assert save_csv(result, target) == target
        assert target.is_file()

    def test_default_output_path(self):
        assert default_output_path() == config.PROCESSED_DIR / "comparison_results.csv"


# --------------------------------------------------------------------- main
def write_pairs_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle, lineterminator="\n").writerows([list(EXPECTED_COLUMNS)] + rows)


class TestMain:
    def test_missing_pairs_file_gives_exit_code_1(self, tmp_path, capsys):
        assert main(["--pairs", str(tmp_path / "missing.csv")]) == 1
        assert "File not found" in capsys.readouterr().out

    def test_unrated_pairs_stop_the_run_with_a_clear_message(self, tmp_path, capsys):
        (tmp_path / "r.txt").write_text("python", encoding="utf-8")
        (tmp_path / "j.txt").write_text("python", encoding="utf-8")
        csv_path = tmp_path / "pairs.csv"
        write_pairs_csv(csv_path, [[1, str(tmp_path / "r.txt"), str(tmp_path / "j.txt"), "", "", ""]])
        assert main(["--pairs", str(csv_path)]) == 1
        out = capsys.readouterr().out
        assert "Cannot compare yet" in out
        assert "not rated yet" in out
        assert "resume_matcher.matching.pairs" in out

    @pytest.fixture
    def rated_csv(self, tmp_path):
        (tmp_path / "r.txt").write_text("python", encoding="utf-8")
        (tmp_path / "j.txt").write_text("python", encoding="utf-8")
        csv_path = tmp_path / "pairs.csv"
        write_pairs_csv(
            csv_path,
            [[1, str(tmp_path / "r.txt"), str(tmp_path / "j.txt"), "70", "Fit", ""]],
        )
        return csv_path

    @pytest.fixture
    def stub_run(self, monkeypatch, result):
        monkeypatch.setattr(cm, "run_comparison", lambda pairs: result)

    def test_success_prints_the_report_and_saves_the_csv(
        self, rated_csv, stub_run, tmp_path, capsys
    ):
        output = tmp_path / "saved" / "out.csv"
        assert main(["--pairs", str(rated_csv), "--output", str(output)]) == 0
        out = capsys.readouterr().out
        assert "Compared 4 rated pairs." in out
        assert "Saved per-pair scores to" in out
        assert output.is_file()

    def test_no_save_writes_nothing(self, rated_csv, stub_run, tmp_path, capsys):
        output = tmp_path / "should_not_exist.csv"
        assert main(["--pairs", str(rated_csv), "--output", str(output), "--no-save"]) == 0
        assert not output.exists()
        assert "Saved per-pair scores" not in capsys.readouterr().out
