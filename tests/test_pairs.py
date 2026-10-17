"""
Tests for resume_matcher.matching.pairs.

Every test builds its own small CSV in a temporary folder. The real
data/labels/pairs.csv is never read, so these tests pass whatever state your
own ratings are in.
"""
import csv

import pytest

from resume_matcher import config
from resume_matcher.matching.pairs import (
    EXPECTED_COLUMNS,
    VALID_VERDICTS,
    LabelledPair,
    PairsError,
    check_consistency,
    default_pairs_path,
    load_pairs,
    main,
)

HEADER = list(EXPECTED_COLUMNS)


def write_csv(path, rows, header=HEADER, encoding="utf-8"):
    with path.open("w", encoding=encoding, newline="") as handle:
        csv.writer(handle, lineterminator="\n").writerows([header] + rows)


def make_row(pair_id=1, score="70", verdict="Fit", notes="", resume="r1.txt", jd="j1.txt"):
    return [pair_id, resume, jd, score, verdict, notes]


@pytest.fixture
def workspace(tmp_path):
    """A folder holding one resume and one JD text, plus a pairs.csv path."""
    (tmp_path / "r1.txt").write_text("resume text", encoding="utf-8")
    (tmp_path / "j1.txt").write_text("jd text", encoding="utf-8")
    return tmp_path


def load(workspace, rows, **kwargs):
    csv_path = workspace / "pairs.csv"
    write_csv(csv_path, rows)
    return load_pairs(csv_path, root=workspace, **kwargs)


def problems_of(workspace, rows, **kwargs):
    with pytest.raises(PairsError) as info:
        load(workspace, rows, **kwargs)
    return info.value.problems


# ---------------------------------------------------------------- loading
class TestLoadValid:
    def test_a_valid_row_is_loaded_with_the_right_types(self, workspace):
        (pair,) = load(workspace, [make_row(3, "72.5", "Partial Fit", "close call")])
        assert isinstance(pair, LabelledPair)
        assert pair.pair_id == 3
        assert pair.human_score == 72.5
        assert pair.human_verdict == "Partial Fit"
        assert pair.notes == "close call"
        assert pair.is_rated is True

    def test_file_paths_are_resolved_against_the_root(self, workspace):
        (pair,) = load(workspace, [make_row()])
        assert pair.resume_file == workspace / "r1.txt"
        assert pair.jd_file == workspace / "j1.txt"

    def test_texts_can_be_read(self, workspace):
        (pair,) = load(workspace, [make_row()])
        assert pair.read_resume() == "resume text"
        assert pair.read_jd() == "jd text"

    def test_absolute_paths_are_accepted(self, workspace):
        row = make_row(resume=str(workspace / "r1.txt"), jd=str(workspace / "j1.txt"))
        (pair,) = load(workspace, [row])
        assert pair.resume_file == workspace / "r1.txt"

    def test_rows_keep_their_order(self, workspace):
        rows = [make_row(5), make_row(2), make_row(9)]
        assert [p.pair_id for p in load(workspace, rows)] == [5, 2, 9]

    @pytest.mark.parametrize("raw", ["fit", "FIT", "  Fit  "])
    def test_verdict_ignores_case_and_spaces(self, workspace, raw):
        (pair,) = load(workspace, [make_row(verdict=raw)])
        assert pair.human_verdict == "Fit"

    def test_all_three_verdicts_are_accepted(self, workspace):
        rows = [make_row(i + 1, "50", v) for i, v in enumerate(VALID_VERDICTS)]
        assert [p.human_verdict for p in load(workspace, rows)] == list(VALID_VERDICTS)

    @pytest.mark.parametrize("raw, expected", [("0", 0.0), ("100", 100.0), (" 45 ", 45.0)])
    def test_score_boundaries_and_spaces(self, workspace, raw, expected):
        (pair,) = load(workspace, [make_row(score=raw)])
        assert pair.human_score == expected

    def test_note_with_a_comma_in_quotes_is_kept(self, workspace):
        (pair,) = load(workspace, [make_row(notes="skills fit, but no experience")])
        assert pair.notes == "skills fit, but no experience"

    def test_a_row_without_the_notes_cell_is_fine(self, workspace):
        csv_path = workspace / "pairs.csv"
        csv_path.write_text(
            ",".join(HEADER) + "\n1,r1.txt,j1.txt,70,Fit\n", encoding="utf-8"
        )
        (pair,) = load_pairs(csv_path, root=workspace)
        assert pair.notes == ""

    def test_a_trailing_empty_comma_is_tolerated(self, workspace):
        row = make_row() + [""]
        assert len(load(workspace, [row])) == 1

    def test_file_saved_with_a_byte_order_mark_loads(self, workspace):
        csv_path = workspace / "pairs.csv"
        write_csv(csv_path, [make_row()], encoding="utf-8-sig")
        assert len(load_pairs(csv_path, root=workspace)) == 1

    def test_pairs_are_immutable(self, workspace):
        (pair,) = load(workspace, [make_row()])
        with pytest.raises(Exception):
            pair.human_score = 1.0


# ---------------------------------------------------------- rating problems
class TestRatingProblems:
    @pytest.mark.parametrize("bad", ["abc", "-1", "101", "nan", "inf", "7,5"])
    def test_bad_scores_are_rejected(self, workspace, bad):
        problems = problems_of(workspace, [make_row(score=bad)])
        assert len(problems) == 1
        assert "score" in problems[0]

    @pytest.mark.parametrize("bad", ["no fit", "maybe", "Partial", "Fit!"])
    def test_bad_verdicts_are_rejected_and_the_choices_are_listed(self, workspace, bad):
        problems = problems_of(workspace, [make_row(verdict=bad)])
        assert len(problems) == 1
        assert "verdict" in problems[0]
        assert all(v in problems[0] for v in VALID_VERDICTS)

    def test_score_without_verdict_is_rejected(self, workspace):
        problems = problems_of(workspace, [make_row(score="70", verdict="")])
        assert any("no verdict" in p for p in problems)

    def test_verdict_without_score_is_rejected(self, workspace):
        problems = problems_of(workspace, [make_row(score="", verdict="Fit")])
        assert any("no score" in p for p in problems)

    def test_half_filled_row_is_rejected_even_when_ratings_are_optional(self, workspace):
        problems = problems_of(
            workspace, [make_row(score="70", verdict="")], require_ratings=False
        )
        assert any("no verdict" in p for p in problems)

    def test_unrated_row_is_rejected_by_default(self, workspace):
        problems = problems_of(workspace, [make_row(score="", verdict="")])
        assert any("not rated yet" in p for p in problems)

    def test_unrated_rows_load_when_ratings_are_optional(self, workspace):
        rows = [make_row(1, "", ""), make_row(2, "60", "Partial Fit")]
        pairs = load(workspace, rows, require_ratings=False)
        assert [p.is_rated for p in pairs] == [False, True]
        assert pairs[0].human_score is None
        assert pairs[0].human_verdict is None


# ------------------------------------------------------------ other problems
class TestOtherProblems:
    def test_missing_resume_file(self, workspace):
        problems = problems_of(workspace, [make_row(resume="nope.txt")])
        assert any("resume_file not found: nope.txt" in p for p in problems)

    def test_missing_jd_file(self, workspace):
        problems = problems_of(workspace, [make_row(jd="nope.txt")])
        assert any("jd_file not found" in p for p in problems)

    def test_empty_file_column(self, workspace):
        problems = problems_of(workspace, [make_row(resume="")])
        assert any("resume_file is empty" in p for p in problems)

    def test_pair_id_must_be_a_whole_number(self, workspace):
        for bad in ("x", "", "1.5"):
            problems = problems_of(workspace, [make_row(pair_id=bad)])
            assert any("pair_id" in p for p in problems), bad

    def test_duplicate_pair_id_names_both_rows(self, workspace):
        problems = problems_of(workspace, [make_row(4), make_row(4)])
        assert any("pair_id 4 is already used in row 2" in p for p in problems)

    def test_extra_content_beyond_six_columns_is_rejected(self, workspace):
        problems = problems_of(workspace, [make_row() + ["oops"]])
        assert any("too many columns" in p for p in problems)

    def test_first_data_row_is_row_two(self, workspace):
        problems = problems_of(workspace, [make_row(), make_row(2, score="abc")])
        assert problems[0].startswith("row 3:")

    def test_all_problems_are_reported_together(self, workspace):
        rows = [
            make_row(1, score="abc"),
            make_row(2, verdict="maybe"),
            make_row(3, resume="nope.txt"),
        ]
        assert len(problems_of(workspace, rows)) == 3

    def test_error_message_has_one_line_per_problem(self, workspace):
        csv_path = workspace / "pairs.csv"
        write_csv(csv_path, [make_row(1, score="abc"), make_row(2, verdict="maybe")])
        with pytest.raises(PairsError) as info:
            load_pairs(csv_path, root=workspace)
        assert str(info.value).count("\n") == 1


# ---------------------------------------------------------- file and header
class TestFileAndHeader:
    def test_wrong_header_is_rejected(self, workspace):
        csv_path = workspace / "pairs.csv"
        write_csv(csv_path, [], header=["id", "resume", "jd"])
        with pytest.raises(PairsError, match="header"):
            load_pairs(csv_path, root=workspace)

    def test_empty_file_is_rejected(self, workspace):
        csv_path = workspace / "pairs.csv"
        csv_path.write_text("", encoding="utf-8")
        with pytest.raises(PairsError, match="header"):
            load_pairs(csv_path, root=workspace)

    def test_header_only_file_gives_no_pairs(self, workspace):
        assert load(workspace, []) == []

    def test_missing_csv_raises_file_not_found(self, workspace):
        with pytest.raises(FileNotFoundError):
            load_pairs(workspace / "missing.csv", root=workspace)

    def test_default_path_is_data_labels_pairs_csv(self):
        assert default_pairs_path() == config.LABELS_DIR / "pairs.csv"


# ------------------------------------------------------------- consistency
def pair(pair_id, score, verdict):
    return LabelledPair(pair_id, None, None, score, verdict, "")


class TestConsistency:
    def test_consistent_ratings_give_no_warnings(self):
        pairs = [pair(1, 90, "Fit"), pair(2, 55, "Partial Fit"), pair(3, 10, "No-Fit")]
        assert check_consistency(pairs) == []

    def test_empty_list_gives_no_warnings(self):
        assert check_consistency([]) == []

    def test_an_inversion_is_reported_with_both_pair_numbers(self):
        warnings = check_consistency([pair(1, 40, "Fit"), pair(2, 55, "No-Fit")])
        assert len(warnings) == 1
        assert "pair 1" in warnings[0] and "pair 2" in warnings[0]

    def test_inversion_between_fit_and_partial_fit(self):
        warnings = check_consistency([pair(1, 40, "Fit"), pair(2, 60, "Partial Fit")])
        assert len(warnings) == 1

    def test_equal_scores_are_not_an_inversion(self):
        assert check_consistency([pair(1, 55, "Fit"), pair(2, 55, "No-Fit")]) == []

    def test_same_verdict_with_different_scores_is_fine(self):
        assert check_consistency([pair(1, 95, "Fit"), pair(2, 60, "Fit")]) == []

    def test_unrated_pairs_are_ignored(self):
        pairs = [pair(1, None, None), pair(2, 55, "No-Fit")]
        assert check_consistency(pairs) == []

    def test_every_inversion_is_listed(self):
        pairs = [pair(1, 30, "Fit"), pair(2, 50, "No-Fit"), pair(3, 60, "No-Fit")]
        assert len(check_consistency(pairs)) == 2


# ------------------------------------------------------------ status command
class TestMain:
    def absolute_rows(self, workspace, *ratings):
        rows = []
        for number, (score, verdict) in enumerate(ratings, start=1):
            rows.append(
                make_row(
                    number, score, verdict,
                    resume=str(workspace / "r1.txt"), jd=str(workspace / "j1.txt"),
                )
            )
        return rows

    def run(self, workspace, rows, capsys):
        csv_path = workspace / "pairs.csv"
        write_csv(csv_path, rows)
        code = main([str(csv_path)])
        return code, capsys.readouterr().out

    def test_progress_is_reported_for_a_half_rated_file(self, workspace, capsys):
        rows = self.absolute_rows(workspace, ("80", "Fit"), ("", ""), ("20", "No-Fit"))
        code, out = self.run(workspace, rows, capsys)
        assert code == 0
        assert "3 pairs, 2 rated, 1 still to rate" in out
        assert "Fit: 1" in out and "No-Fit: 1" in out
        assert "scores: min 20, max 80" in out

    def test_nothing_rated_yet(self, workspace, capsys):
        rows = self.absolute_rows(workspace, ("", ""), ("", ""))
        code, out = self.run(workspace, rows, capsys)
        assert code == 0
        assert "2 pairs, 0 rated, 2 still to rate" in out

    def test_clean_file_says_no_warnings(self, workspace, capsys):
        rows = self.absolute_rows(workspace, ("80", "Fit"), ("20", "No-Fit"))
        _, out = self.run(workspace, rows, capsys)
        assert "no consistency warnings" in out

    def test_inversion_is_printed(self, workspace, capsys):
        rows = self.absolute_rows(workspace, ("30", "Fit"), ("60", "No-Fit"))
        code, out = self.run(workspace, rows, capsys)
        assert code == 0
        assert "1 consistency warning(s)" in out

    def test_problems_give_exit_code_1(self, workspace, capsys):
        rows = self.absolute_rows(workspace, ("abc", "Fit"))
        code, out = self.run(workspace, rows, capsys)
        assert code == 1
        assert "1 problem(s)" in out
        assert "not a number" in out

    def test_missing_file_gives_exit_code_1(self, workspace, capsys):
        code = main([str(workspace / "missing.csv")])
        assert code == 1
        assert "File not found" in capsys.readouterr().out
