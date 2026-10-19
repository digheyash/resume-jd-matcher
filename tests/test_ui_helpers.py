"""Tests for app/ui_helpers.py (the dashboard's plain, non-Streamlit helpers).

No browser, no PDF and no model is needed. The PDF loader is replaced by a
tiny fake, so these tests check OUR logic (file type, size limit, messages,
HTML pieces), not pdfplumber. Numbers in the ring tests are worked out by hand
in the comments: the ring circumference is 2 * pi * 58 = 364.4 (rounded).
"""

import pytest

from resume_matcher.analyzer.analyzer import AnalysisError
from resume_matcher.parsing.loader import DocumentLoadError, LoadedDocument
from resume_matcher.schemas import AnalysisResult, SkillGap
from ui_helpers import (
    MAX_UPLOAD_BYTES,
    MAX_UPLOAD_MB,
    VERDICT_STYLES,
    card_html,
    chip_html,
    chips_html,
    combine_warnings,
    coverage_detail,
    coverage_line,
    coverage_text,
    decode_text_bytes,
    error_box_html,
    format_cosine,
    format_percent,
    format_score,
    friendly_error,
    input_problems,
    meaning_match_detail,
    meaning_match_line,
    meaning_match_text,
    missing_section_html,
    read_uploaded_file,
    score_ring_html,
    skill_count_text,
    stat_html,
    verdict_badge_html,
    verdict_style,
    warnings_box_html,
)


# ----------------------------------------------------------------------
# Fakes and builders
# ----------------------------------------------------------------------
class FakeLoader:
    """Stands in for a loader: remembers what it was given."""

    def __init__(self, text="LOADED TEXT", warnings=None):
        self.calls = []
        self._doc = LoadedDocument(
            text=text, source="fake", page_count=1, warnings=list(warnings or [])
        )

    def __call__(self, value):
        self.calls.append(value)
        return self._doc


def gap(skill, importance="must-have"):
    return SkillGap(skill, importance)


def make_result(matched=(), missing=()):
    return AnalysisResult(
        overall_score=50.0,
        verdict="Partial Fit",
        semantic_score=0.4,
        skill_coverage=0.5,
        matched_skills=list(matched),
        missing_skills=list(missing),
        extra_skills=[],
        warnings=[],
    )


# ----------------------------------------------------------------------
# decode_text_bytes
# ----------------------------------------------------------------------
def test_decode_plain_utf8():
    assert decode_text_bytes("café".encode("utf-8")) == "café"


def test_decode_removes_a_notepad_bom():
    assert decode_text_bytes(b"\xef\xbb\xbfhello") == "hello"


def test_decode_falls_back_to_cp1252():
    # 0x93 and 0x94 are curly quotes in Windows-1252 but control characters in
    # latin-1, so this only passes if cp1252 is tried before latin-1.
    assert decode_text_bytes(b"\x93quoted\x94") == "“quoted”"


def test_decode_falls_back_to_latin1_for_bytes_cp1252_cannot_read():
    # 0x81 is undefined in cp1252; latin-1 maps every byte, so it never fails.
    assert decode_text_bytes(b"\x81") == "\x81"


# ----------------------------------------------------------------------
# read_uploaded_file
# ----------------------------------------------------------------------
def test_pdf_goes_to_the_pdf_loader_with_the_raw_bytes():
    pdf, text = FakeLoader("FROM PDF"), FakeLoader("FROM TEXT")
    doc = read_uploaded_file("resume.pdf", b"%PDF-1.7 data", pdf, text)
    assert doc.text == "FROM PDF"
    assert pdf.calls == [b"%PDF-1.7 data"]
    assert text.calls == []


def test_file_type_is_not_case_sensitive():
    pdf = FakeLoader()
    read_uploaded_file("RESUME.PDF", b"data", pdf, FakeLoader())
    assert len(pdf.calls) == 1


@pytest.mark.parametrize("name", ["cv.txt", "cv.md", "CV.TXT"])
def test_text_files_are_decoded_and_sent_to_the_text_loader(name):
    pdf, text = FakeLoader(), FakeLoader("FROM TEXT")
    doc = read_uploaded_file(name, "naïve résumé".encode("utf-8"), pdf, text)
    assert doc.text == "FROM TEXT"
    assert text.calls == ["naïve résumé"]
    assert pdf.calls == []


def test_default_text_loader_returns_the_decoded_text():
    doc = read_uploaded_file("cv.txt", b"hello world")
    assert doc.text == "hello world"


def test_empty_file_is_refused():
    with pytest.raises(DocumentLoadError, match="empty"):
        read_uploaded_file("resume.pdf", b"", FakeLoader(), FakeLoader())


def test_file_over_the_limit_is_refused_and_the_message_names_the_limit():
    too_big = b"x" * (MAX_UPLOAD_BYTES + 1)
    with pytest.raises(DocumentLoadError) as info:
        read_uploaded_file("cv.txt", too_big, FakeLoader(), FakeLoader())
    assert f"{MAX_UPLOAD_MB} MB" in str(info.value)


def test_upload_limit_matches_the_streamlit_config():
    # .streamlit/config.toml sets server.maxUploadSize = 5 (megabytes).
    # If one changes, change the other.
    assert MAX_UPLOAD_MB == 5
    assert MAX_UPLOAD_BYTES == 5 * 1024 * 1024


def test_file_exactly_at_the_limit_is_accepted():
    text = FakeLoader()
    read_uploaded_file("cv.txt", b"x" * MAX_UPLOAD_BYTES, FakeLoader(), text)
    assert len(text.calls) == 1


def test_unsupported_type_is_refused_with_its_extension():
    with pytest.raises(DocumentLoadError, match="Unsupported file type '.docx'"):
        read_uploaded_file("resume.docx", b"data", FakeLoader(), FakeLoader())


def test_file_without_an_extension_is_refused_by_name():
    with pytest.raises(DocumentLoadError, match="Unsupported file type 'resume'"):
        read_uploaded_file("resume", b"data", FakeLoader(), FakeLoader())


def test_loader_errors_pass_through_unchanged():
    def broken(_data):
        raise DocumentLoadError("Could not open this file as a PDF.")

    with pytest.raises(DocumentLoadError, match="Could not open"):
        read_uploaded_file("resume.pdf", b"data", broken, FakeLoader())


def test_loader_warnings_are_kept():
    pdf = FakeLoader(warnings=["Very little text found"])
    doc = read_uploaded_file("resume.pdf", b"data", pdf, FakeLoader())
    assert doc.warnings == ["Very little text found"]


# ----------------------------------------------------------------------
# input_problems
# ----------------------------------------------------------------------
def test_input_problems_lists_both_missing_inputs():
    assert input_problems("", "  ") == [
        "Add your resume: upload a file or paste the text.",
        "Paste the job description.",
    ]


def test_input_problems_for_one_missing_input():
    assert input_problems("", "a job") == [
        "Add your resume: upload a file or paste the text."
    ]
    assert input_problems("a resume", "") == ["Paste the job description."]


def test_input_problems_treats_a_blank_resume_as_missing():
    assert input_problems("   \n  ", "a job") == [
        "Add your resume: upload a file or paste the text."
    ]


def test_input_problems_is_empty_when_both_are_present():
    assert input_problems("a resume", "a job") == []


def test_input_problems_treats_none_as_empty():
    assert len(input_problems(None, None)) == 2


# ----------------------------------------------------------------------
# friendly_error
# ----------------------------------------------------------------------
def test_friendly_error_for_an_unreadable_file():
    title, message = friendly_error(DocumentLoadError("The file is empty."))
    assert title == "We could not read that file"
    assert message == "The file is empty."


def test_friendly_error_for_an_analysis_error():
    title, message = friendly_error(AnalysisError("Resume text is empty."))
    assert title == "We could not analyse these texts"
    assert message == "Resume text is empty."


def test_friendly_error_hides_the_details_of_unexpected_errors():
    title, message = friendly_error(RuntimeError("secret C:\\path\\detail"))
    assert title == "Something went wrong"
    assert "secret" not in message
    assert "detail" not in message


# ----------------------------------------------------------------------
# combine_warnings
# ----------------------------------------------------------------------
def test_file_warnings_get_the_prefix():
    assert combine_warnings(["No text"], []) == ["Resume: No text"]


def test_file_warnings_come_before_result_warnings():
    assert combine_warnings(["A"], ["B"]) == ["Resume: A", "B"]


def test_a_warning_already_reported_with_the_prefix_is_not_repeated():
    assert combine_warnings(["No text"], ["Resume: No text"]) == ["Resume: No text"]


def test_a_warning_already_reported_without_the_prefix_is_not_repeated():
    assert combine_warnings(["No text"], ["No text"]) == ["No text"]


def test_a_warning_repeated_in_the_result_is_listed_once():
    assert combine_warnings([], ["B", "B"]) == ["B"]


def test_custom_prefix():
    assert combine_warnings(["A"], [], prefix="JD: ") == ["JD: A"]


def test_combine_warnings_with_nothing_to_combine():
    assert combine_warnings([], []) == []


def test_combine_warnings_does_not_change_its_inputs():
    file_w, result_w = ["A"], ["B"]
    combine_warnings(file_w, result_w)
    assert file_w == ["A"] and result_w == ["B"]


# ----------------------------------------------------------------------
# Number formatting
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "score, expected",
    [(52.8, "53"), (0, "0"), (100, "100"), (130.0, "100"), (-3.0, "0")],
)
def test_format_score(score, expected):
    assert format_score(score) == expected


@pytest.mark.parametrize(
    "value, expected", [(0.424, "0.42"), (0, "0.00"), (1.2, "1.00"), (-0.1, "0.00")]
)
def test_format_cosine(value, expected):
    assert format_cosine(value) == expected


@pytest.mark.parametrize(
    "value, expected", [(5 / 14, "36%"), (0, "0%"), (1, "100%"), (0.5, "50%")]
)
def test_format_percent(value, expected):
    # 5/14 = 35.7 percent, which rounds to 36.
    assert format_percent(value) == expected


def test_skill_count_text_counts_matched_plus_missing():
    result = make_result(
        matched=["a", "b", "c", "d", "e"],
        missing=[gap(f"m{i}") for i in range(9)],
    )
    assert skill_count_text(result) == "5 of 14 skills"


def test_skill_count_text_uses_the_singular_for_one_skill():
    assert skill_count_text(make_result(missing=[gap("Docker")])) == "0 of 1 skill"


def test_skill_count_text_when_nothing_is_required():
    assert skill_count_text(make_result()) == "no skills recognised"


def test_skill_count_text_counts_repeated_names_once():
    result = make_result(matched=["a", "a"], missing=[gap("m"), gap("m")])
    assert skill_count_text(result) == "1 of 2 skills"


# ----------------------------------------------------------------------
# Verdict style and badge
# ----------------------------------------------------------------------
def test_every_verdict_style_has_all_three_colours():
    for style in VERDICT_STYLES.values():
        assert set(style) == {"bg", "fg", "ring"}


def test_verdicts_have_different_ring_colours():
    rings = {verdict_style(v)["ring"] for v in ("Fit", "Partial Fit", "No-Fit")}
    assert len(rings) == 3


def test_unknown_verdict_gets_the_neutral_style():
    assert verdict_style("Maybe") == {"bg": "#EEEDEA", "fg": "#4A4F5C", "ring": "#8A8F9C"}


def test_verdict_style_returns_a_copy():
    verdict_style("Fit")["bg"] = "#000000"
    assert VERDICT_STYLES["Fit"]["bg"] == "#E6F4EA"


def test_badge_shows_the_verdict_in_its_colour():
    html = verdict_badge_html("Partial Fit")
    assert "Partial Fit" in html
    assert "#FEF3DC" in html   # pill background
    assert "#8A5300" in html   # pill text
    assert "#D97706" in html   # the small dot


def test_badge_escapes_the_verdict():
    html = verdict_badge_html("<b>x</b>")
    assert "<b>" not in html
    assert "&lt;b&gt;" in html


# ----------------------------------------------------------------------
# Score ring
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "score, offset",
    [
        # offset = 364.4 * (1 - score / 100)
        (50, "182.2"),    # half the ring is empty
        (0, "364.4"),     # whole ring empty
        (100, "0.0"),     # whole ring filled
        (52.8, "172.0"),  # 364.42 * 0.472 = 172.01
        (150, "0.0"),     # above 100 is held at 100
        (-20, "364.4"),   # below 0 is held at 0
    ],
)
def test_ring_offset(score, offset):
    assert f'stroke-dashoffset="{offset}"' in score_ring_html(score, "Fit")


def test_ring_shows_the_whole_number_and_a_screen_reader_label():
    html = score_ring_html(52.8, "Partial Fit")
    assert ">53</div>" in html
    assert 'aria-label="Match score 53 out of 100, Partial Fit"' in html


@pytest.mark.parametrize(
    "verdict, colour",
    [("Fit", "#2E9D5B"), ("Partial Fit", "#D97706"), ("No-Fit", "#D14343"), ("Maybe", "#8A8F9C")],
)
def test_ring_colour_follows_the_verdict(verdict, colour):
    assert f'stroke="{colour}"' in score_ring_html(50, verdict)


def test_ring_escapes_the_verdict_in_its_label():
    html = score_ring_html(50, '"><script>')
    assert "<script>" not in html
    assert "&quot;&gt;&lt;script&gt;" in html


# ----------------------------------------------------------------------
# Stat tile, chips
# ----------------------------------------------------------------------
def test_stat_tile_shows_label_value_and_detail():
    html = stat_html("Skill coverage", "36%", "5 of 14 skills")
    assert "Skill coverage" in html and "36%" in html and "5 of 14 skills" in html


def test_stat_tile_without_detail_has_no_detail_line():
    # The label and the detail each use font-size:12px.
    assert stat_html("A", "B").count("font-size:12px") == 1
    assert stat_html("A", "B", "C").count("font-size:12px") == 2


def test_stat_tile_escapes_its_text():
    assert "<i>" not in stat_html("<i>", "<i>", "<i>")


@pytest.mark.parametrize(
    "kind, background",
    [
        ("matched", "#E6F4EA"),
        ("must-have", "#FDECEC"),
        ("nice-to-have", "#FEF3DC"),
        ("unknown", "#EEEDEA"),
        ("extra", "#EEEDEA"),
        ("something-else", "#EEEDEA"),
    ],
)
def test_chip_colour_by_kind(kind, background):
    assert background in chip_html("Python", kind)


def test_chip_escapes_the_label():
    html = chip_html("C++ <script>", "matched")
    assert "<script>" not in html
    assert "C++ &lt;script&gt;" in html


def test_chips_keep_their_order_and_one_chip_per_label():
    html = chips_html(["Python", "Docker", "SQL"], "matched")
    assert html.count("border-radius:999px") == 3
    assert html.index("Python") < html.index("Docker") < html.index("SQL")


def test_empty_chip_list_shows_the_empty_text():
    html = chips_html([], "matched", empty_text="Nothing here <yet>")
    assert "Nothing here &lt;yet&gt;" in html
    assert "border-radius:999px" not in html


# ----------------------------------------------------------------------
# Missing-skills section
# ----------------------------------------------------------------------
def test_missing_section_orders_groups_and_shows_counts():
    missing = [
        gap("Git", "unknown"),
        gap("AWS", "nice-to-have"),
        gap("Docker"),
        gap("PyTorch"),
    ]
    html = missing_section_html(missing)
    must = html.index("Must-have &middot; 2")
    nice = html.index("Nice-to-have &middot; 1")
    unknown = html.index("Importance not stated &middot; 1")
    assert must < nice < unknown


def test_missing_section_leaves_out_empty_groups():
    html = missing_section_html([gap("Docker")])
    assert "Must-have" in html
    assert "Nice-to-have" not in html
    assert "Importance not stated" not in html


def test_missing_section_with_nothing_missing():
    assert "No required skills are missing." in missing_section_html([])


def test_missing_section_escapes_skill_names():
    html = missing_section_html([gap("<img src=x>")])
    assert "<img" not in html
    assert "&lt;img src=x&gt;" in html


def test_unexpected_importance_label_lands_in_the_unrated_group():
    html = missing_section_html([gap("Docker", "critical")])
    assert "Importance not stated &middot; 1" in html


# ----------------------------------------------------------------------
# Cards and boxes
# ----------------------------------------------------------------------
def test_card_keeps_its_body_html_and_escapes_the_title():
    html = card_html("A & B", "<p>raw</p>")
    assert "<p>raw</p>" in html
    assert "A &amp; B" in html


def test_card_subtitle_is_optional():
    # Only the subtitle uses margin-top:4px.
    assert "margin-top:4px" not in card_html("T", "<p>x</p>")
    assert "margin-top:4px" in card_html("T", "<p>x</p>", "Sub")


def test_cards_and_boxes_leave_space_below_them():
    # Stacked cards and boxes would otherwise touch each other on the page.
    assert "margin-bottom:16px" in card_html("T", "<p>x</p>")
    assert "margin-bottom:16px" in warnings_box_html(["w"])
    assert "margin-bottom:16px" in error_box_html("T", "m")


def test_no_warnings_gives_an_empty_string():
    assert warnings_box_html([]) == ""


def test_warnings_box_lists_and_escapes_each_warning():
    html = warnings_box_html(["Resume: a & b", "<x>"])
    assert "Things to check" in html
    assert "Resume: a &amp; b" in html
    assert "&lt;x&gt;" in html
    assert "<x>" not in html


def test_error_box_escapes_title_and_message():
    html = error_box_html("<t>", "m <script>")
    assert "&lt;t&gt;" in html
    assert "m &lt;script&gt;" in html
    assert "<script>" not in html


# ----------------------------------------------------------------------
# Streamlit's Markdown step breaks HTML that has newlines
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "build",
    [
        lambda: verdict_badge_html("Fit"),
        lambda: score_ring_html(52.8, "Partial Fit"),
        lambda: stat_html("Label", "Value", "Detail"),
        lambda: chip_html("Python", "matched"),
        lambda: chips_html(["a", "b"], "matched"),
        lambda: chips_html([], "matched"),
        lambda: missing_section_html([gap("a"), gap("b", "nice-to-have"), gap("c", "unknown")]),
        lambda: missing_section_html([]),
        lambda: card_html("Title", "<p>x</p>", "Sub"),
        lambda: warnings_box_html(["one", "two"]),
        lambda: error_box_html("Title", "Message"),
    ],
)
def test_html_pieces_are_a_single_line(build):
    assert "\n" not in build()


# ----------------------------------------------------------------------
# Meaning match and coverage that were "not used"
# ----------------------------------------------------------------------
NO_REQUIREMENTS_WARNING = (
    "No requirement lines were found in the job description; the score uses "
    "skill coverage only."
)


def result_without_requirements(**changes):
    result = make_result(matched=["Java", "C++"], missing=[gap("Python", "unknown")])
    result.semantic_score = 0.0  # analyze() returns this placeholder
    result.skill_coverage = 2 / 3
    result.warnings = [NO_REQUIREMENTS_WARNING]
    for name, value in changes.items():
        setattr(result, name, value)
    return result


def test_meaning_match_is_shown_as_a_number_when_it_was_measured():
    assert meaning_match_text(make_result()) == "0.40"
    assert meaning_match_detail(make_result()) == ""
    assert meaning_match_line(make_result()) == "0.40"


def test_meaning_match_is_not_used_without_requirement_lines():
    result = result_without_requirements()
    assert meaning_match_text(result) == "Not used"
    assert meaning_match_detail(result) == "no requirement lines found"
    assert meaning_match_line(result) == "Not used (no requirement lines found)"


def test_a_measured_zero_is_still_shown_as_a_number():
    # Without the warning, 0.0 is a real (very low) value, not a placeholder.
    result = make_result()
    result.semantic_score = 0.0
    assert meaning_match_text(result) == "0.00"


def test_coverage_is_a_percentage_with_a_count_when_skills_were_found():
    result = make_result(matched=["Python"], missing=[gap("Docker")])
    assert coverage_text(result) == "50%"
    assert coverage_detail(result) == "1 of 2 skills"
    assert coverage_line(result) == "50% (1 of 2 skills)"


def test_coverage_is_not_used_when_no_skills_were_recognised():
    result = make_result()
    result.skill_coverage = 0.0
    assert coverage_text(result) == "Not used"
    assert coverage_detail(result) == "no skills recognised"
    assert coverage_line(result) == "Not used (no skills recognised)"


def test_coverage_still_shows_for_a_result_without_requirement_lines():
    result = result_without_requirements()
    assert coverage_line(result) == "67% (2 of 3 skills)"
