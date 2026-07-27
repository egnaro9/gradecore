"""The scalar/text grader family + the legacy bool adapter."""
import pytest

from gradecore import (
    last_line,
    SEVERITIES,
    GradeInput,
    bool_grader,
    contains,
    exact,
    exact_cs,
    number,
    one_of,
    regex,
)


def gi(text, **kw):
    return GradeInput(text=text, **kw)


def test_exact_pass_and_fail():
    g = exact("blue")
    p = g(gi("  BLUE "))
    assert p.passed and p.score == 1.0 and p.severity == "none" and p.grader_id == "exact"
    f = g(gi("the sky is blue"))
    assert not f.passed and f.score == 0.0 and f.severity == "med"


def test_contains_needs_all_needles():
    g = contains("signal", "process")
    assert g(gi("send a signal to a process")).passed
    assert not g(gi("send a signal")).passed


def test_regex_is_dotall_and_case_insensitive():
    g = regex(r"error:\s*\d+")
    assert g(gi("ERROR: 42")).passed
    assert not g(gi("no code")).passed


def test_exact_cs_is_case_sensitive():
    g = exact_cs("HELLO")
    assert g(gi("HELLO")).passed
    assert not g(gi("hello")).passed


def test_one_of_tolerates_a_trailing_period():
    g = one_of("grey", "gray")
    assert g(gi("Gray.")).passed
    assert not g(gi("blue")).passed


def test_number_extracts_and_tolerance_compares():
    g = number(3.14, tol=0.01)
    assert g(gi("about 3.141")).passed
    assert not g(gi("about 3.2")).passed
    assert not g(gi("no digits here")).passed


def test_number_handles_commas_and_sign():
    assert number(-1200)(gi("that's -1,200 units")).passed


def test_empty_output_fails_cleanly_and_never_crashes():
    for g in (exact("x"), contains("x"), regex("x"), exact_cs("x"), one_of("x"), number(1)):
        v = g(gi(""))
        assert not v.passed                 # empty text never accidentally passes
        assert v.severity in SEVERITIES


def test_fail_severity_is_configurable_per_task():
    assert exact("x", fail_severity="critical")(gi("y")).severity == "critical"
    assert exact("x", fail_severity="critical")(gi("x")).severity == "none"  # pass is always none


def test_an_invalid_severity_is_rejected_at_construction():
    with pytest.raises(ValueError):
        exact("x", fail_severity="apocalyptic")


def test_bool_grader_lifts_a_legacy_callable_unchanged():
    legacy = lambda out: "yes" in out.lower()      # a model-drift-style Callable[[str], bool]
    g = bool_grader(legacy, grader_id="legacy-yes", fail_severity="low")
    p = g(gi("Yes indeed"))
    assert p.passed and p.grader_id == "legacy-yes" and p.score == 1.0
    f = g(gi("nope"))
    assert not f.passed and f.severity == "low"


def test_number_which_last_fixes_the_echoed_operand_gotcha():
    from gradecore import number
    # "2 + 2 = 4" — first number is 2 (the false-positive), last is the result 4.
    assert not number(4, which="first")(gi("2 + 2 = 4")).passed
    assert number(4, which="last")(gi("2 + 2 = 4")).passed
    assert not number(4, which="last")(gi("2 + 2 = 5")).passed
    assert number(4, which="any")(gi("adding 2 and 2 gives 4")).passed


# ---------------------------------------------------------------------------
# scope="last_line" — separating "got it wrong" from "narrated the answer"
#
# Measured on real runs: turning thinking OFF makes a model narrate, so a config
# change that alters verbosity showed up as an accuracy difference. Two of eight
# wins in a thinking=high vs thinking=off comparison were the model being marked
# wrong for answers it had stated correctly on its final line.
# ---------------------------------------------------------------------------

WORKED = "Days later: 31 - 1 = 30\n30 / 7 = 4 weeks and 2 days\nWednesday + 2 = Friday\n\nFriday"


def test_full_scope_still_fails_a_worked_solution():
    # The default must not move — every existing suite's scores depend on it.
    assert not exact("Friday")(GradeInput(text=WORKED)).passed


def test_last_line_scope_accepts_a_worked_solution():
    assert exact("Friday", scope="last_line")(GradeInput(text=WORKED)).passed


def test_last_line_scope_still_fails_a_wrong_final_answer():
    # The whole value is that it forgives FORMAT, not correctness.
    wrong = WORKED.replace("\n\nFriday", "\n\nThursday")
    assert not exact("Friday", scope="last_line")(GradeInput(text=wrong)).passed


def test_last_line_ignores_trailing_blank_lines():
    assert exact("Friday", scope="last_line")(GradeInput(text="work\n\nFriday\n\n   \n")).passed


def test_one_of_takes_the_scope_too():
    g = one_of("US", "us", scope="last_line")
    assert g(GradeInput(text="An imperial gallon is bigger, but per ounce:\n\nUS")).passed
    assert not g(GradeInput(text="reasoning\n\nimperial")).passed


def test_last_line_of_a_single_line_reply_is_that_reply():
    assert last_line("4") == "4"
    assert last_line("") == ""
    assert last_line(None) == ""


def test_an_unknown_scope_is_rejected_at_construction():
    # Not at grade time — a typo'd scope must not silently grade the full reply
    # for a whole suite before anyone notices.
    with pytest.raises(ValueError, match="scope"):
        exact("x", scope="lastline")


def test_number_scope_last_line_beats_which_last_on_trailing_prose():
    # The mirror-image bug that cost three false failures: fixing which="first"
    # with which="last" just moved the problem to the other end of the reply.
    trailing = "6500/480 = 13.54, so 14 days, which leaves 220 extra widgets"
    assert not number(14, 0, which="last")(GradeInput(text=trailing)).passed
    ended = "6500/480 = 13.54, which leaves 220 extra widgets\n\n14"
    assert number(14, 0, scope="last_line")(GradeInput(text=ended)).passed


def test_number_scope_does_not_rescue_a_wrong_final_answer():
    listed_four_said_five = "Leap years: 1892, 1896, 1904, 1908\n\n5"
    assert not number(4, 0, scope="last_line")(GradeInput(text=listed_four_said_five)).passed


def test_number_rejects_an_unknown_scope_at_construction():
    with pytest.raises(ValueError, match="scope"):
        number(1, scope="tail")
