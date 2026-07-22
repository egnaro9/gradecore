"""The scalar/text grader family + the legacy bool adapter."""
import pytest

from gradecore import (
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
