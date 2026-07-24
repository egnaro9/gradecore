"""The trajectory grader — scoring an agent's ordered action graph."""
from gradecore import GradeInput, trajectory, trajectory_score
from gradecore.trajectory import _lcs_len


def _calls(*names):
    return tuple({"tool": n} for n in names)


def test_lcs_len_basic():
    assert _lcs_len(["a", "b", "c"], ["a", "b", "c"]) == 3
    assert _lcs_len(["a", "x", "b", "y", "c"], ["a", "b", "c"]) == 3  # detours ignored
    assert _lcs_len(["b", "a", "c"], ["a", "b", "c"]) == 2            # reorder costs
    assert _lcs_len([], ["a"]) == 0


def test_score_perfect_partial_and_reordered():
    plan = ["search", "fetch", "summarize"]
    assert trajectory_score(["search", "fetch", "summarize"], plan) == 1.0
    assert trajectory_score(["search", "summarize"], plan) == 2 / 3       # missed a step
    assert trajectory_score(["fetch", "search", "summarize"], plan) == 2 / 3  # wrong order
    assert trajectory_score([], plan) == 0.0                              # did nothing


def test_allow_extra_toggle_penalizes_wandering():
    plan = ["search", "fetch"]
    wandered = ["search", "calc", "calc", "fetch"]
    assert trajectory_score(wandered, plan, allow_extra=True) == 1.0          # detours free
    assert trajectory_score(wandered, plan, allow_extra=False) == 2 / 4       # detours cost


def test_empty_plan_means_no_tools_expected():
    assert trajectory_score([], []) == 1.0
    assert trajectory_score(["search"], []) == 0.0


def test_grader_passes_on_ordered_plan():
    g = trajectory("search", "fetch", "summarize")
    v = g(GradeInput(text="done", tool_calls=_calls("search", "fetch", "summarize")))
    assert v.passed and v.score == 1.0 and v.severity == "none" and v.grader_id == "trajectory"


def test_grader_fails_and_reports_divergence():
    g = trajectory("search", "fetch", "summarize")  # threshold defaults to 1.0
    v = g(GradeInput(text="?", tool_calls=_calls("search", "summarize")))
    assert v.passed is False
    assert v.score == round(2 / 3, 4)
    assert v.severity == "high"
    assert "expected" in v.detail and "observed" in v.detail


def test_grader_threshold_allows_partial_credit_to_pass():
    g = trajectory("search", "fetch", "summarize", threshold=0.6)
    v = g(GradeInput(text="ok", tool_calls=_calls("search", "summarize")))  # 0.67 >= 0.6
    assert v.passed is True and v.severity == "none"


def test_grader_is_case_insensitive_on_tool_names():
    g = trajectory("Search", "Fetch")
    v = g(GradeInput(text="ok", tool_calls=_calls("search", "FETCH")))
    assert v.passed and v.score == 1.0


def test_grader_rejects_bad_severity():
    import pytest
    with pytest.raises(ValueError):
        trajectory("search", fail_severity="catastrophic")
