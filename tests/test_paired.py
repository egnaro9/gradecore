"""Tests for the paired comparison.

Several cases pin values against harness_core.js (never-touch-ai), because the
point of porting rather than reinventing is that both surfaces agree. If one
drifts, these fail.
"""
import pytest

from gradecore import paired_compare, paired_verdict, sign_test_p, tasks_needed


# ---------------------------------------------------------------------------
# THE SIGN TEST. Exact, two-sided.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("wins,losses,expected", [
    (0, 0, 1.0),
    (1, 0, 1.0),        # 2*1/2
    (2, 0, 0.5),
    (3, 0, 0.25),
    (5, 0, 0.0625),     # just misses 0.05 -- the reason 5 is not enough
    (6, 0, 0.03125),    # the first clean sweep that clears it
    (3, 3, 1.0),
])
def test_sign_test_matches_known_values(wins, losses, expected):
    assert sign_test_p(wins, losses) == pytest.approx(expected)


def test_sign_test_is_symmetric():
    # Two-sided: which side won cannot change the probability.
    for a, b in [(4, 1), (7, 2), (9, 3)]:
        assert sign_test_p(a, b) == sign_test_p(b, a)


def test_six_informative_tasks_is_the_floor():
    # The number the whole 100-task scope is built around.
    assert tasks_needed(0.05) == 6


# ---------------------------------------------------------------------------
# THE COMPARISON
# ---------------------------------------------------------------------------
def test_only_shared_tasks_count():
    # A task one config never ran says nothing about which is better. Treating
    # it as a loss is how a crashed run becomes a false finding.
    r = paired_compare({"a": 1.0, "b": 1.0, "onlyA": 1.0}, {"a": 0.0, "b": 1.0})
    assert r.shared == 2
    assert (r.wins, r.losses, r.ties) == (1, 0, 1)


def test_ties_are_discarded_not_counted_as_evidence():
    r = paired_compare({f"t{i}": 1.0 for i in range(20)}, {f"t{i}": 1.0 for i in range(20)})
    assert r.shared == 20 and r.ties == 20 and r.informative == 0
    assert "no task separates them" in paired_verdict(r)


def test_three_discordant_cannot_decide_even_swept():
    # The exact shape of the 2026-07-25 sweep: 17 ties, 3 discordant, one side
    # taking all three. It still cannot clear alpha.
    a = {f"t{i}": 1.0 for i in range(20)}
    b = dict(a)
    for i in range(3):
        b[f"t{i}"] = 0.0
    r = paired_compare(a, b)
    assert (r.wins, r.losses, r.ties) == (3, 0, 17)
    assert r.underpowered is True
    assert r.decisive is False
    assert r.winner is None
    v = paired_verdict(r, "one drafter", "panel")
    assert "cannot decide" in v and "limit of the suite" in v


def test_underpowered_is_not_reported_as_a_tie():
    # The distinction the tool exists to make: "cannot tell" is not "the same".
    a = {"t0": 1.0, "t1": 1.0}
    b = {"t0": 0.0, "t1": 0.0}
    r = paired_compare(a, b)
    v = paired_verdict(r)
    assert r.underpowered
    assert "Indistinguishable" not in v


def test_a_real_win_is_called():
    a = {f"t{i}": 1.0 for i in range(8)}
    b = {f"t{i}": 0.0 for i in range(8)}
    r = paired_compare(a, b, alpha=0.05)
    assert r.decisive and r.winner == "a"
    assert r.p == pytest.approx(2 / 256)
    assert "is better" in paired_verdict(r, "new", "old")


def test_a_genuine_indistinguishable_result_says_so():
    # Enough informative tasks to have detected a difference, split near evenly.
    a = {f"t{i}": (1.0 if i % 2 else 0.0) for i in range(10)}
    b = {f"t{i}": (0.0 if i % 2 else 1.0) for i in range(10)}
    r = paired_compare(a, b)
    assert r.informative == 10 and not r.underpowered and not r.decisive
    assert "Indistinguishable" in paired_verdict(r)


def test_partial_credit_within_eps_is_a_tie():
    # grounding/trajectory return fractions; float noise must not read as a win.
    r = paired_compare({"t": 0.8333333333}, {"t": 0.8333333334})
    assert r.ties == 1 and r.informative == 0


def test_direction_is_reported_per_task():
    r = paired_compare({"x": 1.0, "y": 0.0}, {"x": 0.0, "y": 1.0})
    by = {d.task: d.winner for d in r.detail}
    assert by == {"x": "a", "y": "b"}


def test_nothing_shared_is_not_a_tie():
    r = paired_compare({"a": 1.0}, {"b": 1.0})
    assert r.shared == 0
    assert "nothing to compare" in paired_verdict(r)
