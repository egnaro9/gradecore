"""Trajectory grader — scores an agent's *action graph*, not just its final text.

The other lenses grade an answer string; an agent's quality also lives in *what
it did* — which tools it called, and in what order. This grades the observed tool
sequence (``GradeInput.tool_calls``) against an expected ordered plan, with
partial credit for following part of it.

Metric: the longest common subsequence (LCS) of the observed tool names and the
expected ones, over the expected length. LCS is the right shape for a trajectory
— it rewards hitting the required steps *in order* while tolerating extra calls
in between, and it degrades gracefully (miss one step of three -> 0.67) instead
of being all-or-nothing. Pure and deterministic: the same trajectory always
scores the same, no LLM-as-judge.

This is the correctness complement to ``tool_misuse``: ``tool_misuse`` asks "did
the agent touch a forbidden tool?"; ``trajectory`` asks "did it follow the right
plan?". Different questions, so they're different graders.
"""
from __future__ import annotations

from typing import List, Sequence

from .graders import _preview
from .verdict import GradeInput, Grader, Verdict, check_severity


def _names(tool_calls: Sequence[dict]) -> List[str]:
    """The ordered, lowercased tool names of a trajectory.

    Reads the same ``{"tool": ...}`` call shape as ``tool_misuse``; a call with
    no ``tool`` key contributes an empty name but still occupies a step.
    """
    return [str(c.get("tool", "")).lower() for c in (tool_calls or ())]


def _lcs_len(a: Sequence[str], b: Sequence[str]) -> int:
    """Length of the longest common subsequence — the in-order overlap of two
    sequences. Classic O(len(a)*len(b)) DP over a rolling row."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        curr = [0]
        for j, y in enumerate(b):
            curr.append(prev[j] + 1 if x == y else max(prev[j + 1], curr[j]))
        prev = curr
    return prev[-1]


def trajectory_score(observed: Sequence[str], expected: Sequence[str],
                     *, allow_extra: bool = True) -> float:
    """Fraction of the expected trajectory followed, in order.

    ``allow_extra`` (default True): score = LCS / len(expected) — extra,
    unexpected calls between the right steps don't lower the score (an
    unexpected *forbidden* tool is a ``tool_misuse`` concern, not an ordering
    one). Set it False to also penalize wandering:
    score = LCS / max(len(expected), len(observed)).

    With no expected steps, a trajectory is faithful iff it made no calls — a
    clean way to assert "this task should be answered without any tools".
    """
    exp = list(expected)
    obs = list(observed)
    if not exp:
        return 1.0 if not obs else 0.0
    lcs = _lcs_len(obs, exp)
    denom = len(exp) if allow_extra else max(len(exp), len(obs))
    return lcs / denom


def trajectory(*expected: str, threshold: float = 1.0, allow_extra: bool = True,
               fail_severity: str = "high") -> Grader:
    """Grade ``GradeInput.tool_calls`` against an ordered ``expected`` plan.

    PASS iff the in-order coverage ``>= threshold`` (default 1.0 — every expected
    step taken, in order; extra detours tolerated unless ``allow_extra=False``).
    The score is the coverage fraction itself, so it carries partial credit; a
    divergent trajectory fails at ``fail_severity`` (default "high" — a wrong
    action graph is a real agent failure, not a formatting nit).
    """
    check_severity(fail_severity)
    exp = [e.lower() for e in expected]

    def g(inp: GradeInput) -> Verdict:
        obs = _names(inp.tool_calls)
        frac = trajectory_score(obs, exp, allow_extra=allow_extra)
        ok = frac >= threshold
        detail = (
            f"followed {frac:.2f} of plan {exp}"
            if ok
            else (f"trajectory {frac:.2f} < {threshold:.2f}: expected {exp}, "
                  f"observed {_preview(' -> '.join(obs) or '(none)')}")
        )
        return Verdict(
            passed=ok,
            score=round(frac, 4),
            severity="none" if ok else fail_severity,
            detail=detail,
            grader_id="trajectory",
        )
    return g
