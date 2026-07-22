"""The scalar/text grader family.

Lifted verbatim in behavior from model-drift's suite combinators
(`suite.py` `_exact/_contains/_regex/_exact_cs/_one_of/_number`) into the
gradecore `GradeInput -> Verdict` signature. Plus `bool_grader`, an adapter that
lifts any existing `Callable[[str], bool]` unchanged — so model-drift's frozen
SUITE runs through gradecore without being rewritten (Phase 0 reuses it as-is).

Each factory takes a `fail_severity` (default "med"): a pass is always
`severity="none"`, and the task decides how bad a failure is — a hallucination
fail is "high", a formatting nit "low". The battery assigns it per task.
"""
from __future__ import annotations

import re
from typing import Callable

from .verdict import GradeInput, Grader, Verdict, check_severity


def _preview(text: str, n: int = 80) -> str:
    """A one-line, length-capped echo of the model's output for fail cards."""
    t = " ".join((text or "").split())
    return t if len(t) <= n else t[: n - 1] + "…"


def _verdict(ok: bool, grader_id: str, detail: str, fail_severity: str) -> Verdict:
    return Verdict(
        passed=ok,
        score=1.0 if ok else 0.0,
        severity="none" if ok else fail_severity,
        detail=detail,
        grader_id=grader_id,
    )


def bool_grader(fn: Callable[[str], bool], grader_id: str,
                fail_severity: str = "med") -> Grader:
    """Lift a legacy `Callable[[str], bool]` (e.g. model-drift's `Task.grade`)
    into a Grader, so an existing frozen suite runs through gradecore unchanged."""
    check_severity(fail_severity)

    def g(inp: GradeInput) -> Verdict:
        ok = bool(fn(inp.text))
        return _verdict(ok, grader_id, f"got {_preview(inp.text)!r}", fail_severity)
    return g


def exact(expected: str, *, fail_severity: str = "med") -> Grader:
    """Exact match, case- and whitespace-insensitive."""
    check_severity(fail_severity)
    e = expected.strip().lower()

    def g(inp: GradeInput) -> Verdict:
        ok = (inp.text or "").strip().lower() == e
        return _verdict(ok, "exact",
                        f"expected {expected!r}, got {_preview(inp.text)!r}", fail_severity)
    return g


def contains(*needles: str, fail_severity: str = "med") -> Grader:
    """Passes iff every needle appears (case-insensitive)."""
    check_severity(fail_severity)
    ns = [n.lower() for n in needles]

    def g(inp: GradeInput) -> Verdict:
        ok = all(n in (inp.text or "").lower() for n in ns)
        return _verdict(ok, "contains", f"needs all of {list(needles)}", fail_severity)
    return g


def regex(pattern: str, *, fail_severity: str = "med") -> Grader:
    """Passes iff the pattern is found (DOTALL + IGNORECASE, like model-drift)."""
    check_severity(fail_severity)
    rx = re.compile(pattern, re.I | re.S)

    def g(inp: GradeInput) -> Verdict:
        ok = rx.search(inp.text or "") is not None
        return _verdict(ok, "regex", f"match {pattern!r}", fail_severity)
    return g


def exact_cs(expected: str, *, fail_severity: str = "med") -> Grader:
    """Case-*sensitive* exact match — for tasks where the case is the instruction."""
    check_severity(fail_severity)

    def g(inp: GradeInput) -> Verdict:
        ok = (inp.text or "").strip() == expected
        return _verdict(ok, "exact_cs", f"expected exactly {expected!r}", fail_severity)
    return g


def one_of(*allowed: str, fail_severity: str = "med") -> Grader:
    """Any of several correct answers, exactly (trailing '.' tolerated)."""
    check_severity(fail_severity)
    opts = {a.strip().lower() for a in allowed}

    def g(inp: GradeInput) -> Verdict:
        ok = (inp.text or "").strip().lower().rstrip(".") in opts
        return _verdict(ok, "one_of", f"one of {list(allowed)}", fail_severity)
    return g


def number(expected: float, tol: float = 1e-6, *, which: str = "first",
           fail_severity: str = "med") -> Grader:
    """Extract a number from the output and tolerance-compare it.

    `which` picks which number when several are present: 'first' (default —
    model-drift's terse one-number answers), 'last' (the restated *result* of a
    task that echoes its operands, e.g. "2 + 2 = 4"), or 'any' (passes if the
    expected value appears anywhere among the numbers). The default preserves the
    original first-number behavior.
    """
    check_severity(fail_severity)
    if which not in ("first", "last", "any"):
        raise ValueError("which must be 'first', 'last', or 'any'")

    def g(inp: GradeInput) -> Verdict:
        vals = [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", (inp.text or "").replace(",", ""))]
        if which == "any":
            ok = any(abs(v - expected) <= tol for v in vals)
        elif vals:
            ok = abs((vals[-1] if which == "last" else vals[0]) - expected) <= tol
        else:
            ok = False
        return _verdict(ok, "number", f"expected {expected} (±{tol}, {which})", fail_severity)
    return g
