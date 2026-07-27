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



def last_line(text: str | None) -> str:
    """The final non-empty line of a reply.

    A task that says "reply with only the number" is asking about the ANSWER; a
    model that shows its work and then states the answer got the answer right and
    the format wrong. Grading the whole reply conflates those, and the conflation
    is not random: turning thinking off makes a model narrate, so a config change
    that alters VERBOSITY shows up as an accuracy difference. Measured: two of
    eight wins in a thinking=high vs thinking=off comparison were the model being
    marked wrong for answers it had stated correctly on the final line.

    Format compliance is worth measuring — with tasks built for it, not as a
    silent tax on every other task. Hence `scope`, explicit and opt-in.
    """
    for line in reversed((text or "").splitlines()):
        if line.strip():
            return line.strip()
    return (text or "").strip()


SCOPES = ("full", "last_line")


def _scoped(text: str | None, scope: str) -> str:
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}, got {scope!r}")
    return last_line(text) if scope == "last_line" else (text or "")


def exact(expected: str, *, scope: str = "full", fail_severity: str = "med") -> Grader:
    """Exact match, case- and whitespace-insensitive.

    `scope="last_line"` matches against the final non-empty line instead of the
    whole reply, so a worked solution ending in its answer still passes. Default
    stays "full" — changing it silently would move every existing suite's scores.
    """
    check_severity(fail_severity)
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}, got {scope!r}")
    e = expected.strip().lower()

    def g(inp: GradeInput) -> Verdict:
        ok = _scoped(inp.text, scope).strip().lower() == e
        return _verdict(ok, "exact",
                        f"expected {expected!r}, got {_preview(_scoped(inp.text, scope))!r}",
                        fail_severity)
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


def one_of(*allowed: str, scope: str = "full", fail_severity: str = "med") -> Grader:
    """Any of several correct answers, exactly (trailing '.' tolerated).

    `scope="last_line"` grades the final non-empty line; see `last_line`.
    """
    check_severity(fail_severity)
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}, got {scope!r}")
    opts = {a.strip().lower() for a in allowed}

    def g(inp: GradeInput) -> Verdict:
        ok = _scoped(inp.text, scope).strip().lower().rstrip(".") in opts
        return _verdict(ok, "one_of", f"one of {list(allowed)}", fail_severity)
    return g


def number(expected: float, tol: float = 1e-6, *, which: str = "first",
           scope: str = "full", fail_severity: str = "med") -> Grader:
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
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}, got {scope!r}")

    def g(inp: GradeInput) -> Verdict:
        # scope narrows WHERE to look, `which` picks within it. scope="last_line"
        # is the robust pairing: "...leaves 220 extra widgets" defeats which="last"
        # over the whole reply, but a reply whose final line is the answer does not.
        src = _scoped(inp.text, scope)
        vals = [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", src.replace(",", ""))]
        if which == "any":
            ok = any(abs(v - expected) <= tol for v in vals)
        elif vals:
            ok = abs((vals[-1] if which == "last" else vals[0]) - expected) <= tol
        else:
            ok = False
        return _verdict(ok, "number",
                        f"expected {expected} (±{tol}, {which}"
                        + (f", {scope}" if scope != "full" else "") + ")",
                        fail_severity)
    return g
