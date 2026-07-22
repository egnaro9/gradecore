"""The unified grade types.

model-drift's graders are `Callable[[str], bool]`; rag-eval-lab's return floats
gated by thresholds. gradecore reconciles both under one signature —
`GradeInput -> Verdict` — that generalizes boolean into a severity-scored verdict
without losing the boolean case (a pass is score 1.0, a fail 0.0).

Pure data: no I/O, no network, and — the whole point — no LLM-as-judge. Every
verdict is a predicate over a string (and, later, over retrieved context or a
tool trajectory), so a grade is reproducible and inspectable, not a second
model's opinion you can't rerun.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

# Ordered least -> most severe. The leaderboard weighting and the severity bands
# rely on this being an ordering, so keep it ordered.
SEVERITIES = ("none", "low", "med", "high", "critical")


def check_severity(severity: str) -> str:
    if severity not in SEVERITIES:
        raise ValueError(f"severity must be one of {SEVERITIES}, got {severity!r}")
    return severity


@dataclass(frozen=True)
class GradeInput:
    """A superset of what any grader needs, so one signature serves all of them.

    Scalar/text graders read `text` (and sometimes `expected`); the
    retrieval/grounding family reads `retrieved`/`gold_ids`/`contexts`; the
    adversarial graders (Phase 1) read `prompt` (for injection canaries) and
    `tool_calls` (for trajectory scoring). A grader ignores the fields it doesn't
    use.
    """
    text: str
    prompt: Optional[str] = None
    retrieved: Sequence[str] = ()
    contexts: Sequence[str] = ()
    citations: Sequence[str] = ()
    tool_calls: Sequence[dict] = ()
    gold_ids: Sequence[str] = ()
    expected: object = None


@dataclass(frozen=True)
class Verdict:
    passed: bool
    score: float          # 0.0..1.0 — partial credit; a boolean grader emits 0/1
    severity: str         # one of SEVERITIES; "none" whenever passed is True
    detail: str           # short, human-readable — what was expected / went wrong
    grader_id: str        # which grader produced this (for fail cards + audit)


Grader = Callable[[GradeInput], "Verdict"]
