"""Paired comparison between two scored runs — and the refusal to over-claim.

Two configs are compared TASK BY TASK, not by their averages. At small suite
sizes a paired test has far more power than comparing means, and more
importantly it makes the honest failure mode visible: ties carry no direction,
so they are discarded, and what remains is the count of tasks that actually
separated the two.

The number that keeps this honest is ``min_p``: given how many tasks were
informative, the smallest p-value ANY split could produce. If even a clean
sweep cannot clear alpha, the suite cannot answer the question — and reporting
"no significant difference" would be a claim the data cannot support. That case
is reported as underpowered, explicitly, rather than as a tie.

Ported from ``harness_core.js`` (never-touch-ai) so both surfaces share one
implementation of the arithmetic rather than drifting apart. The tests pin the
exact values the JS produces.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import comb
from typing import Mapping, Optional, Sequence

DEFAULT_ALPHA = 0.05
DEFAULT_EPS = 1e-9


def sign_test_p(wins: int, losses: int) -> float:
    """Two-sided exact sign test.

    The probability of a split at least this lopsided arising from a fair coin.
    Exact rather than normal-approximated, because the suites this is built for
    are small enough that the approximation misleads.
    """
    n = wins + losses
    if n == 0:
        return 1.0
    hi = max(wins, losses)
    tail = sum(comb(n, k) for k in range(hi, n + 1))
    return min(1.0, (2 * tail) / (2 ** n))


@dataclass(frozen=True)
class TaskDelta:
    task: str
    winner: Optional[str]   # "a", "b", or None for a tie
    delta: float


@dataclass(frozen=True)
class PairedResult:
    shared: int
    wins: int
    losses: int
    ties: int
    informative: int
    p: float
    decisive: bool
    winner: Optional[str]
    min_p: float
    underpowered: bool
    detail: Sequence[TaskDelta] = field(default_factory=tuple)


def paired_compare(
    scores_a: Mapping[str, float],
    scores_b: Mapping[str, float],
    *,
    eps: float = DEFAULT_EPS,
    alpha: float = DEFAULT_ALPHA,
) -> PairedResult:
    """Compare two ``{task_id: score}`` mappings.

    Only tasks present in BOTH count. A task one config never ran tells you
    nothing about which is better, and silently treating it as a loss is how a
    crashed run turns into a false finding.
    """
    shared = sorted(t for t in scores_a if t in scores_b)
    wins = losses = ties = 0
    detail = []

    for t in shared:
        d = scores_a[t] - scores_b[t]
        if abs(d) <= eps:
            ties += 1
            detail.append(TaskDelta(t, None, 0.0))
        elif d > 0:
            wins += 1
            detail.append(TaskDelta(t, "a", d))
        else:
            losses += 1
            detail.append(TaskDelta(t, "b", d))

    p = sign_test_p(wins, losses)
    decisive = p < alpha
    informative = wins + losses
    min_p = sign_test_p(informative, 0)

    return PairedResult(
        shared=len(shared),
        wins=wins,
        losses=losses,
        ties=ties,
        informative=informative,
        p=p,
        decisive=decisive,
        winner=("a" if wins > losses else "b") if decisive else None,
        min_p=min_p,
        underpowered=min_p >= alpha,
        detail=tuple(detail),
    )


def paired_verdict(r: PairedResult, name_a: str = "A", name_b: str = "B") -> str:
    """Plain-language reading.

    Refuses to call a tie a tie when the suite was never capable of detecting a
    difference in the first place — that distinction is the whole point.
    """
    if r.shared == 0:
        return "No task was scored under both configs, so there is nothing to compare."
    if r.informative == 0:
        return (
            f"{name_a} and {name_b} scored identically on all {r.shared} shared "
            f"tasks — no task separates them."
        )
    if r.underpowered:
        s = "" if r.informative == 1 else "s"
        return (
            f"Only {r.informative} task{s} separated {name_a} and {name_b}. "
            f"Even a clean sweep of {r.informative} could not clear p<0.05, so "
            f"this suite cannot decide between them — that is a limit of the "
            f"suite, not a finding about the configs."
        )
    if not r.decisive:
        return (
            f"{name_a} won {r.wins}, {name_b} won {r.losses}, {r.ties} tied "
            f"(p={r.p:.3f}). Indistinguishable on this suite."
        )
    win = name_a if r.winner == "a" else name_b
    hi, lo = max(r.wins, r.losses), min(r.wins, r.losses)
    return f"{win} is better: won {hi}, lost {lo}, {r.ties} tied (p={r.p:.3f})."


def tasks_needed(alpha: float = DEFAULT_ALPHA) -> int:
    """Smallest number of informative tasks at which any result can reach alpha.

    At the default alpha this is 6: a clean sweep of 5 gives p=0.0625, of 6
    gives p=0.03125. Below it, no split of the data can clear the bar — which is
    worth knowing BEFORE spending money on a run rather than after.
    """
    n = 1
    while sign_test_p(n, 0) >= alpha:
        n += 1
        if n > 1000:  # pragma: no cover - alpha would have to be absurd
            raise ValueError(f"no practical n reaches alpha={alpha}")
    return n
