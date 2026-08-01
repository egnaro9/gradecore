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


# ---------------------------------------------------------------------------
# Repeated measures — the correction the noise floor forces
# ---------------------------------------------------------------------------
#
# A single run per config cannot tell a real difference from a model disagreeing
# with itself. Measured on a 41-task suite, claude-haiku-4-5 run three times
# against ITSELF produced ~1.3 "informative" tasks per pairwise comparison. The
# haiku-vs-sonnet comparison on the same suite produced 1. Read from single runs,
# that difference is indistinguishable from noise.
#
# The fix is not a bigger suite. It is repeated measurement plus one rule:
#
#     a task where a config disagrees with ITSELF carries no direction,
#     exactly like a tie, and must be discarded for the same reason.
#
# That rule is what separates the one real finding on that suite (haiku failed
# extract-pro-regular-price 3/3 while sonnet passed) from the two tasks that
# merely flickered.


@dataclass(frozen=True)
class TaskStability:
    task: str
    rate_a: float
    rate_b: float
    stable_a: bool
    stable_b: bool
    verdict: str   # "a", "b", "tie", or "unstable"


@dataclass(frozen=True)
class RepeatedResult:
    reps_a: int
    reps_b: int
    shared: int
    wins: int
    losses: int
    ties: int
    unstable: int
    paired: PairedResult
    detail: Sequence[TaskStability] = field(default_factory=tuple)


DEFAULT_RATE_MARGIN = 0.5


def repeated_compare(
    runs_a: Sequence[Mapping[str, float]],
    runs_b: Sequence[Mapping[str, float]],
    *,
    eps: float = DEFAULT_EPS,
    alpha: float = DEFAULT_ALPHA,
    stability: str = "strict",
    rate_margin: float = DEFAULT_RATE_MARGIN,
) -> RepeatedResult:
    """Compare two configs each measured several times.

    ``runs_a`` / ``runs_b`` are lists of ``{task_id: score}`` mappings, one per
    repetition. A task counts as informative only if BOTH configs were
    self-consistent across their own repetitions AND they disagree with each
    other. Anything flickering within a config is dropped as unstable — reporting
    it as a win would be reporting noise with a direction attached.

    Requires at least 2 repetitions on each side; with one run there is no way to
    observe self-consistency and the honest tool is ``paired_compare``.

    ``stability`` picks the rule for "carries no direction":

    - ``"strict"`` (default) — discard unless BOTH configs were perfectly
      self-consistent. Never counts noise as signal. Its cost is measurable and
      worth knowing: because every extra repetition is another chance to observe
      disagreement, the discard count RISES with more repetitions. On one real
      159-task pair: 2 reps discarded ~8.7 tasks, 3 reps discarded 13. In the limit
      it throws away every genuinely stochastic task — which are exactly the tasks
      carrying the most information about a noisy config. More measurement does not
      fix this; only a different rule does.
    - ``"rate"`` — count a task when the two pass RATES differ by at least
      ``rate_margin``, self-consistent or not. Admits "A wins this most of the
      time", which strict discards. Weaker on purpose: the margin is a judgement
      call, not a test, so report which rule produced a number.
    """
    if stability not in ("strict", "rate"):
        raise ValueError('stability must be "strict" or "rate"')
    if len(runs_a) < 2 or len(runs_b) < 2:
        raise ValueError(
            "repeated_compare needs >= 2 repetitions per config; "
            "with a single run, use paired_compare and read min_p"
        )

    shared = sorted(
        t for t in runs_a[0]
        if all(t in r for r in runs_a) and all(t in r for r in runs_b)
    )

    detail = []
    stable_a_scores: dict[str, float] = {}
    stable_b_scores: dict[str, float] = {}
    unstable = 0

    for t in shared:
        va = [r[t] for r in runs_a]
        vb = [r[t] for r in runs_b]
        rate_a, rate_b = sum(va) / len(va), sum(vb) / len(vb)
        stable_a = max(va) - min(va) <= eps
        stable_b = max(vb) - min(vb) <= eps
        if stability == "rate":
            # A task counts when the two PASS RATES are far enough apart, whether
            # or not either config was internally consistent. Deliberately weaker
            # than "strict" and reported as such: the threshold is a judgement, not
            # a test, and it admits tasks a config merely tends to win.
            usable = abs(rate_a - rate_b) >= rate_margin or abs(rate_a - rate_b) <= eps
            stable_a = stable_b = usable

        if not (stable_a and stable_b):
            unstable += 1
            verdict = "unstable"
        else:
            stable_a_scores[t] = rate_a
            stable_b_scores[t] = rate_b
            verdict = ("tie" if abs(rate_a - rate_b) <= eps
                       else "a" if rate_a > rate_b else "b")
        detail.append(TaskStability(t, rate_a, rate_b, stable_a, stable_b, verdict))

    paired = paired_compare(stable_a_scores, stable_b_scores, eps=eps, alpha=alpha)
    return RepeatedResult(
        reps_a=len(runs_a),
        reps_b=len(runs_b),
        shared=len(shared),
        wins=paired.wins,
        losses=paired.losses,
        ties=paired.ties,
        unstable=unstable,
        paired=paired,
        detail=tuple(detail),
    )


def repeated_verdict(r: RepeatedResult, name_a: str = "A", name_b: str = "B") -> str:
    """Plain-language reading that names the unstable tasks as their own category.

    Folding unstable tasks into "tied" would hide the most actionable number in
    the report: how much of the suite is measuring noise.
    """
    head = paired_verdict(r.paired, name_a, name_b)
    if r.unstable == 0:
        return f"{head} All {r.shared} tasks were self-consistent across repetitions."
    s = "" if r.unstable == 1 else "s"
    return (
        f"{head} {r.unstable} of {r.shared} task{s} was discarded as unstable — "
        f"a config disagreed with itself across repetitions, so the task carries "
        f"no direction no matter which config 'won' it on any single run."
    )


def noise_floor(runs: Sequence[Mapping[str, float]], *, eps: float = DEFAULT_EPS) -> float:
    """Mean informative-task count from comparing a config's repetitions to EACH OTHER.

    This is the number any real finding has to clear. If a comparison between two
    different configs yields no more informative tasks than a config compared with
    itself, the "difference" is within the instrument's own variance.
    """
    if len(runs) < 2:
        raise ValueError("noise_floor needs >= 2 repetitions")
    counts = []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            counts.append(paired_compare(runs[i], runs[j], eps=eps).informative)
    return sum(counts) / len(counts)
