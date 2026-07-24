"""Grounding (faithfulness) grader — the retrieval/RAG lens of gradecore.

Extracted from rag-eval-lab so one engine grades three lenses: model-drift
(longitudinal), the crash-test platform (adversarial), and rag-eval-lab
(retrieval grounding). The logic is byte-identical to rag-eval-lab's original
``faithfulness``: a lowercase ``[a-z0-9]+`` tokenizer, a small content-word
stoplist, and the fraction of the answer's content tokens that also appear in
the retrieved context. Pure and deterministic — no LLM-as-judge — so a grounding
score reproduces exactly, and the SciFact numbers rag-eval-lab reports are
unchanged by the move.
"""
from __future__ import annotations

import re
from typing import List, Sequence

from .verdict import GradeInput, Grader, Verdict, check_severity

# The one tokenizer used across the retrieval/grounding family. Matches
# rag-eval-lab's ``ragevallab.embedder.tokenize`` exactly — do NOT "improve" it;
# the grounding numbers are calibrated against this precise behavior.
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    """Lowercase word tokenizer: the substrings matching ``[a-z0-9]+``."""
    return _TOKEN_RE.findall(text.lower())


# Common function words, so grounding measures *content* overlap rather than
# stopword coincidence. Identical to rag-eval-lab's ``_STOP``.
STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be",
    "to", "of", "in", "on", "for", "with", "as", "by", "at", "it", "its",
    "this", "that", "these", "those", "from", "into", "than", "then", "so",
    "you", "your", "i", "we", "they", "he", "she", "them", "can", "will",
})

FAITHFULNESS_THRESHOLD = 0.6


def content_tokens(text: str) -> List[str]:
    """Tokens that carry meaning — everything but the stopwords."""
    return [t for t in tokenize(text) if t not in STOPWORDS]


def grounding_score(answer: str, contexts: Sequence[str]) -> float:
    """Fraction of the answer's *content* tokens grounded in ``contexts``.

    ``1.0`` = every content word appears in a retrieved chunk. A fabricated
    answer that mentions entities absent from the context scores well below 1.0.
    An empty answer scores ``0.0`` (there is no content to ground).

    That ``0.0`` is a floor, not a claim that an honest "I don't know" is a
    hallucination — distinguishing a legitimate abstention from a fabrication is
    the job of ``must_abstain``, not this grader.
    """
    ans = content_tokens(answer)
    if not ans:
        return 0.0
    supported: set = set()
    for c in contexts:
        supported |= set(content_tokens(c))
    grounded = sum(1 for t in ans if t in supported)
    return grounded / len(ans)


def grounding(threshold: float = FAITHFULNESS_THRESHOLD, *, severity: str = "med") -> Grader:
    """A grader: is the answer (``text``) faithful to its retrieved ``contexts``?

    Passes when the grounded fraction ``>= threshold``. The score is the fraction
    itself, so it carries partial credit; a failure is a hallucination signal and
    defaults to ``med`` severity.
    """
    check_severity(severity)

    def _grounding(gi: GradeInput) -> Verdict:
        frac = grounding_score(gi.text, gi.contexts)
        ok = frac >= threshold
        return Verdict(
            passed=ok,
            score=round(frac, 4),
            severity="none" if ok else severity,
            detail=(
                f"grounded {frac:.2f} >= {threshold:.2f}"
                if ok
                else f"grounded {frac:.2f} < {threshold:.2f} threshold — unsupported content"
            ),
            grader_id="grounding",
        )

    return _grounding
