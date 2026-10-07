"""Freeze-and-fingerprint discipline.

Generalized from model-drift's `suite_hash()` (`suite.py:209`): a run records a
short, stable fingerprint of the exact suite it answered, so a silently-edited
suite is detectable and a leaderboard/drift comparison across two *different*
suites is caught rather than being quietly meaningless. A suite is only a
baseline if you can prove two runs answered the same questions.

Two constructions live here and the difference is a contract, not a preference.
`suite_hash` is frozen for compatibility with model-drift's independent copy of
the same arithmetic; `suite_hash_injective` is the one to reach for in new code.
"""
from __future__ import annotations

import hashlib
from typing import Iterable

# Bump when a suite's task set changes on purpose (mirrors model-drift's
# SUITE_VERSION freeze). A version says "this changed deliberately"; the hash
# proves nothing changed by accident.
#
# It feeds `suite_hash_injective` and NOT `suite_hash`. The legacy construction
# predates it and is pinned by consumers, so folding it in there now would change
# every existing fingerprint; see the known limit below.
SCHEMA_VERSION = "gradecore-v1"


def suite_hash(identities: Iterable[str]) -> str:
    """A sha256[:12] over each task's identity string, joined with "|".

    Same construction as model-drift's `"|".join(f"{id}:{prompt}")` fingerprint,
    but generic: pass whatever you want covered. To also catch an edited
    answer-key (which model-drift's id:prompt hash misses), fold the grader id /
    expected value into each identity string.

    KNOWN LIMIT, stated rather than fixed: the join is NOT injective, so this is
    blind to where one identity ends and the next begins. Two different suites
    collide whenever a "|" appears inside an identity:

        suite_hash(["a|b", "c"]) == suite_hash(["a", "b|c"])   # both a52dd81bfd5e

    The advice above is what walks into it, because folding a grader id or an
    expected value into an identity is exactly when a delimiter shows up in the
    content. Prefer `suite_hash_injective` in new code, and if you must stay on
    this construction, pick a separator that cannot occur in your identities.

    It is a limit rather than a bug fix because this arithmetic is a COMPATIBILITY
    CONTRACT, not an implementation detail. model-drift carries its own
    independent copy of it, crashkit asserts the two agree
    (`tests/test_runner.py::test_the_lift_preserves_every_task_id_and_prompt`),
    the resulting digest is frozen into a VAC attestation (`crashkit/vac/vac.json`)
    and published as a reproducible figure. Measured, not assumed: changing the
    construction here turns that cross-implementation test red and makes the
    published digest unreproducible. Any escape, length prefix or nested hash has
    the same effect, so the collision cannot be closed in place.
    """
    blob = "|".join(identities)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def suite_hash_injective(identities: Iterable[str]) -> str:
    """A sha256[:12] fingerprint with no delimiter to collide on.

    Each identity is hashed to a fixed 32 bytes first, and the fingerprint is the
    hash of those digests in order. Because every identity occupies exactly 32
    bytes, no identity's CONTENT can be mistaken for a boundary, so distinct
    suites give distinct fingerprints and the colliding pair above separates.

    `SCHEMA_VERSION` is folded in, which is what that constant is for: bumping it
    moves every fingerprint deliberately, so a suite whose task set changed on
    purpose cannot be confused with one that drifted. Nothing is pinned to this
    function yet, so it can carry the version the frozen one cannot.

    Prefer this for new suites. It is NOT interchangeable with `suite_hash` —
    the two return different values for the same input, by design — so a stored
    baseline must record which one produced it.
    """
    inner = hashlib.sha256(SCHEMA_VERSION.encode()).digest()
    for identity in identities:
        inner += hashlib.sha256(identity.encode()).digest()
    return hashlib.sha256(inner).hexdigest()[:12]
