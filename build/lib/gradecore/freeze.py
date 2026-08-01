"""Freeze-and-fingerprint discipline.

Generalized from model-drift's `suite_hash()` (`suite.py:209`): a run records a
short, stable fingerprint of the exact suite it answered, so a silently-edited
suite is detectable and a leaderboard/drift comparison across two *different*
suites is caught rather than being quietly meaningless. A suite is only a
baseline if you can prove two runs answered the same questions.
"""
from __future__ import annotations

import hashlib
from typing import Iterable

# Bump when a suite's task set changes on purpose (mirrors model-drift's
# SUITE_VERSION freeze). A version says "this changed deliberately"; the hash
# proves nothing changed by accident.
SCHEMA_VERSION = "gradecore-v1"


def suite_hash(identities: Iterable[str]) -> str:
    """A sha256[:12] over each task's identity string.

    Same construction as model-drift's `"|".join(f"{id}:{prompt}")` fingerprint,
    but generic: pass whatever you want covered. To also catch an edited
    answer-key (which model-drift's id:prompt hash misses), fold the grader id /
    expected value into each identity string.
    """
    blob = "|".join(identities)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]
