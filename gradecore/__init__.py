"""gradecore — one deterministic, no-LLM-judge grading engine.

Shared by model-drift (longitudinal monitoring) and the crash-test platform
(on-demand adversarial testing). Every grade is a pure predicate over a string,
so it reproduces exactly — the property both a drift board and a vulnerability
score depend on.
"""
from .freeze import SCHEMA_VERSION, suite_hash
from .graders import (
    bool_grader,
    contains,
    exact,
    exact_cs,
    number,
    one_of,
    regex,
)
from .verdict import SEVERITIES, GradeInput, Grader, Verdict, check_severity

__version__ = "0.1.0"

__all__ = [
    "GradeInput", "Verdict", "Grader", "SEVERITIES", "check_severity",
    "exact", "contains", "regex", "exact_cs", "one_of", "number", "bool_grader",
    "suite_hash", "SCHEMA_VERSION",
]
