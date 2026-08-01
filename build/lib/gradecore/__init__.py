"""gradecore — one deterministic, no-LLM-judge grading engine.

Shared by model-drift (longitudinal monitoring) and the crash-test platform
(on-demand adversarial testing). Every grade is a pure predicate over a string,
so it reproduces exactly — the property both a drift board and a vulnerability
score depend on.
"""
from .adversarial import (
    ABSTENTION_MARKERS,
    REFUSAL_MARKERS,
    injection_resistance,
    must_abstain,
    must_comply,
    must_refuse,
    tool_misuse,
    valid_json,
)
from .freeze import SCHEMA_VERSION, suite_hash
from .graders import (
    last_line,
    bool_grader,
    contains,
    exact,
    exact_cs,
    number,
    one_of,
    regex,
)
from .grounding import (
    FAITHFULNESS_THRESHOLD,
    STOPWORDS,
    grounding,
    grounding_score,
    tokenize,
)
from .paired import (
    DEFAULT_ALPHA,
    PairedResult,
    TaskDelta,
    DEFAULT_RATE_MARGIN,
    noise_floor,
    paired_compare,
    paired_verdict,
    repeated_compare,
    repeated_verdict,
    sign_test_p,
    tasks_needed,
)
from .trajectory import trajectory, trajectory_score
from .verdict import SEVERITIES, GradeInput, Grader, Verdict, check_severity

__version__ = "0.10.0"

__all__ = [
    "GradeInput", "Verdict", "Grader", "SEVERITIES", "check_severity",
    "exact", "contains", "regex", "exact_cs", "one_of", "number", "bool_grader",
    "suite_hash", "SCHEMA_VERSION",
    # adversarial
    "must_refuse", "must_comply", "must_abstain", "injection_resistance",
    "valid_json", "tool_misuse", "REFUSAL_MARKERS", "ABSTENTION_MARKERS",
    # grounding (retrieval/RAG lens)
    "grounding", "grounding_score", "tokenize", "STOPWORDS", "FAITHFULNESS_THRESHOLD",
    # trajectory (agent action-graph lens)
    "trajectory", "trajectory_score",
    # paired comparison between two scored runs
    "paired_compare", "paired_verdict", "sign_test_p", "tasks_needed", "last_line",
    "repeated_compare", "repeated_verdict", "noise_floor", "DEFAULT_RATE_MARGIN",
    "PairedResult", "TaskDelta", "DEFAULT_ALPHA",
]
