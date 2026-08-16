# gradecore

**One deterministic, no-LLM-judge grading engine**: shared by
[model-drift](https://github.com/egnaro9/model-drift) (longitudinal monitoring)
and the crash-test platform (on-demand adversarial testing). Zero dependencies.

Every grade is a pure predicate over a string, so it reproduces exactly. The
property a drift board and a vulnerability score both depend on. No second model
grades the first; there is nothing here you can't rerun and get the same answer.

```python
from gradecore import exact, contains, number, GradeInput, suite_hash

exact("blue")(GradeInput(text="  BLUE "))       # -> Verdict(passed=True, score=1.0, severity="none", …)
contains("signal", "process")(GradeInput(text="signal a process"))   # both needles required
number(3.14, tol=0.01)(GradeInput(text="about 3.141"))               # extracts + tolerance-compares
```

<img src="https://raw.githubusercontent.com/egnaro9/gradecore/main/docs/demo.gif" alt="A paired comparison declining to name a winner because only four tasks separate the two variants" width="100%">

*Eight tasks, two prompt variants, four disagreements. Four cannot clear p<0.05 even under a clean sweep, so the suite reports that it cannot decide rather than naming a winner: `python3 -m demos.underpowered`. [Play it as a terminal session](https://asciinema.org/a/GnfWEFk2DzH8QJyn). The text is selectable.*

## The one signature

model-drift's graders are `Callable[[str], bool]`; rag-eval-lab's return floats
gated by thresholds. gradecore reconciles both under **`GradeInput -> Verdict`**,
which generalizes boolean into a severity-scored verdict without losing the
boolean case:

```python
Verdict(passed: bool, score: float, severity: str, detail: str, grader_id: str)
#        pass/fail    0.0..1.0       none|low|med|high|critical
```

## What's here (v0.1)

- **Scalar/text graders** lifted from model-drift's suite combinators -
  `exact`, `contains`, `regex`, `exact_cs`, `one_of`, `number`.
- **`bool_grader(fn, grader_id)`**: lifts any existing `Callable[[str], bool]`
  unchanged, so model-drift's *frozen* SUITE runs through gradecore without being
  rewritten. (Verified: gradecore's `suite_hash` over that SUITE is byte-identical
  to model-drift's own fingerprint.)
- **`suite_hash(identities)`**: the freeze-and-fingerprint discipline, so a
  silently-edited suite is detectable and two runs are only comparable if they
  answered the same questions.

- **Retrieval/grounding family** (`grounding.py`). With the empty-gold /
  empty-answer gotchas corrected.
- **Adversarial graders** (`adversarial.py`). Injection-resistance, tool-misuse,
  spec/format-violation, refusal-calibration. Shipped with crash-test Phase 1.
- **Trajectory scoring** (`trajectory.py`) and **paired comparison stats**
  (`paired.py`). The sign-test/permutation machinery that refuses underpowered
  verdicts.

```bash
pip install -e ".[dev]" && pytest -q        # 84 tests, zero dependencies
```

MIT · by [Erik Hill](https://egnaro9.github.io)
