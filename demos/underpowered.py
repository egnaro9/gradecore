"""Ask an 8-task suite to pick a winner between two prompts, and watch it refuse.

Both configs are graded on the same tasks, so the comparison is paired: only the
tasks where they *disagree* carry information. Four disagreements cannot reach
p<0.05 even if they all break the same way, so the honest output is "this suite
cannot decide" — a statement about the instrument, not about the configs.

Everything here is arithmetic on the two dicts below. No model, no network.
"""

import textwrap

from demos._ansi import accent, dim, fail, muted, ok, text
from gradecore import paired_compare, paired_verdict, tasks_needed

# 1 = task passed, 0 = task failed.
prompt_v1 = {"t1": 1, "t2": 1, "t3": 0, "t4": 1, "t5": 0, "t6": 1, "t7": 1, "t8": 0}
prompt_v2 = {"t1": 1, "t2": 0, "t3": 1, "t4": 1, "t5": 1, "t6": 1, "t7": 1, "t8": 1}

r = paired_compare(prompt_v1, prompt_v2)

print(muted("prompt-v1  ") + "".join(ok("■") if v else dim("□") for v in prompt_v1.values()))
print(muted("prompt-v2  ") + "".join(ok("■") if v else dim("□") for v in prompt_v2.values()))
print(
    muted("           ")
    + "".join(
        accent("↑") if a != b else dim("·")
        for a, b in zip(prompt_v1.values(), prompt_v2.values())
    )
    + muted("   ← only these carry information")
)
print()

for line in textwrap.wrap(paired_verdict(r, "prompt-v1", "prompt-v2"), width=78):
    print(text(line, bold=True))
print()


def row(label, value, paint=text):
    print(f"  {muted(label.ljust(16))}{paint(str(value))}")


row("shared tasks", r.shared)
row("informative", f"{r.informative}   " + dim("(the rest tie and separate nothing)"))
row("p", f"{r.p:.4f}", accent)
row("decisive", r.decisive, ok if r.decisive else fail)
row("underpowered", r.underpowered, fail if r.underpowered else ok)
print()
print(
    "  "
    + dim("a suite needs ")
    + accent(f">= {tasks_needed()}")
    + dim(" informative tasks before ANY split clears p<0.05")
)
