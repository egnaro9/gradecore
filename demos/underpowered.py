"""Ask an 8-task suite to pick a winner between two prompts, and watch it refuse.

Both configs are graded on the same tasks, so the comparison is paired: only the
tasks where they *disagree* carry information. Four disagreements cannot reach
p<0.05 even if they all break the same way, so the honest output is "this suite
cannot decide" — a statement about the instrument, not about the configs.

Everything here is arithmetic on the two dicts below. No model, no network.
"""

import textwrap

from demos._ansi import amber, dim, faint, fg, red, teal
from gradecore import paired_compare, paired_verdict, tasks_needed

# 1 = task passed, 0 = task failed.
prompt_v1 = {"t1": 1, "t2": 1, "t3": 0, "t4": 1, "t5": 0, "t6": 1, "t7": 1, "t8": 0}
prompt_v2 = {"t1": 1, "t2": 0, "t3": 1, "t4": 1, "t5": 1, "t6": 1, "t7": 1, "t8": 1}

r = paired_compare(prompt_v1, prompt_v2)

print(dim("prompt-v1  ") + "".join(teal("■") if v else faint("□") for v in prompt_v1.values()))
print(dim("prompt-v2  ") + "".join(teal("■") if v else faint("□") for v in prompt_v2.values()))
print(
    dim("           ")
    + "".join(
        amber("↑") if a != b else faint("·")
        for a, b in zip(prompt_v1.values(), prompt_v2.values())
    )
    + dim("   ← only these carry information")
)
print()

for line in textwrap.wrap(paired_verdict(r, "prompt-v1", "prompt-v2"), width=78):
    print(fg(line, bold=True))
print()


def row(label, value, colour=fg):
    print(f"  {dim(label.ljust(16))}{colour(str(value))}")


row("shared tasks", r.shared)
row("informative", f"{r.informative}   " + faint("(the rest tie and separate nothing)"))
row("p", f"{r.p:.4f}", amber)
row("decisive", r.decisive, red if not r.decisive else teal)
row("underpowered", r.underpowered, red if r.underpowered else teal)
print()
print(
    "  "
    + faint("a suite needs ")
    + amber(f">= {tasks_needed()}")
    + faint(" informative tasks before ANY split clears p<0.05")
)
