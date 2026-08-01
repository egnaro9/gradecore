"""Ask an 8-task suite to pick a winner between two prompts, and watch it refuse.

Both configs are graded on the same tasks, so the comparison is paired: only the
tasks where they *disagree* carry information. Four disagreements cannot reach
p<0.05 even if they all break the same way, so the honest output is "this suite
cannot decide" — a statement about the instrument, not about the configs.

Everything here is arithmetic on the two dicts below. No model, no network.
"""

import textwrap

from gradecore import paired_compare, paired_verdict, tasks_needed

# 1 = task passed, 0 = task failed.
prompt_v1 = {"t1": 1, "t2": 1, "t3": 0, "t4": 1, "t5": 0, "t6": 1, "t7": 1, "t8": 0}
prompt_v2 = {"t1": 1, "t2": 0, "t3": 1, "t4": 1, "t5": 1, "t6": 1, "t7": 1, "t8": 1}

r = paired_compare(prompt_v1, prompt_v2)

print(textwrap.fill(paired_verdict(r, "prompt-v1", "prompt-v2"), width=78))
print()
print(f"  shared tasks      {r.shared}")
print(f"  informative       {r.informative}   (the rest tie and cannot separate anything)")
print(f"  p                 {r.p:.4f}")
print(f"  decisive          {r.decisive}")
print(f"  underpowered      {r.underpowered}")
print()
print(f"  a suite needs >= {tasks_needed()} informative tasks before ANY split clears p<0.05")
