"""The freeze-and-fingerprint discipline: a suite's hash is stable and any silent
edit changes it — the property a baseline comparison depends on."""
from gradecore import suite_hash


def test_hash_is_deterministic_and_12_hex():
    ids = ["t1:what colour is a clear sky", "t2:2+2"]
    h = suite_hash(ids)
    assert h == suite_hash(ids)                      # same input -> same fingerprint
    assert len(h) == 12 and all(c in "0123456789abcdef" for c in h)


def test_editing_a_task_changes_the_hash():
    base = ["t1:what colour is a clear sky", "t2:2+2"]
    edited = ["t1:what colour is the deep SEA", "t2:2+2"]
    assert suite_hash(base) != suite_hash(edited)    # a silent edit is detectable


def test_adding_a_task_changes_the_hash():
    base = ["t1:a", "t2:b"]
    assert suite_hash(base) != suite_hash(base + ["t3:c"])
