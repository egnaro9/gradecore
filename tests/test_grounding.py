"""The grounding grader — the retrieval/RAG lens, extracted from rag-eval-lab.

The last test is the load-bearing one: it recomputes rag-eval-lab's ORIGINAL
faithfulness inline and asserts the extracted engine reproduces it exactly, so
"one engine, three lenses" is a checkable claim, not a hopeful one.
"""
import re

from gradecore import GradeInput, grounding, grounding_score
from gradecore.grounding import content_tokens, tokenize


def test_tokenizer_lowercases_and_splits_on_nonalnum():
    assert tokenize("Cats, Dogs & 3 Birds!") == ["cats", "dogs", "3", "birds"]


def test_content_tokens_drop_stopwords():
    assert content_tokens("the cat is on a mat") == ["cat", "mat"]


def test_grounding_score_full_partial_and_empty():
    assert grounding_score("cats are mammals", ["cats are mammals"]) == 1.0   # all supported
    assert grounding_score("cats are birds", ["cats are mammals"]) == 0.5     # birds unsupported
    assert grounding_score("zebras gallop", ["cats are mammals"]) == 0.0      # nothing supported
    assert grounding_score("the cat", ["cat"]) == 1.0                         # stopword ignored
    assert grounding_score("", ["anything"]) == 0.0                           # empty -> floor
    assert grounding_score("only the a an", ["x"]) == 0.0                     # all-stopword -> floor


def test_grounding_grader_emits_a_scored_verdict():
    g = grounding(threshold=0.6)
    ok = g(GradeInput(text="cats are mammals", contexts=["cats are mammals"]))
    assert ok.passed and ok.score == 1.0 and ok.severity == "none" and ok.grader_id == "grounding"

    bad = g(GradeInput(text="cats are birds", contexts=["cats are mammals"]))
    assert bad.passed is False and bad.score == 0.5 and bad.severity == "med"
    assert "unsupported" in bad.detail


def test_matches_rag_eval_lab_original_faithfulness_exactly():
    """Differential guard against drift: the extracted logic == the original."""
    _RE = re.compile(r"[a-z0-9]+")
    _STOP = {"the", "a", "an", "and", "or", "but", "is", "are", "was", "were",
             "be", "to", "of", "in", "on", "for", "with", "as", "by", "at", "it",
             "its", "this", "that", "these", "those", "from", "into", "than",
             "then", "so", "you", "your", "i", "we", "they", "he", "she", "them",
             "can", "will"}

    def reference_faithfulness(ans, ctxs):
        at = [t for t in _RE.findall(ans.lower()) if t not in _STOP]
        if not at:
            return 0.0
        sup = set()
        for c in ctxs:
            sup |= {t for t in _RE.findall(c.lower()) if t not in _STOP}
        return sum(1 for t in at if t in sup) / len(at)

    cases = [
        ("The retriever found that aspirin reduces fever in adults.",
         ["Aspirin reduces fever and inflammation.", "It is used by adults."]),
        ("Vaccines cause autism.",
         ["Multiple large studies found no link between vaccines and autism."]),
        ("", ["some context"]),
        ("only the a an from", ["the a an"]),
        ("MODEL-drift tracks 16 LLMs.", ["model-drift tracks 16 llms across 5 labs"]),
    ]
    for ans, ctxs in cases:
        assert grounding_score(ans, ctxs) == reference_faithfulness(ans, ctxs)
