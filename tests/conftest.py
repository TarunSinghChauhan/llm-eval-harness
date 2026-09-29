"""
Stubs heavy ML dependencies (rouge_score, bert_score — the latter pulls in
torch, which we deliberately avoid installing here) so that any module
importing src.evaluation.scorers.metrics can be collected and tested without
a real install. Tests that need specific scorer behavior patch the relevant
attribute directly (e.g. scorer._rouge, or metrics.bert_score_fn) rather than
relying on these stubs to compute anything meaningful.
"""
import sys
from unittest.mock import MagicMock

if "rouge_score" not in sys.modules:
    _rouge_score_pkg = MagicMock()
    _rouge_scorer_submodule = MagicMock()
    _rouge_score_pkg.rouge_scorer = _rouge_scorer_submodule
    sys.modules["rouge_score"] = _rouge_score_pkg
    sys.modules["rouge_score.rouge_scorer"] = _rouge_scorer_submodule

if "bert_score" not in sys.modules:
    _bert_score_pkg = MagicMock()
    _bert_score_pkg.score = MagicMock()
    sys.modules["bert_score"] = _bert_score_pkg
