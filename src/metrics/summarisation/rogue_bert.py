"""Summarisation metrics using ROUGE and BERTScore."""

import logging
from typing import Dict, Any
from ..base_metric import BaseMetric
from rouge_score import rouge_scorer
from bert_score import score as bert_scorer

logger = logging.getLogger(__name__)

class RougeBertMetrics(BaseMetric):
    """Metric for evaluating summarisation quality using ROUGE and BERTScore."""

    def __init__(self):
        """Initialise ROUGE and BERTScore metrics."""
        super().__init__("summarisation_rouge_bert", "summarisation")
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def compute(self,
                response: str,
                metadata: Dict[str, Any],
                reference: str = None,
                **kwargs) -> Dict[str, float]:
        """
        Compute ROUGE and BERTScore metrics.

        Args:
            response: The generated summary.
            metadata: Response metadata.
            reference: The ground truth summary.

        Returns:
            Dictionary of ROUGE and BERTScore scores.
        """
        if not response or not reference:
            return {}

        scores = {}

        try:
            rouge_scores = self.rouge_scorer.score(reference, response)
            for key, value in rouge_scores.items():
                scores[f"summarisation_{key}_precision"] = value.precision
                scores[f"summarisation_{key}_recall"] = value.recall
                scores[f"summarisation_{key}_f1"] = value.fmeasure
        except Exception as e:
            logger.error(f"Error computing ROUGE scores: {e}")
            scores["summarisation_rouge_error"] = 1.0

        try:
            P, R, F1 = bert_scorer([response], [reference], lang="en", verbose=False)
            scores["summarisation_bert_precision"] = P.mean().item()
            scores["summarisation_bert_recall"] = R.mean().item()
            scores["summarisation_bert_f1"] = F1.mean().item()
        except Exception as e:
            logger.error(f"Error computing BERTScore: {e}")
            scores["summarisation_bert_error"] = 1.0

        return scores

    def get_description(self) -> str:
        """Return a description of what this metric measures."""
        return "Evaluates summarisation quality using ROUGE and BERTScore."