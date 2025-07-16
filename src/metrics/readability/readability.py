"""Readability metrics using the textstat library."""

import logging
from typing import Dict, Any
from ..base_metric import BaseMetric
import textstat

logger = logging.getLogger(__name__)

class ReadabilityMetrics(BaseMetric):
    """Metric for evaluating text readability."""

    def __init__(self):
        """Initialise Readability metrics."""
        super().__init__("readability", "all")

    def compute(self,
                response: str,
                metadata: Dict[str, Any],
                **kwargs) -> Dict[str, float]:
        """
        Compute readability metrics.

        Args:
            response: The LLM response text.
            metadata: Response metadata.

        Returns:
            Dictionary of readability scores.
        """
        if not response:
            return {}

        try:
            return {
                "readability_flesch_kincaid": textstat.flesch_kincaid_grade(response),
                "readability_flesch_reading_ease": textstat.flesch_reading_ease(response),
                "readability_gunning_fog": textstat.gunning_fog(response),
                "readability_smog_index": textstat.smog_index(response),
                "readability_coleman_liau_index": textstat.coleman_liau_index(response),
                "readability_ari": textstat.automated_readability_index(response),
            }
        except Exception as e:
            logger.error(f"Error computing readability metrics: {e}")
            return {"readability_computation_error": 1.0}

    def get_description(self) -> str:
        """Return a description of what this metric measures."""
        return "Evaluates the readability of the generated text using a suite of standard metrics."