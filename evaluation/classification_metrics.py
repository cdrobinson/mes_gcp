from abc import abstractmethod
from typing import Dict, Any, Optional, List
from .base_metric import BaseMetric

class AbstractClassificationMetric(BaseMetric):
    """Abstract base class for classification metrics."""
    @abstractmethod
    def calculate(self, # type: ignore
                  llm_response: str, 
                  confidence: Optional[float] = None,
                  allowed_labels: Optional[List[str]] = None,
                  **kwargs) -> Dict[str, Any]:
        """
        Calculates classification-specific metrics.
        Args:
            llm_response: The predicted label string from the LLM.
            confidence: Optional confidence score for the prediction.
            allowed_labels: Optional list of valid labels for validation.
            **kwargs: Additional arguments specific to the metric.
        Returns:
            A dictionary of calculated metrics.
        """
        pass

class LabelProperties(AbstractClassificationMetric):
    """Calculates properties of a predicted label."""
    def calculate(self, 
                  llm_response: str, 
                  confidence: Optional[float] = None,
                  allowed_labels: Optional[List[str]] = None,
                  **kwargs) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}
        predicted_label = llm_response # llm_response is the predicted label

        if not predicted_label or not predicted_label.strip():
            metrics["classification_label_provided"] = False
            metrics["classification_label_validity"] = "not_provided"
        else:
            metrics["classification_label_provided"] = True
            if allowed_labels:
                if predicted_label in allowed_labels:
                    metrics["classification_label_validity"] = "valid"
                else:
                    metrics["classification_label_validity"] = "invalid_not_in_allowed_set"
            else:
                metrics["classification_label_validity"] = "valid_no_allowed_set_defined"

        metrics["classification_confidence"] = confidence # Will be None if not provided
        
        return metrics

class ResponseLengthChars(AbstractClassificationMetric):
    """Measures the character length of the LLM response for classification."""
    def calculate(self, 
                  llm_response: str, 
                  **kwargs) -> Dict[str, Any]:
        return {"classification_response_char_length": len(llm_response)}

class PotentialExplanationPresence(AbstractClassificationMetric):
    """
    Checks if the classification response potentially contains an explanation
    beyond just a simple label, based on word count.
    """
    def calculate(self, 
                  llm_response: str, 
                  explanation_threshold_words: int = 5,
                  **kwargs) -> Dict[str, Any]:
        if not llm_response or not llm_response.strip():
            return {
                "classification_explanation_present_heuristic": False,
                "classification_response_word_count": 0
            }

        word_count = len(llm_response.split())
        explanation_present = word_count > explanation_threshold_words
        
        return {
            "classification_explanation_present_heuristic": explanation_present,
            "classification_response_word_count": word_count
        }
