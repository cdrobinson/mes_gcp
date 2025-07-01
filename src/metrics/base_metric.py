"""Base metric class for all evaluation metrics"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd


class BaseMetric(ABC):
    """Abstract base class for all evaluation metrics"""
    
    def __init__(self, name: str, use_case: str):
        """
        Initialise the metric
        
        Args:
            name: Name of the metric
            use_case: The use case this metric applies to (transcription, summarisation, etc.)
        """
        self.name = name
        self.use_case = use_case
    
    @abstractmethod
    def compute(self, 
                response: str, 
                metadata: Dict[str, Any], 
                reference: Optional[str] = None,
                audio_path: Optional[str] = None) -> Dict[str, float]:
        """
        Compute the metric for a given response
        
        Args:
            response: The LLM response text
            metadata: Response metadata including tokens, latency, etc.
            reference: Reference/ground truth text (if available)
            audio_path: Path to the original audio file (if needed)
            
        Returns:
            Dictionary of metric name to score mappings
        """
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Return a description of what this metric measures"""
        pass
    
    def is_applicable(self, use_case: str) -> bool:
        """Check if this metric is applicable to the given use case"""
        return self.use_case == use_case or self.use_case == "all"
    
    def supports_batch_evaluation(self) -> bool:
        """Check if this metric supports batch evaluation"""
        return False
    
    def batch_compute(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute metrics for a batch of responses
        
        Args:
            results_df: DataFrame containing experiment results
            
        Returns:
            DataFrame with additional metric columns
        """
        # Default implementation: fall back to individual compute calls
        if not self.supports_batch_evaluation():
            import warnings
            warnings.warn(f"Metric {self.name} does not support batch evaluation, falling back to individual calls")
            
            # Apply compute to each row individually
            metric_columns = []
            for idx, row in results_df.iterrows():
                if hasattr(row, 'response_text') and hasattr(row, 'metadata'):
                    scores = self.compute(
                        response=row.get('response_text', ''),
                        metadata=row.get('metadata', {})
                    )
                    
                    # Add scores to the row
                    for metric_name, score in scores.items():
                        if metric_name not in metric_columns:
                            metric_columns.append(metric_name)
                        results_df.loc[idx, metric_name] = score
            
            return results_df
        else:
            raise NotImplementedError("Batch evaluation not implemented for this metric")
