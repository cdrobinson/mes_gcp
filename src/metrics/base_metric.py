"""Base metric class for all evaluation metrics"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import pandas as pd


class BaseMetric(ABC):
    """Abstract base class for all evaluation metrics"""
    
    def __init__(self, name: str, use_case: str):
        """
        Initialize the metric
        
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
    
    def get_aggregated_score(self, scores: List[float]) -> Dict[str, float]:
        """
        Compute aggregated statistics for a list of scores
        
        Args:
            scores: List of individual scores
            
        Returns:
            Dictionary with mean, std, min, max, etc.
        """
        if not scores:
            return {}
        
        import numpy as np
        
        return {
            f"{self.name}_mean": np.mean(scores),
            f"{self.name}_std": np.std(scores),
            f"{self.name}_min": np.min(scores),
            f"{self.name}_max": np.max(scores),
            f"{self.name}_median": np.median(scores)
        }


class CompositeMetric(BaseMetric):
    """A metric that combines multiple sub-metrics"""
    
    def __init__(self, name: str, use_case: str, sub_metrics: List[BaseMetric]):
        super().__init__(name, use_case)
        self.sub_metrics = sub_metrics
    
    def compute(self, 
                response: str, 
                metadata: Dict[str, Any], 
                reference: Optional[str] = None,
                audio_path: Optional[str] = None) -> Dict[str, float]:
        """Compute all sub-metrics and return combined results"""
        results = {}
        
        for metric in self.sub_metrics:
            if metric.is_applicable(self.use_case):
                metric_results = metric.compute(response, metadata, reference, audio_path)
                results.update(metric_results)
        
        return results
    
    def get_description(self) -> str:
        descriptions = [metric.get_description() for metric in self.sub_metrics]
        return f"Composite metric combining: {'; '.join(descriptions)}"
