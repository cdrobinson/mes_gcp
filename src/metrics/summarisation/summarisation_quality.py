"""Summarisation quality metrics using token log probabilities and format validation"""

import logging
import math
import re
from typing import Dict, Any
from ..base_metric import BaseMetric

logger = logging.getLogger(__name__)


class SummarisationQualityMetric(BaseMetric):
    """Metric for evaluating summarisation quality using native LLM metrics and format validation"""
    
    EXPECTED_SECTIONS = {
        "Speaker",
        "Topic",
        "Introduction", 
        "New Claim",
        "Follow-up Claim",
        "Policy and Coverage Discussion or Queries",
        "Policy Renewals and Cancellation",
        "Customer Feedback",
        "Financials & Payments",
        "Key Actions and Next Steps"
    }
    
    MANDATORY_SECTIONS = {
        "Speaker",
        "Topic",
        "Introduction",
        "Key Actions and Next Steps"
    }

    def __init__(self):
        """Initialise Summarisation Quality metric"""
        super().__init__("summarisation_quality", "summarisation")
    
    def compute(self, 
                response: str, 
                metadata: Dict[str, Any],
                **kwargs) -> Dict[str, float]:
        """
        Compute summarisation quality metrics
        
        Args:
            response: The LLM summarisation response
            metadata: Response metadata including log probabilities
            
        Returns:
            Dictionary of quality scores
        """
        scores = {}
        
        # Extract log probability metrics from metadata if they exist
        avg_log_prob = metadata.get("avg_logprobs")
        if avg_log_prob is not None:
            scores["summarisation_avg_log_probability"] = avg_log_prob
            # Use max to avoid math domain error for negative log_prob
            scores["summarisation_confidence"] = math.exp(min(avg_log_prob, 0))
        
        # Format validation for summarisation
        format_scores = self._validate_summarisation_format(response)
        scores.update(format_scores)
        
        return scores
    
    def _validate_summarisation_format(self, response: str) -> Dict[str, float]:
        """
        Validate summarisation format according to prompt requirements
        
        Args:
            response: The summarisation text
            
        Returns:
            Dictionary of format validation scores
        """
        scores = {}
        
        if not response:
            return {"summarisation_format_empty": 1.0, "summarisation_section_coverage": 0.0, "summarisation_required_sections": 0.0}
        
        sections = self._parse_sections(response)
        present_sections = set(sections.keys())
        
        # Calculate section coverage based on all possible sections
        if self.EXPECTED_SECTIONS:
            scores["summarisation_section_coverage"] = len(present_sections.intersection(self.EXPECTED_SECTIONS)) / len(self.EXPECTED_SECTIONS)
        else:
            scores["summarisation_section_coverage"] = 0.0
            
        # Check if all mandatory sections are present
        scores["summarisation_required_sections"] = 1.0 if self.MANDATORY_SECTIONS.issubset(present_sections) else 0.0
        
        # Validate section length compliance (updated to 50-60 words from the prompt)
        length_compliant_sections = 0
        total_sections = len(sections)
        
        for section_content in sections.values():
            word_count = len(section_content.split())
            if 50 <= word_count <= 60:
                length_compliant_sections += 1
        
        if total_sections > 0:
            scores["summarisation_section_length_compliance"] = length_compliant_sections / total_sections
        else:
            scores["summarisation_section_length_compliance"] = 0.0
        
        return scores
    
    def _parse_sections(self, response: str) -> Dict[str, str]:
        """
        Parse sections from the response text using regex to handle the '**Section:**' format.
        
        Args:
            response: The summarisation response text
            
        Returns:
            Dictionary mapping section names to their content
        """
        sections = {}
        # This regex looks for a bolded title (e.g., **Topic:**), captures the title,
        # and then captures the content until the next bolded title or the end of the string.
        pattern = r'\*\*(.*?):\*\*\s*(.*?)(?=\n\n\*\*|$)'
        matches = re.findall(pattern, response, re.DOTALL)

        for match in matches:
            section_name = match[0].strip()
            section_content = match[1].strip()
            sections[section_name] = section_content
            
        return sections
    
    def get_description(self) -> str:
        """Return description of the summarisation quality metric"""
        return "Summarisation quality evaluation using log probabilities and format validation for structured insurance call summaries"