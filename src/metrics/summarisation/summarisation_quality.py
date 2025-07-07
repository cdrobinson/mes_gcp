"""Summarisation quality metrics using token log probabilities and format validation"""

import logging
import math
from typing import Dict, Any
from ..base_metric import BaseMetric

logger = logging.getLogger(__name__)


class SummarisationQualityMetric(BaseMetric):
    """Metric for evaluating summarisation quality using native LLM metrics and format validation"""
    
    # Expected section headers from the prompt
    EXPECTED_SECTIONS = {
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
    
    def __init__(self):
        """Initialise Summarisation Quality metric"""
        super().__init__("summarisation_quality", "summarisation")
    
    def compute(self, 
                response: str, 
                metadata: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute summarisation quality metrics
        
        Args:
            response: The LLM summarisation response
            metadata: Response metadata including log probabilities
            
        Returns:
            Dictionary of quality scores
        """
        scores = {}
        
        # Extract log probability metrics from metadata
        avg_log_prob = metadata.get("avg_logprobs")
        if avg_log_prob is not None:
            scores["summarisation_avg_log_probability"] = avg_log_prob
            scores["summarisation_confidence"] = math.exp(avg_log_prob)
        
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
            return {"summarisation_format_empty": 1.0}
        
        sections = self._parse_sections(response)
        
        present_sections = set(sections.keys())
        expected_sections = self.EXPECTED_SECTIONS
        
        if len(expected_sections) > 0:
            scores["summarisation_section_coverage"] = len(present_sections.intersection(expected_sections)) / len(expected_sections)
        else:
            scores["summarisation_section_coverage"] = 0.0
            
        # Check if required sections are present
        required_sections = {"Topic"}  # Topic is always required
        scores["summarisation_required_sections"] = 1.0 if required_sections.issubset(present_sections) else 0.0
        
        # Validate section length compliance (100-200 words)
        length_compliant_sections = 0
        total_sections = len(sections)
        
        for section_name, section_content in sections.items():
            word_count = len(section_content.split())
            if 100 <= word_count <= 200:
                length_compliant_sections += 1
        
        if total_sections > 0:
            scores["summarisation_section_length_compliance"] = length_compliant_sections / total_sections
        else:
            scores["summarisation_section_length_compliance"] = 0.0
        
        return scores
    
    def _parse_sections(self, response: str) -> Dict[str, str]:
        """
        Parse sections from the response text
        
        Args:
            response: The summarisation response text
            
        Returns:
            Dictionary mapping section names to their content
        """
        sections = {}
        lines = response.strip().split('\n')
        
        current_section = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if this line is a section header (ends with colon)
            if ':' in line and not line.startswith(' '):
                # Save previous section if exists
                if current_section and current_content:
                    sections[current_section] = ' '.join(current_content).strip()
                
                # Start new section
                section_name = line.split(':', 1)[0].strip()
                current_section = section_name
                # Get content after colon if present
                content_after_colon = line.split(':', 1)[1].strip()
                current_content = [content_after_colon] if content_after_colon else []
            else:
                # Add to current section content
                if current_section:
                    current_content.append(line)
        
        # Save last section
        if current_section and current_content:
            sections[current_section] = ' '.join(current_content).strip()
        
        return sections
    
    def get_description(self) -> str:
        """Return description of the summarisation quality metric"""
        return "summarisation quality evaluation using log probabilities and format validation for structured insurance call summaries"
