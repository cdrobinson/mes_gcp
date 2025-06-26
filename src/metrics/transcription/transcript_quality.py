"""Transcript quality metrics using token log probabilities"""

import logging
import math
from typing import Dict, Any
from ..base_metric import BaseMetric

logger = logging.getLogger(__name__)


class TranscriptQualityMetric(BaseMetric):
    """Metric for evaluating transcript quality using native LLM metrics"""
    
    def __init__(self):
        """Initialise Transcript Quality metric"""
        super().__init__("transcript_quality", "transcription")
    
    def compute(self, 
                response: str, 
                metadata: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute transcript quality metrics
        
        Args:
            response: The LLM transcription response
            metadata: Response metadata including log probabilities
            
        Returns:
            Dictionary of quality scores
        """
        scores = {}
        
        # Extract log probability metrics from metadata
        avg_log_prob = metadata.get("avg_logprobs")
        if avg_log_prob is not None:
            scores["transcript_avg_log_probability"] = avg_log_prob
            scores["transcript_confidence"] = math.exp(avg_log_prob)
        
        
        # Text format validation for transcription
        format_scores = self._validate_transcript_format(response)
        scores.update(format_scores)
        
        return scores
    
    def _validate_transcript_format(self, response: str) -> Dict[str, float]:
        """
        Validate transcript format according to requirements
        
        Args:
            response: The transcript text
            
        Returns:
            Dictionary of format validation scores
        """
        scores = {}
        
        if not response:
            return {"transcript_format_empty": 1.0}
        
        lines = response.strip().split('\n')
        
        # Check for timestamp format (MM:SS)
        timestamp_count = 0
        speaker_label_count = 0
        valid_format_lines = 0
        
        import re
        timestamp_pattern = r'^\d{1,2}:\d{2}\s+'
        speaker_patterns = ['Call_Agent', 'Customer']
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for timestamp at start
            if re.match(timestamp_pattern, line):
                timestamp_count += 1
                
                # Check for speaker labels
                for speaker in speaker_patterns:
                    if speaker in line:
                        speaker_label_count += 1
                        break
                
                # Check if line has basic required format
                if any(speaker in line for speaker in speaker_patterns):
                    valid_format_lines += 1
        
        total_lines = len([l for l in lines if l.strip()])
        
        if total_lines > 0:
            scores["transcript_timestamp_coverage"] = timestamp_count / total_lines
            scores["transcript_speaker_coverage"] = speaker_label_count / total_lines
            scores["transcript_format_compliance"] = valid_format_lines / total_lines
        else:
            scores["transcript_format_compliance"] = 0.0
        
        # Check for required speaker labels
        has_call_agent = 'Call_Agent' in response
        has_customer = 'Customer' in response
        scores["transcript_has_both_speakers"] = 1.0 if (has_call_agent and has_customer) else 0.0
        
        return scores
    
    def get_description(self) -> str:
        """Return description of the transcript quality metric"""
        return "Transcript quality evaluation using log probabilities and format validation"
