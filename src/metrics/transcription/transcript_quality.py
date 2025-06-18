"""Transcript quality metrics using token log probabilities"""

import logging
import numpy as np
from typing import Dict, Any, Optional, List
from ..base_metric import BaseMetric

logger = logging.getLogger(__name__)


class TranscriptQualityMetric(BaseMetric):
    """Metric for evaluating transcript quality using native LLM metrics"""
    
    def __init__(self):
        """Initialize Transcript Quality metric"""
        super().__init__("transcript_quality", "transcription")
    
    def compute(self, 
                response: str, 
                metadata: Dict[str, Any], 
                reference: Optional[str] = None,
                audio_path: Optional[str] = None) -> Dict[str, float]:
        """
        Compute transcript quality metrics
        
        Args:
            response: The LLM transcription response
            metadata: Response metadata including log probabilities
            reference: Reference transcript (if available)
            audio_path: Audio path (not used for this metric)
            
        Returns:
            Dictionary of quality scores
        """
        scores = {}
        
        # Extract log probability metrics from metadata
        avg_log_prob = metadata.get("avg_log_probability")
        if avg_log_prob is not None:
            scores["transcript_avg_log_probability"] = avg_log_prob
            
            # Convert log probability to confidence score (0-1)
            # Log probabilities are typically negative, closer to 0 means higher confidence
            confidence = self._log_prob_to_confidence(avg_log_prob)
            scores["transcript_confidence"] = confidence
        
        # Token count metrics
        token_count = metadata.get("token_count", 0)
        if token_count > 0:
            scores["transcript_token_count"] = token_count
            
            # Words per token ratio (estimate)
            word_count = len(response.split()) if response else 0
            if word_count > 0:
                scores["transcript_words_per_token"] = word_count / token_count
        
        # Text format validation for transcription
        format_scores = self._validate_transcript_format(response)
        scores.update(format_scores)
        
        # If reference transcript is available, compute comparison metrics
        if reference:
            comparison_scores = self._compare_with_reference(response, reference)
            scores.update(comparison_scores)
        
        # Basic text quality metrics
        quality_scores = self._compute_text_quality(response)
        scores.update(quality_scores)
        
        return scores
    
    def _log_prob_to_confidence(self, log_prob: float) -> float:
        """
        Convert log probability to confidence score
        
        Args:
            log_prob: Average log probability
            
        Returns:
            Confidence score between 0 and 1
        """
        # Empirical mapping based on typical log probability ranges
        # Log probabilities typically range from -10 to 0
        # Map to confidence scale where:
        # -0.5 to 0 -> high confidence (0.8-1.0)
        # -2.0 to -0.5 -> medium confidence (0.5-0.8)
        # -10.0 to -2.0 -> low confidence (0.0-0.5)
        
        if log_prob >= -0.5:
            return 0.8 + (log_prob + 0.5) * 0.4  # Maps [-0.5, 0] to [0.8, 1.0]
        elif log_prob >= -2.0:
            return 0.5 + (log_prob + 2.0) * 0.2  # Maps [-2.0, -0.5] to [0.5, 0.8]
        else:
            return max(0.0, 0.5 + (log_prob + 2.0) * 0.0625)  # Maps [-10.0, -2.0] to [0.0, 0.5]
    
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
    
    def _compare_with_reference(self, response: str, reference: str) -> Dict[str, float]:
        """
        Compare transcript with reference using simple metrics
        
        Args:
            response: Generated transcript
            reference: Reference transcript
            
        Returns:
            Dictionary of comparison scores
        """
        scores = {}
        
        # Basic text similarity
        response_words = set(response.lower().split())
        reference_words = set(reference.lower().split())
        
        if reference_words:
            # Word overlap
            overlap = len(response_words.intersection(reference_words))
            scores["transcript_word_overlap"] = overlap / len(reference_words)
            
            # Length ratio
            scores["transcript_length_ratio"] = len(response.split()) / len(reference.split())
        
        # Character-level similarity (simple)
        response_clean = ''.join(response.lower().split())
        reference_clean = ''.join(reference.lower().split())
        
        if reference_clean:
            # Simple character overlap
            common_chars = sum(1 for c in response_clean if c in reference_clean)
            scores["transcript_char_similarity"] = common_chars / len(reference_clean)
        
        return scores
    
    def _compute_text_quality(self, response: str) -> Dict[str, float]:
        """
        Compute basic text quality metrics
        
        Args:
            response: The transcript text
            
        Returns:
            Dictionary of text quality scores
        """
        scores = {}
        
        if not response:
            return {"transcript_text_quality": 0.0}
        
        # Word count and basic stats
        words = response.split()
        scores["transcript_word_count"] = len(words)
        
        # Average word length
        if words:
            avg_word_length = sum(len(word.strip('.,!?:;')) for word in words) / len(words)
            scores["transcript_avg_word_length"] = avg_word_length
        
        # Character diversity
        unique_chars = len(set(response.lower()))
        total_chars = len(response)
        if total_chars > 0:
            scores["transcript_char_diversity"] = unique_chars / total_chars
        
        # Basic readability (simplified)
        sentences = len([s for s in response.split('.') if s.strip()])
        if sentences > 0:
            scores["transcript_words_per_sentence"] = len(words) / sentences
        
        # Check for repetitive patterns (potential transcription errors)
        repetition_score = self._detect_repetition(response)
        scores["transcript_repetition_score"] = repetition_score
        
        return scores
    
    def _detect_repetition(self, text: str) -> float:
        """
        Detect repetitive patterns in text
        
        Args:
            text: Input text
            
        Returns:
            Repetition score (0 = no repetition, 1 = high repetition)
        """
        if not text:
            return 0.0
        
        words = text.lower().split()
        if len(words) < 4:
            return 0.0
        
        # Look for repeated sequences of 2-4 words
        repetition_count = 0
        total_sequences = 0
        
        for seq_len in [2, 3, 4]:
            sequences = {}
            for i in range(len(words) - seq_len + 1):
                seq = ' '.join(words[i:i+seq_len])
                sequences[seq] = sequences.get(seq, 0) + 1
                total_sequences += 1
            
            # Count sequences that appear more than once
            repetition_count += sum(1 for count in sequences.values() if count > 1)
        
        if total_sequences > 0:
            return min(1.0, repetition_count / total_sequences)
        return 0.0
    
    def get_description(self) -> str:
        """Return description of the transcript quality metric"""
        return "Transcript quality evaluation using log probabilities and format validation"
