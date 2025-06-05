import nltk
import numpy as np
from rouge_score import rouge_scorer
from nltk.tokenize import sent_tokenize
from typing import Dict, Any, List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure nltk resources are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

class RougeEvaluator:
    """
    ROUGE evaluation for summaries without ground truth.
    Uses extractive reference methods and self-evaluation techniques.
    """
    
    def __init__(self, rouge_types: List[str] = None):
        """
        Initialize ROUGE evaluator.
        
        Args:
            rouge_types: List of ROUGE types to compute (default: ['rouge1', 'rouge2', 'rougeL'])
        """
        self.rouge_types = rouge_types or ['rouge1', 'rouge2', 'rougeL']
        self.scorer = rouge_scorer.RougeScorer(self.rouge_types, use_stemmer=True)
    
    def _extract_top_sentences(self, text: str, num_sentences: int = 3) -> str:
        """
        Extract top sentences from text using TF-IDF scoring as pseudo-reference.
        
        Args:
            text: Input text to extract sentences from
            num_sentences: Number of top sentences to extract
            
        Returns:
            Extracted sentences as reference text
        """
        if not text or not text.strip():
            return ""
            
        sentences = sent_tokenize(text)
        if len(sentences) <= num_sentences:
            return text
            
        # Use TF-IDF to score sentences
        vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
            # Score sentences by sum of TF-IDF values
            sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
            
            # Get top sentences indices
            top_indices = sentence_scores.argsort()[-num_sentences:][::-1]
            top_indices = sorted(top_indices)  # Maintain original order
            
            return ' '.join([sentences[i] for i in top_indices])
        except:
            # Fallback to first few sentences if TF-IDF fails
            return ' '.join(sentences[:num_sentences])
    
    def evaluate_with_extractive_reference(self, 
                                         transcript: str, 
                                         summary: str,
                                         reference_sentences: int = 3) -> Dict[str, Any]:
        """
        Evaluate summary using extractive reference from the original transcript.
        
        Args:
            transcript: Original transcript
            summary: Generated summary to evaluate
            reference_sentences: Number of sentences to extract as reference
            
        Returns:
            Dictionary containing ROUGE scores
        """
        if not summary or not summary.strip():
            return {f"rouge_extractive_{rouge_type}": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0} 
                   for rouge_type in self.rouge_types}
        
        # Create extractive reference
        extractive_reference = self._extract_top_sentences(transcript, reference_sentences)
        
        if not extractive_reference:
            return {f"rouge_extractive_{rouge_type}": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0} 
                   for rouge_type in self.rouge_types}
        
        # Calculate ROUGE scores
        scores = self.scorer.score(extractive_reference, summary)
        
        # Format results
        results = {}
        for rouge_type in self.rouge_types:
            if rouge_type in scores:
                score = scores[rouge_type]
                results[f"rouge_extractive_{rouge_type}"] = {
                    "precision": round(score.precision, 4),
                    "recall": round(score.recall, 4),
                    "fmeasure": round(score.fmeasure, 4)
                }
        
        return results
    
    def evaluate_self_rouge(self, text: str, summary: str) -> Dict[str, Any]:
        """
        Calculate ROUGE scores using the original text as reference.
        Useful for measuring extractiveness of the summary.
        
        Args:
            text: Original text (transcript)
            summary: Generated summary
            
        Returns:
            Dictionary containing self-ROUGE scores
        """
        if not summary or not summary.strip() or not text or not text.strip():
            return {f"rouge_self_{rouge_type}": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0} 
                   for rouge_type in self.rouge_types}
        
        scores = self.scorer.score(text, summary)
        
        results = {}
        for rouge_type in self.rouge_types:
            if rouge_type in scores:
                score = scores[rouge_type]
                results[f"rouge_self_{rouge_type}"] = {
                    "precision": round(score.precision, 4),
                    "recall": round(score.recall, 4),
                    "fmeasure": round(score.fmeasure, 4)
                }
        
        return results

    def get_all_rouge_metrics(self, transcript: str, summary: str) -> Dict[str, Any]:
        """
        Get comprehensive ROUGE evaluation metrics.
        
        Args:
            transcript: Original transcript
            summary: Generated summary
            
        Returns:
            Dictionary containing all ROUGE metrics
        """
        metrics = {}
        
        # Extractive reference ROUGE
        extractive_metrics = self.evaluate_with_extractive_reference(transcript, summary)
        metrics.update(extractive_metrics)
        
        # Self ROUGE (measures extractiveness)
        self_metrics = self.evaluate_self_rouge(transcript, summary)
        metrics.update(self_metrics)
        
        return metrics
