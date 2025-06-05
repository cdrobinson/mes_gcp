import time
from typing import Dict, Any, List, Optional, Tuple
from core.gcp_client import get_gemini_model, get_gemini_generation_config
from vertexai.generative_models import Part, GenerationConfig
from core.retry_handler import default_retry_decorator

class LLMJudgeEvaluator:
    """
    LLM-as-a-Judge evaluator for assessing text quality across multiple dimensions.
    Uses a separate LLM instance to evaluate generated content.
    """
    
    def __init__(self, judge_llm_config_name: str = None):
        """
        Initialize LLM Judge evaluator.
        
        Args:
            judge_llm_config_name: LLM configuration to use for judging (should be different from evaluated model)
        """
        # Use a different LLM config for judging to avoid self-evaluation bias
        self.judge_llm_config_name = judge_llm_config_name or "gemini_1_5_flash_default"
        self.judge_model = get_gemini_model(self.judge_llm_config_name)
        self.judge_gen_config = get_gemini_generation_config(self.judge_llm_config_name)
        
        # Make judge more deterministic
        judge_config_dict = self.judge_gen_config.to_dict()
        judge_config_dict.update({"temperature": 0.1})  # Lower temperature for consistency
        self.judge_gen_config = GenerationConfig(**judge_config_dict)
    
    def _get_summary_evaluation_prompt(self) -> str:
        """Get the prompt template for summary evaluation."""
        return """You are an expert evaluator of text summaries. Please evaluate the following summary based on the original transcript.

Rate each dimension on a scale of 1-5, where:
1 = Very Poor, 2 = Poor, 3 = Average, 4 = Good, 5 = Excellent

EVALUATION DIMENSIONS:
1. **Accuracy**: How factually correct is the summary compared to the original?
2. **Completeness**: How well does the summary capture the main points?
3. **Conciseness**: How well does the summary avoid unnecessary details while maintaining clarity?
4. **Coherence**: How logically structured and easy to follow is the summary?
5. **Clarity**: How clear and understandable is the language used?

ORIGINAL TRANSCRIPT:
{transcript}

SUMMARY TO EVALUATE:
{summary}

Please provide your evaluation in the following format:
ACCURACY: [score] - [brief explanation]
COMPLETENESS: [score] - [brief explanation]  
CONCISENESS: [score] - [brief explanation]
COHERENCE: [score] - [brief explanation]
CLARITY: [score] - [brief explanation]
OVERALL: [score] - [brief overall assessment]
"""

    def _get_analysis_evaluation_prompt(self) -> str:
        """Get the prompt template for analysis evaluation."""
        return """You are an expert evaluator of call analysis reports. Please evaluate the following analysis based on the original transcript.

Rate each dimension on a scale of 1-5, where:
1 = Very Poor, 2 = Poor, 3 = Average, 4 = Good, 5 = Excellent

EVALUATION DIMENSIONS:
1. **Accuracy**: How factually correct is the analysis compared to the original?
2. **Insight Quality**: How valuable and actionable are the insights provided?
3. **Thoroughness**: How comprehensively does the analysis address relevant aspects?
4. **Structure**: How well-organized and logical is the analysis format?
5. **Actionability**: How useful are the recommendations/observations for improvement?

ORIGINAL TRANSCRIPT:
{transcript}

ANALYSIS TO EVALUATE:
{analysis}

Please provide your evaluation in the following format:
ACCURACY: [score] - [brief explanation]
INSIGHT_QUALITY: [score] - [brief explanation]
THOROUGHNESS: [score] - [brief explanation]
STRUCTURE: [score] - [brief explanation]
ACTIONABILITY: [score] - [brief explanation]
OVERALL: [score] - [brief overall assessment]
"""

    @default_retry_decorator()
    def _call_judge_llm(self, prompt: str) -> Tuple[str, float, Dict[str, Any]]:
        """Call the judge LLM with retry logic."""
        start_time = time.time()
        
        response = self.judge_model.generate_content(
            contents=[Part.from_text(prompt)],
            generation_config=self.judge_gen_config
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        llm_output = ""
        if response.candidates:
            for part in response.candidates[0].content.parts:
                if part.text:
                    llm_output += part.text
        
        usage_metadata = response.usage_metadata if hasattr(response, 'usage_metadata') else None
        metadata = {
            "judge_llm_config": self.judge_llm_config_name,
            "judge_processing_time": round(processing_time, 3),
            "judge_tokens_used": usage_metadata.total_token_count if usage_metadata else 0
        }
        
        return llm_output.strip(), processing_time, metadata

    def _parse_evaluation_scores(self, evaluation_text: str) -> Dict[str, Any]:
        """Parse numerical scores from the LLM judge evaluation."""
        scores = {}
        lines = evaluation_text.split('\n')
        
        dimension_mappings = {
            'ACCURACY': 'accuracy',
            'COMPLETENESS': 'completeness', 
            'CONCISENESS': 'conciseness',
            'COHERENCE': 'coherence',
            'CLARITY': 'clarity',
            'INSIGHT_QUALITY': 'insight_quality',
            'THOROUGHNESS': 'thoroughness',
            'STRUCTURE': 'structure',
            'ACTIONABILITY': 'actionability',
            'OVERALL': 'overall'
        }
        
        for line in lines:
            line = line.strip()
            for key, mapped_key in dimension_mappings.items():
                if line.startswith(f"{key}:"):
                    try:
                        # Extract score (first number after colon)
                        score_part = line.split(':')[1].strip()
                        score = int(score_part.split()[0])
                        if 1 <= score <= 5:  # Validate score range
                            scores[f"llm_judge_{mapped_key}"] = score
                    except:
                        continue  # Skip if parsing fails
        
        return scores

    def evaluate_summary(self, transcript: str, summary: str) -> Dict[str, Any]:
        """
        Evaluate a summary using LLM-as-a-Judge.
        
        Args:
            transcript: Original transcript
            summary: Generated summary to evaluate
            
        Returns:
            Dictionary containing LLM judge scores and metadata
        """
        if not summary or not summary.strip():
            return {
                "llm_judge_accuracy": 0,
                "llm_judge_completeness": 0,
                "llm_judge_conciseness": 0,
                "llm_judge_coherence": 0,
                "llm_judge_clarity": 0,
                "llm_judge_overall": 0,
                "llm_judge_evaluation_text": "Error: Empty summary",
                "llm_judge_metadata": {"error": "Empty summary provided"}
            }
        
        prompt = self._get_summary_evaluation_prompt().format(
            transcript=transcript[:2000],  # Limit length to avoid token limits
            summary=summary
        )
        
        try:
            evaluation_text, _, metadata = self._call_judge_llm(prompt)
            scores = self._parse_evaluation_scores(evaluation_text)
            
            result = {
                "llm_judge_evaluation_text": evaluation_text,
                "llm_judge_metadata": metadata,
                **scores
            }
            
            return result
            
        except Exception as e:
            return {
                "llm_judge_accuracy": 0,
                "llm_judge_completeness": 0,
                "llm_judge_conciseness": 0,
                "llm_judge_coherence": 0,
                "llm_judge_clarity": 0,
                "llm_judge_overall": 0,
                "llm_judge_evaluation_text": f"Error: {str(e)}",
                "llm_judge_metadata": {"error": str(e)}
            }

    def evaluate_analysis(self, transcript: str, analysis: str) -> Dict[str, Any]:
        """
        Evaluate an analysis using LLM-as-a-Judge.
        
        Args:
            transcript: Original transcript
            analysis: Generated analysis to evaluate
            
        Returns:
            Dictionary containing LLM judge scores and metadata
        """
        if not analysis or not analysis.strip():
            return {
                "llm_judge_accuracy": 0,
                "llm_judge_insight_quality": 0,
                "llm_judge_thoroughness": 0,
                "llm_judge_structure": 0,
                "llm_judge_actionability": 0,
                "llm_judge_overall": 0,
                "llm_judge_evaluation_text": "Error: Empty analysis",
                "llm_judge_metadata": {"error": "Empty analysis provided"}
            }
        
        prompt = self._get_analysis_evaluation_prompt().format(
            transcript=transcript[:2000],  # Limit length to avoid token limits
            analysis=analysis
        )
        
        try:
            evaluation_text, _, metadata = self._call_judge_llm(prompt)
            scores = self._parse_evaluation_scores(evaluation_text)
            
            result = {
                "llm_judge_evaluation_text": evaluation_text,
                "llm_judge_metadata": metadata,
                **scores
            }
            
            return result
            
        except Exception as e:
            return {
                "llm_judge_accuracy": 0,
                "llm_judge_insight_quality": 0,
                "llm_judge_thoroughness": 0,
                "llm_judge_structure": 0,
                "llm_judge_actionability": 0,
                "llm_judge_overall": 0,
                "llm_judge_evaluation_text": f"Error: {str(e)}",
                "llm_judge_metadata": {"error": str(e)}
            }
