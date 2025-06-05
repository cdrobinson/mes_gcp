from typing import Dict, Any, Optional
from evaluation.metrics_calculator import MetricsCalculator
from evaluation.rouge_evaluator import RougeEvaluator
from evaluation.llm_judge_evaluator import LLMJudgeEvaluator

class AdvancedMetricsCalculator:
    """
    Comprehensive metrics calculator that combines basic metrics, ROUGE, and LLM-as-a-Judge evaluation.
    """
    
    def __init__(self, judge_llm_config_name: Optional[str] = None):
        """
        Initialize the advanced metrics calculator.
        
        Args:
            judge_llm_config_name: LLM configuration name to use for LLM-as-a-Judge evaluation
        """
        self.basic_metrics = MetricsCalculator()
        self.rouge_evaluator = RougeEvaluator()
        self.llm_judge = LLMJudgeEvaluator(judge_llm_config_name)
    
    def calculate_comprehensive_metrics(self,
                                      transcript: str,
                                      llm_output: str,
                                      agent_type: str,
                                      audio_duration_seconds: Optional[float] = None,
                                      stt_processing_time_seconds: Optional[float] = None,
                                      llm_processing_time_seconds: Optional[float] = None,
                                      llm_metadata: Optional[Dict[str, Any]] = None,
                                      include_llm_judge: bool = True,
                                      include_rouge: bool = True) -> Dict[str, Any]:
        """
        Calculate comprehensive metrics including basic metrics, ROUGE, and LLM-as-a-Judge.
        
        Args:
            transcript: Original transcript
            llm_output: LLM generated output (summary/analysis)
            agent_type: Type of agent (for determining evaluation approach)
            audio_duration_seconds: Duration of audio file
            stt_processing_time_seconds: Time taken for speech-to-text
            llm_processing_time_seconds: Time taken for LLM processing
            llm_metadata: Metadata from LLM processing
            include_llm_judge: Whether to include LLM-as-a-Judge evaluation
            include_rouge: Whether to include ROUGE evaluation
            
        Returns:
            Dictionary containing all calculated metrics
        """
        # Start with basic metrics
        all_metrics = self.basic_metrics.get_all_metrics(
            transcript=transcript,
            llm_output=llm_output,
            audio_duration_seconds=audio_duration_seconds,
            stt_processing_time_seconds=stt_processing_time_seconds,
            llm_processing_time_seconds=llm_processing_time_seconds,
            llm_metadata=llm_metadata
        )
        
        # Add ROUGE metrics if requested
        if include_rouge and transcript and llm_output:
            try:
                rouge_metrics = self.rouge_evaluator.get_all_rouge_metrics(transcript, llm_output)
                # Prefix ROUGE metrics for clarity
                for key, value in rouge_metrics.items():
                    all_metrics[f"metric_{key}"] = value
            except Exception as e:
                print(f"Error calculating ROUGE metrics: {e}")
                all_metrics["metric_rouge_error"] = str(e)
        
        # Add LLM-as-a-Judge metrics if requested
        if include_llm_judge and transcript and llm_output:
            try:
                if agent_type == "call_summarization":
                    judge_metrics = self.llm_judge.evaluate_summary(transcript, llm_output)
                elif agent_type == "call_agent_analysis":
                    judge_metrics = self.llm_judge.evaluate_analysis(transcript, llm_output)
                else:
                    # Default to summary evaluation for unknown agent types
                    judge_metrics = self.llm_judge.evaluate_summary(transcript, llm_output)
                
                # Prefix LLM judge metrics for clarity
                for key, value in judge_metrics.items():
                    all_metrics[f"metric_{key}"] = value
                    
            except Exception as e:
                print(f"Error calculating LLM-as-a-Judge metrics: {e}")
                all_metrics["metric_llm_judge_error"] = str(e)
        
        return all_metrics
    
    def get_metric_summary(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and summarize key metrics for quick analysis.
        
        Args:
            metrics: Full metrics dictionary
            
        Returns:
            Dictionary containing summarized key metrics
        """
        summary = {}
        
        # Basic metrics summary
        summary["word_count_ratio"] = (
            metrics.get("metric_llm_llm_output_length_words", 0) / 
            max(metrics.get("metric_transcript_transcript_length_words", 1), 1)
        )
        
        # ROUGE summary (F-measure scores)
        rouge_keys = [k for k in metrics.keys() if "rouge" in k and "fmeasure" in str(metrics.get(k, {}))]
        for key in rouge_keys:
            if isinstance(metrics[key], dict) and "fmeasure" in metrics[key]:
                summary[f"{key}_f1"] = metrics[key]["fmeasure"]
        
        # LLM Judge summary
        judge_keys = [k for k in metrics.keys() if "llm_judge" in k and k.endswith(("accuracy", "overall"))]
        for key in judge_keys:
            if isinstance(metrics[key], (int, float)):
                summary[key] = metrics[key]
        
        return summary
