"""VertexAI Evaluation Service metrics for LLM-as-a-judge evaluations"""

import logging
import pandas as pd
from typing import Dict, Any, List, Optional

import vertexai
from vertexai.preview.evaluation import EvalTask, PointwiseMetric, MetricPromptTemplateExamples

from clients.gemini_client import GeminiClient
from clients.gcs_client import GCSClient
from metrics.base_metric import BaseMetric
from utils.prompt_manager import PromptManager

logger = logging.getLogger(__name__)

# Mapping from metric names to Vertex AI metric templates
METRIC_TEMPLATE_MAPPING = {
    "vertexai_fluency": MetricPromptTemplateExamples.Pointwise.FLUENCY,
    "vertexai_coherence": MetricPromptTemplateExamples.Pointwise.COHERENCE,
    "vertexai_safety": MetricPromptTemplateExamples.Pointwise.SAFETY,
    "vertexai_groundedness": MetricPromptTemplateExamples.Pointwise.GROUNDEDNESS,
    "vertexai_instruction_following": MetricPromptTemplateExamples.Pointwise.INSTRUCTION_FOLLOWING,
    "vertexai_verbosity": MetricPromptTemplateExamples.Pointwise.VERBOSITY,
    "vertexai_text_quality": MetricPromptTemplateExamples.Pointwise.TEXT_QUALITY,
    "vertexai_summarization_quality": MetricPromptTemplateExamples.Pointwise.SUMMARIZATION_QUALITY,
    "vertexai_question_answering_quality": MetricPromptTemplateExamples.Pointwise.QUESTION_ANSWERING_QUALITY
}


class VertexAIEvaluationMetric(BaseMetric):
    """Metric using VertexAI Evaluation Service for LLM-as-a-judge metrics."""

    def __init__(self, project_id: str, location: str, bucket_name: str):
        """
        Initialise VertexAI Evaluation metric.

        Args:
            project_id: GCP project ID
            location: GCP location for VertexAI services
        """
        super().__init__("vertexai_evaluation", "all")
        self.project_id = project_id
        self.location = location
        
        vertexai.init(project=project_id, location=location)
        
        self.gcs_client = GCSClient(bucket_name=bucket_name)
        self.prompt_manager = PromptManager(project=project_id, location=location)
        
        logger.info("Initialised VertexAI Evaluation metric.")

    def supports_batch_evaluation(self) -> bool:
        """Indicate that this metric supports batch evaluation"""
        return True

    def batch_compute(self, 
                      results_df: pd.DataFrame, 
                      experiment_config: Optional[Dict[str, Any]] = None,
                      global_config: Optional[Dict[str, Any]] = None,
                      # Legacy parameters for backward compatibility
                      use_case: Optional[str] = None, 
                      prompt_id: Optional[str] = None, 
                      metrics_config: Optional[List[str]] = None,
                      reference_config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Compute LLM-as-a-judge metrics for a batch of responses.
        
        Args:
            results_df: DataFrame containing experiment results with 'response_text' column
            experiment_config: Configuration for the specific experiment
            global_config: Global configuration
            
            # Legacy parameters (for backward compatibility):
            use_case: The use case of the experiment (e.g., summarisation, call_analysis)
            prompt_id: The ID of the prompt used in the experiment
            metrics_config: List of vertexai metrics to run
            reference_config: Configuration for generating references (transcripts)
            
        Returns:
            DataFrame with additional evaluation metric columns
        """
        # Handle backward compatibility with old interface
        if experiment_config is None and use_case is not None:
            # Legacy call - use old parameters
            return self._compute_batch_legacy(results_df, use_case, prompt_id, metrics_config, reference_config)
        
        # Extract parameters from new standardized configs
        if experiment_config is None:
            raise ValueError("experiment_config is required for new interface")
            
        use_case = experiment_config['use_case']
        prompt_id = experiment_config['prompt_id']
        
        # Filter metrics to only VertexAI ones
        all_metrics = experiment_config.get('metrics', [])
        metrics_config = [m for m in all_metrics if m.startswith('vertexai_')]
        
        reference_config = global_config.get('reference_config') if global_config else None
        
        return self._compute_batch_legacy(results_df, use_case, prompt_id, metrics_config, reference_config)

    def _compute_batch_legacy(self,
                             results_df: pd.DataFrame, 
                             use_case: str, 
                             prompt_id: str, 
                             metrics_config: List[str],
                             reference_config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Legacy implementation of batch compute for backward compatibility
        
        Args:
            results_df: DataFrame containing experiment results with 'response_text' column
            use_case: The use case of the experiment (e.g., summarisation, call_analysis)
            prompt_id: The ID of the prompt used in the experiment
            metrics_config: List of vertexai metrics to run
            reference_config: Configuration for generating references (transcripts)
            
        Returns:
            DataFrame with additional evaluation metric columns
        """
        logger.info(f"Running batch VertexAI evaluation on {len(results_df)} responses for use case: {use_case}")
        
        if results_df.empty:
            logger.warning("Empty DataFrame provided for batch evaluation")
            return results_df
        
        if 'response_text' not in results_df.columns:
            logger.error("DataFrame missing 'response_text' column")
            return results_df
        
        try:
            eval_data = results_df[results_df['response_text'].str.strip() != ''].copy()
            
            if eval_data.empty:
                logger.warning("No valid responses found for evaluation")
                return results_df

            # Add prompt to the evaluation data
            prompt_text = self.prompt_manager.load(prompt_id)
            eval_data['prompt'] = prompt_text

            # Handle transcription for summarisation and call analysis use cases
            if use_case in ["summarisation", "call_analysis"] and 'reference' not in eval_data.columns:
                if reference_config:
                    logger.info("Generating transcripts for reference...")
                    # Initialise a temporary client for reference generation
                    reference_gemini_client = GeminiClient(
                        model_id=reference_config['client']['model_id'],
                        config={'project_id': self.project_id, 'location': self.location}
                    )
                    eval_data['reference'] = eval_data['audio_source'].apply(
                        lambda audio_source: self._get_transcript(audio_source, reference_config, reference_gemini_client)
                    )
                else:
                    logger.warning("Reference needed for groundedness but no reference_config provided.")

            # Rename columns for evaluation
            eval_data = eval_data.rename(columns={'response_text': 'response'})

            # Run batch evaluation
            metrics = [METRIC_TEMPLATE_MAPPING[metric] for metric in metrics_config if metric in METRIC_TEMPLATE_MAPPING]
            if not metrics:
                logger.warning("No valid Vertex AI metrics to run for this experiment.")
                return results_df

            eval_scores_df = self._run_batch_evaluation(eval_data, metrics)
            
            # Merge results back
            results_with_eval = results_df.merge(eval_scores_df, left_index=True, right_index=True, how='left')
            
            logger.info("Batch VertexAI evaluation completed successfully")
            return results_with_eval
            
        except Exception as e:
            logger.error(f"Error in batch VertexAI evaluation: {e}")
            results_df['vertexai_eval_batch_error'] = 1.0
            return results_df

    def _get_transcript(self, audio_source: str, reference_config: Dict[str, Any], gemini_client: GeminiClient) -> str:
        """Get transcript for an audio file from GCS using the configured Gemini model."""
        try:
            audio_bytes = self.gcs_client.download_bytes(audio_source)
            prompt = self.prompt_manager.load(reference_config['prompt_id'])
            
            result = gemini_client.generate_from_audio(
                audio_data=audio_bytes,
                prompt=prompt,
                generation_config=reference_config['generation_config']
            )
            return result.get("response_text", "")
        except Exception as e:
            logger.error(f"Failed to get transcript for {audio_source}: {e}")
            return ""

    def _run_batch_evaluation(self, eval_data: pd.DataFrame, metrics: List[PointwiseMetric]) -> pd.DataFrame:
        """Run batch evaluation for all configured metrics."""
        try:
            eval_task = EvalTask(
                dataset=eval_data,
                metrics=metrics,
                experiment="llm-response-evaluation"
            )
            
            eval_result = eval_task.evaluate()
            
            metrics_table = eval_result.metrics_table
            
            # Extract score columns and explanation columns
            score_columns = [col for col in metrics_table.columns if col.endswith('/score')]
            explanation_columns = [col for col in metrics_table.columns if not col.endswith('/score') and col not in ['prompt', 'response', 'reference']]
            
            results_df = pd.DataFrame()
            
            # Add score columns with cleaner names (remove '/score' suffix)
            for col in score_columns:
                clean_name = col.replace('/score', '')
                results_df[clean_name] = metrics_table[col]
            
            # Add explanation columns as-is
            for col in explanation_columns:
                results_df[col] = metrics_table[col]
            
            results_df['prompt'] = eval_data['prompt'].values
            results_df['response'] = eval_data['response'].values
            if 'reference' in eval_data.columns:
                results_df['reference'] = eval_data['reference'].values
            
            return results_df

        except Exception as e:
            logger.error(f"Error running batch evaluation: {e}")
            return pd.DataFrame({'vertexai_eval_batch_error': [1.0] * len(eval_data)})

    def get_description(self) -> str:
        """Return a description of what this metric measures"""
        return "Evaluates responses using VertexAI Evaluation Service for various quality and safety metrics."
