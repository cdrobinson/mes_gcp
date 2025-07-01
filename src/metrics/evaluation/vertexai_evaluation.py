"""VertexAI Evaluation Service metrics for LLM-as-a-judge evaluations"""

import logging
import pandas as pd

import vertexai
from vertexai.evaluation import EvalTask, PointwiseMetric, constants

from metrics.base_metric import BaseMetric

logger = logging.getLogger(__name__)

# Mapping from the metric name in the config to the SDK's PointwiseMetric object
POINTWISE_METRIC_MAP = {
    "FLUENCY": PointwiseMetric(metric=constants.Metric.FLUENCY),
    "COHERENCE": PointwiseMetric(metric=constants.Metric.COHERENCE),
    "SAFETY": PointwiseMetric(metric=constants.Metric.SAFETY),
    "GROUNDEDNESS": PointwiseMetric(metric=constants.Metric.GROUNDEDNESS),
    "INSTRUCTION_FOLLOWING": PointwiseMetric(metric=constants.Metric.INSTRUCTION_FOLLOWING),
    "VERBOSITY": PointwiseMetric(metric=constants.Metric.VERBOSITY),
    "TEXT_QUALITY": PointwiseMetric(metric=constants.Metric.TEXT_QUALITY),
    "SUMMARIZATION_QUALITY": PointwiseMetric(metric=constants.Metric.SUMMARIZATION_QUALITY),
    "QUESTION_ANSWERING_QUALITY": PointwiseMetric(metric=constants.Metric.QUESTION_ANSWERING_QUALITY),
}


class VertexAIEvaluationMetric(BaseMetric):
    """Metric using VertexAI Evaluation Service for LLM-as-a-judge metrics."""

    def __init__(self, project_id: str, location: str = "us-central1"):
        super().__init__("vertexai_evaluation", "all")
        self.project_id = project_id
        self.location = location
        vertexai.init(project=project_id, location=location)
        logger.info("Initialised VertexAI Evaluation metric.")

    def supports_batch_evaluation(self) -> bool:
        return True

    def batch_compute(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes LLM-as-a-judge metrics for a batch of responses.
        It groups by experiment, as each experiment can have a different set of metrics.
        """
        if 'experiment_metrics' not in results_df.columns:
            logger.warning("`experiment_metrics` column not found. Skipping Vertex AI Evaluation.")
            return results_df

        final_results_df = results_df.copy()

        for experiment_name, group in results_df.groupby('experiment_name'):
            logger.info(f"Running Vertex AI evaluation for experiment: {experiment_name}")
            
            metrics_for_exp = group['experiment_metrics'].iloc[0]
            
            metrics_to_run = []
            for name in metrics_for_exp:
                map_key = name.replace('vertexai_', '').upper()
                if map_key in POINTWISE_METRIC_MAP:
                    metrics_to_run.append(POINTWISE_METRIC_MAP[map_key])

            if not metrics_to_run:
                logger.info(f"No applicable Vertex AI metrics for '{experiment_name}'. Skipping.")
                continue

            required_cols = ['response', 'prompt']
            if any(m.metric == constants.Metric.GROUNDEDNESS for m in metrics_to_run):
                required_cols.append('reference')

            eval_dataset = group.dropna(subset=required_cols).copy()

            if eval_dataset.empty:
                logger.warning(f"No valid rows with required columns for Vertex AI evaluation in '{experiment_name}'.")
                continue

            try:
                eval_task = EvalTask(dataset=eval_dataset, metrics=metrics_to_run)
                eval_result = eval_task.evaluate()
                
                metric_results_table = eval_result.metrics_table
                
                eval_score_cols = [col for col in metric_results_table.columns if col not in eval_dataset.columns]
                
                rename_map = {col: f"vertexai_{col.lower().replace('/score', '').replace('_', '')}" for col in eval_score_cols}
                metric_results_table.rename(columns=rename_map, inplace=True)

                final_results_df = final_results_df.merge(
                    metric_results_table[list(rename_map.values())],
                    left_index=True,
                    right_index=True,
                    how='left'
                )

            except Exception as e:
                logger.error(f"Error during Vertex AI batch evaluation for '{experiment_name}': {e}", exc_info=True)
                final_results_df.loc[group.index, f'vertexai_eval_error'] = 1.0
        
        return final_results_df

    def compute(self, response: str, metadata: dict, **kwargs) -> dict:
        """Individual compute is not supported for this batch-optimized metric."""
        logger.warning(f"{self.name} is a batch-only metric. Use batch_compute().")
        return {}

    def get_description(self) -> str:
        return "Evaluates responses using pre-built templates from the VertexAI Evaluation Service."