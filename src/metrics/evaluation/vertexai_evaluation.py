"""VertexAI Evaluation Service metrics for LLM-as-a-judge evaluations"""

import logging
import pandas as pd

import vertexai
from vertexai.evaluation import EvalTask, PointwiseMetric, constants, MetricPromptTemplateExamples

from metrics.base_metric import BaseMetric

logger = logging.getLogger(__name__)

# Mapping from the metric name in the config to the SDK's PointwiseMetric object
POINTWISE_METRIC_MAP = {
    "FLUENCY": MetricPromptTemplateExamples.Pointwise.FLUENCY,
    "COHERENCE": MetricPromptTemplateExamples.Pointwise.COHERENCE,
    "SAFETY": MetricPromptTemplateExamples.Pointwise.SAFETY,
    "GROUNDEDNESS": MetricPromptTemplateExamples.Pointwise.GROUNDEDNESS,
    "INSTRUCTION_FOLLOWING": MetricPromptTemplateExamples.Pointwise.INSTRUCTION_FOLLOWING,
    "VERBOSITY": MetricPromptTemplateExamples.Pointwise.VERBOSITY,
    "TEXT_QUALITY": MetricPromptTemplateExamples.Pointwise.TEXT_QUALITY,
    "SUMMARIZATION_QUALITY": MetricPromptTemplateExamples.Pointwise.SUMMARIZATION_QUALITY,
    "QUESTION_ANSWERING_QUALITY": MetricPromptTemplateExamples.Pointwise.QUESTION_ANSWERING_QUALITY,
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

    def _prepare_eval_dataset(self, group: pd.DataFrame, metrics_for_exp) -> pd.DataFrame:
        """
        Prepare the eval dataset for VertexAI evaluation.
        If GROUNDEDNESS or SUMMARIZATION_QUALITY is in the metrics, append reference to prompt.
        Only keep 'response' and 'prompt' columns.
        """
        needs_reference = any(
            name.replace('vertexai_', '').upper() in ("GROUNDEDNESS", "SUMMARIZATION_QUALITY")
            for name in metrics_for_exp
        )
        required_cols = ['response', 'prompt']
        if needs_reference:
            required_cols.append('reference')

        eval_dataset = group.dropna(subset=required_cols).copy()

        if needs_reference and not eval_dataset.empty:
            eval_dataset['prompt'] = (
                eval_dataset['prompt'].astype(str) +
                "\n\nContext: " +
                eval_dataset['reference'].astype(str)
            )

        eval_dataset = eval_dataset[['response', 'prompt']]
        return eval_dataset

    def batch_compute(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes LLM-as-a-judge metrics for a batch of responses.
        Groups by experiment so that each experiment can request different metrics.
        """
        if 'experiment_metrics' not in results_df.columns:
            logger.warning("`experiment_metrics` column not found. Skipping Vertex AI Evaluation.")
            return results_df

        final_results_df = results_df.copy()

        for experiment_name, group in results_df.groupby('experiment_name'):
            logger.info(f"Running Vertex AI evaluation for experiment: {experiment_name}")

            metrics_for_exp = group['experiment_metrics'].iloc[0]

            metrics_to_run = [
                POINTWISE_METRIC_MAP[name.replace('vertexai_', '').upper()]
                for name in metrics_for_exp
                if name.replace('vertexai_', '').upper() in POINTWISE_METRIC_MAP
            ]

            if not metrics_to_run:
                logger.info(f"No applicable Vertex AI metrics for '{experiment_name}'. Skipping.")
                continue

            eval_dataset = self._prepare_eval_dataset(group, metrics_for_exp)
            if eval_dataset.empty:
                logger.warning(
                    f"No valid rows with required columns for Vertex AI evaluation in '{experiment_name}'."
                )
                continue

            try:
                eval_task = EvalTask(dataset=eval_dataset,
                                       metrics=metrics_to_run,
                                       experiment="mes-experiment")
                eval_result = eval_task.evaluate()

                metric_results_table = eval_result.metrics_table

                eval_score_cols = [
                    col for col in metric_results_table.columns
                    if col not in eval_dataset.columns
                ]

                rename_map = {
                    col: f"vertexai_{col.lower().replace('/score', '').replace('_', '')}"
                    for col in eval_score_cols
                }
                metric_results_table.rename(columns=rename_map, inplace=True)

                for col in metric_results_table.columns:
                    # Add the column if this is the first experiment that produced it
                    if col not in final_results_df.columns:
                        final_results_df[col] = pd.NA

                    final_results_df.loc[group.index, col] = metric_results_table[col]

            except Exception as e:
                logger.error(
                    f"Error during Vertex AI batch evaluation for '{experiment_name}': {e}",
                    exc_info=True,
                )
                final_results_df.loc[group.index, 'vertexai_eval_error'] = 1.0

        return final_results_df

    def compute(self, response: str, metadata: dict, **kwargs) -> dict:
        """Individual compute is not supported for this batch-optimized metric."""
        logger.warning(f"{self.name} is a batch-only metric. Use batch_compute().")
        return {}

    def get_description(self) -> str:
        return "Evaluates responses using pre-built templates from the VertexAI Evaluation Service."