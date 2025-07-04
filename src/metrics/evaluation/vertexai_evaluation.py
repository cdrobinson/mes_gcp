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
        Keeps one metric column per name and avoids _x/_y by overwriting in place,
        fter realigning the metric table’s index to the rows that were scored.
        """
        if "experiment_metrics" not in results_df.columns:
            logger.warning("`experiment_metrics` column not found – skipping Vertex AI evaluation.")
            return results_df

        final_results_df = results_df.copy()

        for exp_name, group in results_df.groupby("experiment_name"):
            logger.info("Vertex AI evaluation: %s", exp_name)

            metrics_cfg = group["experiment_metrics"].iloc[0]
            metrics_to_run = [
                POINTWISE_METRIC_MAP[m.replace("vertexai_", "").upper()]
                for m in metrics_cfg
                if m.replace("vertexai_", "").upper() in POINTWISE_METRIC_MAP
            ]
            if not metrics_to_run:
                logger.info("No Vertex AI metrics requested for %s – skipping.", exp_name)
                continue

            eval_dataset = self._prepare_eval_dataset(group, metrics_cfg)
            if eval_dataset.empty:
                logger.warning("No rows with all required fields for %s – skipping.", exp_name)
                continue

            try:
                eval_task = EvalTask(dataset=eval_dataset,
                                   metrics=metrics_to_run,
                                   experiment="mes-experiment")
                eval_result = eval_task.evaluate()

                metrics_tbl = eval_result.metrics_table
                metrics_tbl.index = eval_dataset.index

                score_cols = [c for c in metrics_tbl.columns if c not in eval_dataset.columns]
                rename = {
                    c: f"vertexai_{c.lower().replace('/score', '').replace('_', '')}"
                    for c in score_cols
                }
                metrics_tbl = metrics_tbl.rename(columns=rename)[list(rename.values())]

                missing_cols = [c for c in metrics_tbl.columns if c not in final_results_df]
                for c in missing_cols:
                    final_results_df[c] = pd.NA

                final_results_df.loc[metrics_tbl.index, metrics_tbl.columns] = metrics_tbl.values

            except Exception as err:
                logger.error("Vertex AI evaluation failed for %s: %s", exp_name, err, exc_info=True)
                final_results_df.loc[group.index, "vertexai_eval_error"] = 1.0

        return final_results_df


    def compute(self, response: str, metadata: dict, **kwargs) -> dict:
        """Individual compute is not supported for this batch-optimized metric."""
        logger.warning(f"{self.name} is a batch-only metric. Use batch_compute().")
        return {}

    def get_description(self) -> str:
        return "Evaluates responses using pre-built templates from the VertexAI Evaluation Service."