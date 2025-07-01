"""Experiment orchestrator for LLM evaluations"""

import logging
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import yaml

from clients import GeminiClient, BaseLLMClient
from clients.gcs_client import GCSClient
from clients.bigquery_client import BigQueryClient
from utils.prompt_manager import PromptManager
from metrics.base_metric import BaseMetric
from metrics.transcription.transcript_quality import TranscriptQualityMetric
from metrics.safety.safety import SafetyMetric
from metrics.evaluation.vertexai_evaluation import VertexAIEvaluationMetric, POINTWISE_METRIC_MAP

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Orchestrator for LLM evaluation experiments"""
    
    def __init__(self, config_path: str):
        """
        Initialise the experiment runner
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        self.gcs_client = self._init_gcs_client()
        
        vertexai_config = self.config.get('vertexai', {})
        project_id = vertexai_config.get('project_id', self.config['project'])
        location = vertexai_config.get('location', self.config['location'])

        self.prompt_manager = PromptManager(project_id, location)
        
        self.reference_client = self._init_llm_client(self.config['reference_generation']['client'])
        
        self.bigquery_client = BigQueryClient(
            project_id=self.config.get('bigquery', {}).get('project_id', project_id),
            location=self.config.get('bigquery', {}).get('location', location)
        )
        
        self.available_metrics = {
            "transcript_quality": TranscriptQualityMetric(),
            "safety": SafetyMetric(
                project_id=project_id,
                location=location,
                template_id=vertexai_config.get('model_armour_template_id', 'default')
            ),
            "vertexai_evaluation": VertexAIEvaluationMetric(
                project_id=project_id,
                location=location
            )
        }
        
        self.results = []
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"ExperimentRunner initialised with {len(self.available_metrics)} available metrics")

    def _init_llm_client(self, client_config: Dict[str, Any]) -> BaseLLMClient:
        client_name = client_config.get('name', 'gemini')
        model_id = client_config.get('model_id')
        if not model_id:
            raise ValueError("LLM 'model_id' not specified in client config")
        if client_name == 'gemini':
            return GeminiClient(
                model_id=model_id,
                config={'project_id': self.config['project'], 'location': self.config['location']},
                max_attempts=self.config.get('retry', {}).get('max_attempts', 3)
            )
        raise ValueError(f"Unknown LLM client: {client_name}")

    def _load_config(self) -> Dict[str, Any]:
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        required_keys = ['project', 'location', 'bucket', 'experiments', 'reference_generation']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Required configuration key missing: {key}")
        logger.info(f"Loaded configuration with {len(config['experiments'])} experiments")
        return config
    
    def _init_gcs_client(self) -> GCSClient:
        return GCSClient(bucket_name=self.config['bucket'])

    def run_experiments(self, experiment_names: Optional[List[str]] = None) -> pd.DataFrame:
        experiments_to_run = self.config['experiments']
        if experiment_names:
            experiments_to_run = [exp for exp in experiments_to_run if exp['name'] in experiment_names]
        
        logger.info(f"Running {len(experiments_to_run)} experiments")
        audio_files = self._get_audio_files()
        logger.info(f"Processing {len(audio_files)} audio files")
        
        for experiment in experiments_to_run:
            self._run_single_experiment(experiment, audio_files)
        
        if not self.results:
            logger.warning("No results generated to process")
            return pd.DataFrame()
        
        results_df = pd.DataFrame(self.results)
        results_df = self._run_batch_evaluations(results_df)
        
        if self.config.get('write_to_bigquery', False) and self.bigquery_client:
            self._write_results_to_bigquery(results_df)
        
        return results_df

    def _run_single_experiment(self, experiment: Dict[str, Any], audio_files: List[str]):
        logger.info(f"Starting experiment: {experiment['name']}")
        llm_client = self._init_llm_client(experiment['client'])
        prompt = self.prompt_manager.load(experiment['prompt_id'])
        metrics_to_run = self._get_experiment_metrics(experiment)
        
        max_workers = self.config.get('concurrency', {}).get('max_workers', 4)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self._process_single_audio, experiment, llm_client, metrics_to_run, audio_file, prompt)
                for audio_file in audio_files
            ]
            for future in futures:
                try:
                    result = future.result(timeout=300)
                    if result:
                        self.results.append(result)
                except Exception as e:
                    logger.error(f"A processing task failed and was skipped: {e}", exc_info=True)

    def _process_single_audio(self, 
                              experiment: Dict[str, Any],
                              llm_client: BaseLLMClient,
                              metrics: List[BaseMetric],
                              audio_file: str,
                              prompt: str) -> Optional[Dict[str, Any]]:
        """Generates a response, reference, and computes individual metrics for a single audio file."""
        start_time = time.time()
        try:
            _, path = self.gcs_client.parse_gcs_uri(audio_file)
            audio_data = self.gcs_client.download_bytes(path)
            mime_type = self._get_mime_type_from_path(path)
            generation_config = experiment.get('generation_config', {})
            use_case = experiment['use_case']
            reference_text = None

            if use_case in ['summarisation', 'call_analysis']:
                ref_config = self.config['reference_generation']
                ref_prompt = self.prompt_manager.load(ref_config['prompt_id'])
                transcript_response = self.reference_client.generate_from_audio(
                    audio_data=audio_data, prompt=ref_prompt,
                    generation_config=ref_config.get('generation_config', {}), mime_type=mime_type
                )
                reference_text = transcript_response.get('response_text')
                if not reference_text:
                    logger.warning(f"Failed to generate reference transcript for {audio_file}.")

            response_data = llm_client.generate_from_audio(
                audio_data=audio_data, prompt=prompt,
                generation_config=generation_config, mime_type=mime_type
            )
            response_text = response_data.get('response_text', '')
            metadata = response_data.get('metadata', {})

            metric_scores = {}
            for metric in metrics:
                if not metric.supports_batch_evaluation():
                    if metric.is_applicable(use_case):
                        try:
                            scores = metric.compute(response=response_text, metadata=metadata, reference=reference_text)
                            metric_scores.update(scores)
                        except Exception as e:
                            logger.error(f"Error computing individual metric {metric.name} for {audio_file}: {e}", exc_info=True)
                            metric_scores[f"{metric.name}_error"] = 1.0
            
            return {
                'experiment_name': experiment['name'],
                'model_id': experiment['client']['model_id'],
                'use_case': use_case,
                'audio_file': audio_file,
                'prompt': prompt,
                'response': response_text,
                'reference': reference_text,
                'metadata': metadata,
                'experiment_metrics': experiment.get('metrics', []),
                'input_tokens': metadata.get('input_tokens'),
                'output_tokens': metadata.get('output_tokens'),
                'total_tokens': metadata.get('total_tokens'),
                'processing_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat(),
                **metric_scores
            }
        except Exception as e:
            logger.error(f"Error processing {audio_file} for experiment {experiment['name']}: {e}", exc_info=True)
            return None

    def _run_batch_evaluations(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Runs all applicable batch-supporting metrics on the results."""
        if results_df.empty:
            return results_df
        
        logger.info("Running batch evaluations for supported metrics...")
        
        all_exp_metrics = set(m for exp_metrics in results_df['experiment_metrics'] for m in exp_metrics)
        needs_vertex_eval = any(m in POINTWISE_METRIC_MAP for m in all_exp_metrics)

        if needs_vertex_eval:
            vertex_eval_metric = self.available_metrics.get("vertexai_evaluation")
            if vertex_eval_metric:
                try:
                    logger.info("Running batch evaluation for: VertexAIEvaluationMetric")
                    results_df = vertex_eval_metric.batch_compute(results_df)
                except Exception as e:
                    logger.error(f"Error in batch evaluation for VertexAIEvaluationMetric: {e}", exc_info=True)
                    results_df["vertexai_evaluation_batch_error"] = 1.0
        
        return results_df

    def _get_experiment_metrics(self, experiment: Dict[str, Any]) -> List[BaseMetric]:
        """Gets the metric instances for a specific experiment based on config."""
        metric_names = experiment.get('metrics', [])
        metrics_to_run = []
        
        # Check for specific metric classes
        for name in metric_names:
            if name in self.available_metrics:
                metrics_to_run.append(self.available_metrics[name])

        # Check if any of the metrics are Vertex AI pointwise templates
        if any(name in POINTWISE_METRIC_MAP for name in metric_names):
            if "vertexai_evaluation" not in [m.name for m in metrics_to_run]:
                 metrics_to_run.append(self.available_metrics["vertexai_evaluation"])

        return metrics_to_run

    def _write_results_to_bigquery(self, results_df: pd.DataFrame) -> None:
        """
        Write experiment results to BigQuery
        
        Args:
            results_df: DataFrame containing experiment results
        """
        try:
            bigquery_config = self.config.get('bigquery', {})
            dataset_name = bigquery_config.get('dataset_name', 'llm_experiments')
            table_name = bigquery_config.get('table_name', 'experiment_results')
            
            logger.info(f"Writing {len(results_df)} results to BigQuery: {dataset_name}.{table_name}")
            
            # Ensure dataset exists
            self.bigquery_client.create_dataset(dataset_name, exists_ok=True)
            
            # Prepare data for BigQuery
            # Convert datetime columns to strings and handle complex data types
            df_copy = results_df.copy()
            
            # Convert timestamp column if exists
            if 'timestamp' in df_copy.columns:
                df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
            
            # Convert metadata column to JSON string if it exists
            if 'metadata' in df_copy.columns:
                df_copy['metadata'] = df_copy['metadata'].astype(str)
            
            self.bigquery_client.insert_rows_from_dataframe(
                dataset_name=dataset_name,
                table_name=table_name,
                df=df_copy,
                ignore_unknown_values=True,
                skip_invalid_rows=False
            )
            
            logger.info(f"Successfully wrote {len(df_copy)} rows to BigQuery")
            
        except Exception as e:
            logger.error(f"Error writing results to BigQuery: {e}")
            raise

    def _get_audio_files(self) -> List[str]:
        """Get list of audio files from GCS"""
        try:
            patterns = self.config.get('gcs_files', [])
            audio_files = self.gcs_client.list_audio_files(patterns)
            
            # Validate audio files
            valid_files = []
            for file_uri in audio_files:
                _, path = self.gcs_client.parse_gcs_uri(file_uri)
                metadata = self.gcs_client.get_file_metadata(path)
                
                if 'error' not in metadata:
                    valid_files.append(file_uri)
                else:
                    logger.warning(f"Skipping invalid audio file: {file_uri}")
            
            return valid_files
            
        except Exception as e:
            logger.error(f"Error getting audio files: {e}")
            return []
    
    def _get_mime_type_from_path(self, path: str) -> str:
        """Get MIME type from file path"""
        extension = path.lower().split('.')[-1]
        mime_types = {
            'wav': 'audio/wav',
            'mp3': 'audio/mpeg',
            'm4a': 'audio/mp4'
        }
        return mime_types.get(extension, 'audio/wav')

    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get a summary of the current experiment run"""
        if not self.results:
            return {"message": "No results available"}
        
        df = pd.DataFrame(self.results)
        
        summary = {
            "run_timestamp": self.run_timestamp,
            "total_experiments": len(df['experiment_name'].unique()),
            "total_audio_files": len(df['audio_file'].unique()),
            "total_results": len(df),
            "experiments": list(df['experiment_name'].unique()),
            "models_tested": list(df['model_id'].unique()),
            "avg_processing_time": df['processing_time'].mean(),
            "total_processing_time": df['processing_time'].sum()
        }
        
        # Add metric summaries
        metric_columns = [col for col in df.columns if col.startswith(('transcript_', 'safety_', 'vertexai_'))]
        for metric_col in metric_columns:
            if df[metric_col].dtype in ['float64', 'int64']:
                summary[f"{metric_col}_avg"] = df[metric_col].mean()
        
        return summary
