"""Experiment orchestrator for LLM evaluations"""

import json
import logging
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
from metrics.summarisation.summarisation_quality import SummarisationQualityMetric
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
        self.transcript_gcs_client = self._init_transcript_gcs_client()
        
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
            "summarisation_quality": SummarisationQualityMetric(),
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
        self.reference_cache = {}
        logger.info(f"ExperimentRunner initialised with {len(self.available_metrics)} available metrics")

    def _init_llm_client(self, client_config: Dict[str, Any]) -> BaseLLMClient:
        """
        Initialise an LLM client based on the provided configuration.

        Args:
            client_config: Dictionary containing client configuration.

        Returns:
            An instance of BaseLLMClient.
        """
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
        """
        Load the experiment configuration from a YAML file.

        Returns:
            Dictionary containing the loaded configuration.
        """
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        required_keys = ['project', 'location', 'bucket', 'experiments', 'reference_generation']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Required configuration key missing: {key}")
        logger.info(f"Loaded configuration with {len(config['experiments'])} experiments")
        return config
    
    def _init_gcs_client(self) -> GCSClient:
        """
        Initialise the GCS client using the configured bucket.

        Returns:
            An instance of GCSClient.
        """
        return GCSClient(bucket_name=self.config['bucket'])

    def _init_transcript_gcs_client(self) -> GCSClient:
        """
        Initialise the GCS client for transcript storage.

        Returns:
            An instance of GCSClient for the transcript storage bucket.
        """
        transcript_bucket = self.config.get('transcript_storage', {}).get('bucket', 'mes-experiment-data')
        return GCSClient(bucket_name=transcript_bucket)

    def run_experiments(self, experiment_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Run all or selected experiments as defined in the configuration.

        Args:
            experiment_names: Optional list of experiment names to run.

        Returns:
            DataFrame containing the results of the experiments.
        """
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
        """
        Run a single experiment on a list of audio files.

        Args:
            experiment: Dictionary containing experiment configuration.
            audio_files: List of audio file URIs to process.
        """
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
                    result = future.result(timeout=600)
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
        """
        Generates a response, reference, and computes individual metrics for a single audio file.

        Args:
            experiment: Dictionary containing experiment configuration.
            llm_client: The LLM client to use for generation.
            metrics: List of metric instances to compute.
            audio_file: URI of the audio file to process.
            prompt: Prompt string to use for generation.

        Returns:
            Dictionary with results for the audio file, or None if processing failed.
        """
        try:
            _, path = self.gcs_client.parse_gcs_uri(audio_file)
            audio_data = self.gcs_client.download_bytes(path)
            mime_type = self._get_mime_type_from_path(path)
            generation_config = experiment.get('generation_config', {})
            use_case = experiment['use_case']
            reference_text = None

            if use_case in ['summarisation', 'call_analysis']:
                # Check memory cache first
                if audio_file in self.reference_cache:
                    reference_text = self.reference_cache[audio_file]
                    logger.debug(f"Using cached reference transcript for {audio_file}")
                else:
                    # Check GCS transcript storage
                    reference_text = self._get_transcript_from_storage(audio_file)
                    
                    if reference_text:
                        self.reference_cache[audio_file] = reference_text
                        logger.debug(f"Using stored transcript for {audio_file}")
                    else:
                        ref_config = self.config['reference_generation']
                        ref_prompt = self.prompt_manager.load(ref_config['prompt_id'])
                        transcript_response = self.reference_client.generate_from_audio(
                            audio_data=audio_data, prompt=ref_prompt,
                            generation_config=ref_config.get('generation_config', {}), mime_type=mime_type
                        )
                        reference_text = transcript_response.get('response_text')
                        if reference_text:
                            self.reference_cache[audio_file] = reference_text
                            self._store_transcript(audio_file, reference_text)
                            logger.debug(f"Generated and stored new transcript for {audio_file}")
                        else:
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
                            scores = metric.compute(response=response_text, metadata=metadata)
                            metric_scores.update(scores)
                        except Exception as e:
                            logger.error(f"Error computing individual metric {metric.name} for {audio_file}: {e}", exc_info=True)
                            metric_scores[f"{metric.name}_error"] = 1.0
            
            return {
                'experiment_name': experiment['name'],
                'model_id': experiment['client']['model_id'],
                'use_case': use_case,
                'audio_file': audio_file,
                'prompt_id': experiment['prompt_id'],
                'prompt': prompt,
                'response': response_text,
                'reference': reference_text,
                'metadata': metadata,
                'experiment_metrics': experiment.get('metrics', []),
                'input_tokens': metadata.get('input_tokens'),
                'output_tokens': metadata.get('output_tokens'),
                'total_tokens': metadata.get('total_tokens'),
                'processing_time': metadata.get('latency_seconds', 0.0),
                'temperature': generation_config.get('temperature'),
                'top_k': generation_config.get('top_k'),
                'top_p': generation_config.get('top_p'),
                'timestamp': datetime.now().isoformat(),
                **metric_scores
            }
        except Exception as e:
            logger.error(f"Error processing {audio_file} for experiment {experiment['name']}: {e}", exc_info=True)
            return None

    def _run_batch_evaluations(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Runs all applicable batch-supporting metrics on the results.

        Args:
            results_df: DataFrame containing experiment results.

        Returns:
            DataFrame with batch metric results added.
        """
        if results_df.empty:
            return results_df
        
        logger.info("Running batch evaluations for supported metrics...")
        
        all_exp_metrics = set(m for exp_metrics in results_df['experiment_metrics'] for m in exp_metrics)
        needs_vertex_eval = any(m.replace('vertexai_', '').upper() in POINTWISE_METRIC_MAP for m in all_exp_metrics)

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
        """
        Gets the metric instances for a specific experiment based on config.

        Args:
            experiment: Dictionary containing experiment configuration.

        Returns:
            List of metric instances to run for the experiment.
        """
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

    def write_results_to_bigquery(self, results_df: pd.DataFrame) -> None:
        """
        Upload experiment results to BigQuery.
        Assumes the table was created with `create_experiment_results_table`.
        """
        bigquery_cfg = self.config.get("bigquery", {})
        dataset_name = bigquery_cfg.get("dataset_name", "llm_experiments")
        table_name   = bigquery_cfg.get("table_name",   "experiment_results")

        try:
            df = results_df.copy()

            df.columns = [c.replace("/", "_") for c in df.columns]

            # Convert timestamp column to pandas datetime
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

            for col in ("metadata", "experiment_metrics"):
                if col in df.columns:
                    df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, (dict, list)) else str(x))

            self.bigquery_client.insert_rows_from_dataframe(
                dataset_name=dataset_name,
                table_name=table_name,
                df=df,
                ignore_unknown_values=False,   # safer
                skip_invalid_rows=False,
            )
            logger.info("Successfully wrote %s rows to %s.%s", len(df), dataset_name, table_name)

        except Exception as exc:
            logger.exception("Failed to write results to BigQuery")
            raise

    def _get_audio_files(self) -> List[str]:
        """
        Get list of audio files from GCS.

        Returns:
            List of valid audio file URIs.
        """
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
        """
        Get MIME type from file path.

        Args:
            path: File path string.

        Returns:
            MIME type string.
        """
        extension = path.lower().split('.')[-1]
        mime_types = {
            'wav': 'audio/wav',
            'mp3': 'audio/mpeg',
            'm4a': 'audio/mp4'
        }
        return mime_types.get(extension, 'audio/wav')

    def get_experiment_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current experiment run.

        Returns:
            Dictionary containing summary statistics for the run.
        """
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

    def _get_transcript_path(self, audio_file: str) -> str:
        """
        Generate the transcript file path for a given audio file.
        
        Args:
            audio_file: URI of the audio file
            
        Returns:
            Path to the transcript file in the transcript storage bucket
        """
        _, audio_path = self.gcs_client.parse_gcs_uri(audio_file)
        audio_filename = audio_path.split('/')[-1]
        base_name = audio_filename.rsplit('.', 1)[0]
        
        transcript_folder = self.config.get('transcript_storage', {}).get('folder', 'transcripts')
        return f"{transcript_folder}/{base_name}.txt"

    def _get_transcript_from_storage(self, audio_file: str) -> Optional[str]:
        """
        Retrieve transcript from GCS storage if it exists.
        
        Args:
            audio_file: URI of the audio file
            
        Returns:
            Transcript text if found, None otherwise
        """
        try:
            transcript_path = self._get_transcript_path(audio_file)
            transcript_bytes = self.transcript_gcs_client.download_bytes(transcript_path)
            transcript_text = transcript_bytes.decode('utf-8')
            logger.debug(f"Retrieved existing transcript for {audio_file} from {transcript_path}")
            return transcript_text
        except Exception as e:
            logger.debug(f"No existing transcript found for {audio_file}: {e}")
            return None

    def _store_transcript(self, audio_file: str, transcript_text: str) -> None:
        """
        Store transcript to GCS storage.
        
        Args:
            audio_file: URI of the audio file
            transcript_text: The transcript text to store
        """
        try:
            import tempfile
            import os
            
            transcript_path = self._get_transcript_path(audio_file)
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
                tmp_file.write(transcript_text)
                tmp_file_path = tmp_file.name
            
            try:
                self.transcript_gcs_client.upload_file(tmp_file_path, transcript_path)
                logger.debug(f"Stored transcript for {audio_file} to {transcript_path}")
            finally:
                os.unlink(tmp_file_path)  
        except Exception as e:
            logger.error(f"Failed to store transcript for {audio_file}: {e}")
