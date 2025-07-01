"""Experiment orchestrator for LLM evaluations"""

import logging
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
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
from metrics.evaluation.vertexai_evaluation import VertexAIEvaluationMetric

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
        
        bigquery_config = self.config.get('bigquery', {})
        self.bigquery_client = BigQueryClient(
            project_id=bigquery_config.get('project_id', project_id),
            location=bigquery_config.get('location', location)
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
                location=location,
                bucket_name=self.config['bucket']
            )
        }
        
        # Results storage
        self.results = []
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info(f"ExperimentRunner initialised with {len(self.available_metrics)} available metrics")

    def _init_llm_client(self, client_config: Dict[str, Any]) -> BaseLLMClient:
        """Initialise LLM client based on config"""
        client_name = client_config.get('name')
        model_id = client_config.get('model_id')
        
        if not client_name:
            raise ValueError("LLM client name not specified in experiment config")

        if client_name == 'gemini':
            return GeminiClient(
                model_id=model_id,
                config={'project_id': self.config['project'], 'location': self.config['location']},
                max_attempts=self.config.get('retry', {}).get('max_attempts', 3)
            )
        # Example for a future ModelGardenClient
        # if client_name == 'model_garden':
        #     return ModelGardenClient(endpoint_id=client_config.get('endpoint_id'))
        
        raise ValueError(f"Unknown LLM client: {client_name}")

    def _load_config(self) -> Dict[str, Any]:
        """Load and validate the YAML configuration"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Basic validation
            required_keys = ['project', 'location', 'bucket', 'experiments']
            for key in required_keys:
                if key not in config:
                    raise ValueError(f"Required configuration key missing: {key}")
            
            logger.info(f"Loaded configuration with {len(config['experiments'])} experiments")
            return config
            
        except Exception as e:
            logger.error(f"Error loading configuration from {self.config_path}: {e}")
            raise
    
    def _init_gcs_client(self) -> GCSClient:
        """Initialise Google Cloud Storage client"""
        try:
            return GCSClient(
                bucket_name=self.config['bucket']
            )
        except Exception as e:
            logger.error(f"Error initialising GCS client: {e}")
            raise

    def run_experiments(self, experiment_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Run the specified experiments
        
        Args:
            experiment_names: List of experiment names to run (None = run all)
            
        Returns:
            DataFrame with experiment results
        """
        experiments_to_run = self.config['experiments']
        
        if experiment_names:
            experiments_to_run = [
                exp for exp in experiments_to_run 
                if exp['name'] in experiment_names
            ]
        
        logger.info(f"Running {len(experiments_to_run)} experiments")

        experiment_metrics = {
            metric_name
            for exp in experiments_to_run
            for metric_name in exp.get('metrics', [])
        }
        
        # Get audio files
        audio_files = self._get_audio_files()
        logger.info(f"Processing {len(audio_files)} audio files")
        
        for experiment in experiments_to_run:
            self._run_single_experiment(experiment, audio_files)
        
        if not self.results:
            logger.warning("No results to process")
            return pd.DataFrame()
        
        results_df = pd.DataFrame(self.results)
        
        # Run batch evaluation for metrics that support it
        results_df = self._run_batch_evaluations(results_df)
        
        if self.config.get('write_to_bigquery', False) and self.bigquery_client:
            self._write_results_to_bigquery(results_df)
        
        return results_df

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
    
    def _run_single_experiment(self, experiment: Dict[str, Any], audio_files: List[str]):
        """Run a single experiment on all audio files"""
        experiment_name = experiment['name']
        logger.info(f"Starting experiment: {experiment_name}")
        
        llm_client = self._init_llm_client(experiment['client'])
        
        experiment_metrics = self._get_experiment_metrics(experiment)
        
        prompt = self._get_prompt(experiment)
        
        concurrency_config = self.config.get('concurrency', {})
        max_workers = concurrency_config.get('max_workers', 4)
        batch_size = concurrency_config.get('batch_size', 8)
        
        for batch_start in range(0, len(audio_files), batch_size):
            batch_end = min(batch_start + batch_size, len(audio_files))
            batch_files = audio_files[batch_start:batch_end]
            
            logger.info(f"Processing batch {batch_start//batch_size + 1}: files {batch_start+1}-{batch_end}")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                
                for audio_file in batch_files:
                    future = executor.submit(
                        self._process_single_audio,
                        experiment, llm_client, experiment_metrics,
                        audio_file, prompt
                    )
                    futures.append((future, audio_file))
                
                for future, audio_file in futures:
                    try:
                        result = future.result(timeout=300)  # 5 minute timeout
                        if result:
                            self.results.append(result)
                            logger.info(f"Completed processing: {audio_file}")
                    except Exception as e:
                        logger.error(f"Error processing {audio_file}: {e}")
        
        logger.info(f"Completed experiment: {experiment_name}")

    def _get_mime_type_from_path(self, path: str) -> str:
        """Get MIME type from file path"""
        extension = path.lower().split('.')[-1]
        mime_types = {
            'wav': 'audio/wav',
            'mp3': 'audio/mpeg',
            'm4a': 'audio/mp4'
        }
        return mime_types.get(extension, 'audio/wav')

    def _get_experiment_metrics(self, experiment: Dict[str, Any]) -> List[BaseMetric]:
        """Get metrics for a specific experiment"""
        metric_names = experiment.get('metrics', [])
        experiment_metrics = []
        
        for metric_name in metric_names:
            if metric_name in self.available_metrics:
                metric = self.available_metrics[metric_name]
                experiment_metrics.append(metric)
            else:
                logger.warning(f"Metric not found: {metric_name}")
        
        return experiment_metrics

    def _get_prompt(self, experiment: Dict[str, Any]) -> str:
        """Retrieve the prompt text for an experiment via Vertex AI Prompt Management service"""
        prompt_id = experiment['prompt_id']

        try:
            return self.prompt_manager.load(prompt_id)
        except Exception as e:
            raise ValueError(f"Could not load prompt '{prompt_id}': {e}")

    def _process_single_audio(self, 
                            experiment: Dict[str, Any],
                            llm_client: BaseLLMClient,
                            metrics: List[BaseMetric],
                            audio_file: str,
                            prompt: str) -> Optional[Dict[str, Any]]:
        """Process a single audio file through the experiment pipeline"""
        start_time = time.time()
        
        try:
            _, path = self.gcs_client.parse_gcs_uri(audio_file)
            audio_data = self.gcs_client.download_bytes(path)

            mime_type = self._get_mime_type_from_path(path)

            generation_config = experiment.get('generation_config', {})
            
            response_data = llm_client.generate_from_audio(
                audio_data=audio_data,
                prompt=prompt,
                generation_config=generation_config,
                mime_type=mime_type
            )
            
            response_text = response_data.get('response_text', '')
            metadata = response_data.get('metadata', {})
            
            # Compute metrics (skip batch metrics here, they'll be computed later)
            metric_scores = {}
            for metric in metrics:
                if metric.is_applicable(experiment['use_case']):
                    # Skip metrics that support batch evaluation - they'll be computed later
                    if hasattr(metric, 'supports_batch_evaluation') and metric.supports_batch_evaluation():
                        logger.debug(f"Skipping {metric.name} - will be computed in batch")
                        continue
                        
                    try:
                        scores = metric.compute(
                            response=response_text,
                            metadata=metadata,
                        )
                        metric_scores.update(scores)
                    except Exception as e:
                        logger.error(f"Error computing {metric.name}: {e}")
                        metric_scores[f"{metric.name}_error"] = 1.0
            
            result = {
                'experiment_name': experiment['name'],
                'model_id': experiment['client']['model_id'],
                'use_case': experiment['use_case'],
                'audio_file': audio_file,
                'response_text': response_text,
                'metadata': metadata,  # Store metadata for batch evaluation
                'input_tokens': metadata['input_tokens'],
                'output_tokens': metadata['output_tokens'],
                'total_tokens': metadata['total_tokens'],
                'processing_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat(),
                **metric_scores
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing {audio_file}: {e}")
            return {
                'experiment_name': experiment['name'],
                'audio_file': audio_file,
                'error': str(e),
                'processing_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }

    def _run_batch_evaluations(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Run batch evaluations for metrics that support it
        
        Args:
            results_df: DataFrame with initial experiment results
            
        Returns:
            DataFrame with additional evaluation metrics
        """
        if results_df.empty:
            return results_df
        
        logger.info("Running batch evaluations for supported metrics")
        all_results = []

        for experiment_name, group in results_df.groupby('experiment_name'):
            experiment_config = self._get_experiment_config(experiment_name)
            if not experiment_config:
                all_results.append(group)
                continue
                
            group = self._apply_batch_metrics(group, experiment_config)
            all_results.append(group)

        return pd.concat(all_results) if all_results else pd.DataFrame()

    def _get_experiment_config(self, experiment_name: str) -> Optional[Dict[str, Any]]:
        """
        Get experiment configuration by name
        
        Args:
            experiment_name: Name of the experiment
            
        Returns:
            Experiment configuration dictionary or None if not found
        """
        experiment_config = next(
            (exp for exp in self.config['experiments'] if exp['name'] == experiment_name), 
            None
        )
        if not experiment_config:
            logger.warning(f"Could not find config for experiment {experiment_name}, skipping batch evaluations.")
        return experiment_config

    def _apply_batch_metrics(self, group: pd.DataFrame, experiment_config: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply all batch metrics to a group of results from a single experiment
        
        Args:
            group: DataFrame containing results for a single experiment
            experiment_config: Configuration for the experiment
            
        Returns:
            DataFrame with batch metric results added
        """
        logger.info(f"Running batch evaluations for experiment: {experiment_config['name']}")
        
        batch_metrics = self._get_batch_metrics_for_experiment(experiment_config)
        
        for metric_name, metric in batch_metrics:
            group = self._apply_metric_safely(metric_name, metric, group, experiment_config)
            
        return group

    def _get_batch_metrics_for_experiment(self, experiment_config: Dict[str, Any]) -> List[Tuple[str, BaseMetric]]:
        """
        Get list of batch metrics that should be applied to an experiment
        
        Args:
            experiment_config: Configuration for the experiment
            
        Returns:
            List of (metric_name, metric_instance) tuples for batch metrics
        """
        experiment_metrics = experiment_config.get('metrics', [])
        batch_metrics = []
        
        for metric_name in experiment_metrics:
            if metric_name not in self.available_metrics:
                continue
                
            metric = self.available_metrics[metric_name]
            if hasattr(metric, 'supports_batch_evaluation') and metric.supports_batch_evaluation():
                batch_metrics.append((metric_name, metric))
        
        return batch_metrics

    def _apply_metric_safely(self, metric_name: str, metric: BaseMetric, group: pd.DataFrame, experiment_config: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply a single batch metric with error handling
        
        Args:
            metric_name: Name of the metric
            metric: The metric instance
            group: DataFrame containing results for a single experiment
            experiment_config: Configuration for the experiment
            
        Returns:
            DataFrame with metric results added or error columns if failed
        """
        try:
            logger.info(f"Running batch evaluation for: {metric_name}")
            
            # Use the new standardized interface for all metrics
            return metric.batch_compute(group, experiment_config, self.config)
                
        except Exception as e:
            logger.error(f"Error in batch evaluation for {metric_name}: {e}")
            error_col = f"{metric_name}_batch_error"
            group[error_col] = 1.0
            return group

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
