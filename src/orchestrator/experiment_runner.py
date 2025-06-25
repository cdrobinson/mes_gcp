"""Experiment orchestrator for LLM evaluations"""

import logging
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import yaml

from clients.gcs_client import GCSClient
from clients.gemini_client import GeminiClient
from clients.bigquery_client import BigQueryClient
from utils.audio_loader import AudioLoader
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
        self.audio_loader = AudioLoader()
        
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
                location=location
            )
        }
        
        # Results storage
        self.results = []
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info(f"ExperimentRunner initialised with {len(self.available_metrics)} available metrics")

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
        results_df = self._run_batch_evaluations(results_df, experiment_metrics)
        
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
        
        gemini_client = GeminiClient(
            model_id=experiment['model_id'],
            config={'project_id': self.config['project'], 'location': self.config['location']},
            max_attempts=self.config.get('retry', {}).get('max_attempts', 3)
        )
        
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
                        experiment, gemini_client, experiment_metrics,
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
                            gemini_client: GeminiClient,
                            metrics: List[BaseMetric],
                            audio_file: str,
                            prompt: str) -> Optional[Dict[str, Any]]:
        """Process a single audio file through the experiment pipeline"""
        start_time = time.time()
        
        try:
            _, path = self.gcs_client.parse_gcs_uri(audio_file)
            audio_data = self.gcs_client.download_bytes(path)
            
            # Get MIME type from file extension
            mime_type = self._get_mime_type_from_path(path)
            
            # Generate response using Gemini
            generation_config = experiment['generation_config']
            llm_response = gemini_client.generate_from_audio(
                audio_data=audio_data,
                prompt=prompt,
                generation_config=generation_config,
                mime_type=mime_type
            )
            
            response_text = llm_response['response_text']
            response_metadata = llm_response['metadata']
            
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
                            metadata=response_metadata,
                        )
                        metric_scores.update(scores)
                    except Exception as e:
                        logger.error(f"Error computing {metric.name}: {e}")
                        metric_scores[f"{metric.name}_error"] = 1.0
            
            result = {
                'experiment_name': experiment['name'],
                'model_id': experiment['model_id'],
                'use_case': experiment['use_case'],
                'audio_file': audio_file,
                'response_text': response_text,
                'metadata': response_metadata,  # Store metadata for batch evaluation
                'input_tokens': response_metadata['input_tokens'],
                'output_tokens': response_metadata['output_tokens'],
                'total_tokens': response_metadata['total_tokens'],
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

    def _run_batch_evaluations(self, results_df: pd.DataFrame, experiment_metrics) -> pd.DataFrame:
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
        
        # Identify metrics that support batch evaluation
        batch_metrics = {}
        individual_metrics = {}
        
        for metric_name, metric in self.available_metrics.items():
            if metric_name not in experiment_metrics:
                continue
            if hasattr(metric, 'supports_batch_evaluation') and metric.supports_batch_evaluation():
                batch_metrics[metric_name] = metric
            else:
                individual_metrics[metric_name] = metric
        
        logger.info(f"Found {len(batch_metrics)} batch metrics and {len(individual_metrics)} individual metrics")
        
        for metric_name, metric in batch_metrics.items():
            try:
                logger.info(f"Running batch evaluation for: {metric_name}")
                results_df = metric.batch_compute(results_df)
            except Exception as e:
                logger.error(f"Error in batch evaluation for {metric_name}: {e}")
                # Add error column
                results_df[f"{metric_name}_batch_error"] = 1.0
        
        return results_df

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
