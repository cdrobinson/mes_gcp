"""Simplified experiment orchestrator for transcription LLM evaluations"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import yaml

from clients.gcs_client import GCSClient
from clients.gemini_client import GeminiClient
from utils.audio_loader import AudioLoader
from metrics.base_metric import BaseMetric
from metrics.transcription.transcript_quality import TranscriptQualityMetric
from metrics.safety.safety import SafetyMetric

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Simplified orchestrator for transcription LLM evaluation experiments"""
    
    def __init__(self, config_path: str):
        """
        Initialize the experiment runner
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        self.gcs_client = self._init_gcs_client()
        self.audio_loader = AudioLoader()
        
        self.available_metrics = {
            "transcript_quality": TranscriptQualityMetric(),
            "safety": SafetyMetric()
        }
        
        # Results storage
        self.results = []
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info(f"ExperimentRunner initialized with {len(self.available_metrics)} available metrics")
    
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
        """Initialize Google Cloud Storage client"""
        try:
            return GCSClient(
                bucket_name=self.config['bucket'],
                project_id=self.config['project']
            )
        except Exception as e:
            logger.error(f"Error initializing GCS client: {e}")
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
        
        # Get audio files
        audio_files = self._get_audio_files()
        logger.info(f"Processing {len(audio_files)} audio files")
        
        # Run experiments
        for experiment in experiments_to_run:
            self._run_single_experiment(experiment, audio_files)
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(self.results)
        
        return results_df
    
    def _get_audio_files(self) -> List[str]:
        """Get list of audio files from GCS"""
        try:
            patterns = self.config.get('gcs_files', [])
            audio_files = self.gcs_client.list_audio_files(patterns)
            
            # Validate audio files
            valid_files = []
            for file_uri in audio_files:
                bucket, path = self.gcs_client.parse_gcs_uri(file_uri)
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
        
        # Get metrics for this experiment
        experiment_metrics = self._get_experiment_metrics(experiment)
        
        # Get the prompt for this experiment
        prompt = self._get_prompt(experiment)
        
        # Set up concurrency (keep batch processing as requested)
        concurrency_config = self.config.get('concurrency', {})
        max_workers = concurrency_config.get('max_workers', 4)
        batch_size = concurrency_config.get('batch_size', 8)
        
        # Process audio files in batches
        for batch_start in range(0, len(audio_files), batch_size):
            batch_end = min(batch_start + batch_size, len(audio_files))
            batch_files = audio_files[batch_start:batch_end]
            
            logger.info(f"Processing batch {batch_start//batch_size + 1}: files {batch_start+1}-{batch_end}")
            
            # Process batch with thread pool
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                
                for audio_file in batch_files:
                    future = executor.submit(
                        self._process_single_audio,
                        experiment, gemini_client, experiment_metrics,
                        audio_file, prompt
                    )
                    futures.append((future, audio_file))
                
                # Collect results
                for future, audio_file in futures:
                    try:
                        result = future.result(timeout=300)  # 5 minute timeout
                        if result:
                            self.results.append(result)
                            logger.info(f"Completed processing: {audio_file}")
                    except Exception as e:
                        logger.error(f"Error processing {audio_file}: {e}")
        
        logger.info(f"Completed experiment: {experiment_name}")

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
        """Get the prompt template for an experiment"""
        use_case = experiment['use_case']
        prompt_id = experiment['prompt_id']
        
        prompts = self.config.get('prompts', {})
        if use_case in prompts and prompt_id in prompts[use_case]:
            return prompts[use_case][prompt_id]
        else:
            raise ValueError(f"Prompt not found: {use_case}/{prompt_id}")

    def _process_single_audio(self, 
                            experiment: Dict[str, Any],
                            gemini_client: GeminiClient,
                            metrics: List[BaseMetric],
                            audio_file: str,
                            prompt: str) -> Optional[Dict[str, Any]]:
        """Process a single audio file through the experiment pipeline"""
        start_time = time.time()
        
        try:
            # Download audio file
            bucket, path = self.gcs_client.parse_gcs_uri(audio_file)
            local_path = f"/tmp/{Path(path).name}"
            self.gcs_client.download_file(path, local_path)
            
            # Load audio data
            audio_data = self.audio_loader.prepare_audio_for_api(local_path)
            if not audio_data:
                logger.error(f"Failed to load audio: {audio_file}")
                return None
            
            # Get MIME type
            mime_type = self.audio_loader.get_audio_metadata(local_path).get('mime_type', 'audio/wav')
            
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
            
            # Compute metrics
            metric_scores = {}
            for metric in metrics:
                if metric.is_applicable(experiment['use_case']):
                    try:
                        scores = metric.compute(
                            response=response_text,
                            metadata=response_metadata,
                            audio_path=local_path
                        )
                        metric_scores.update(scores)
                    except Exception as e:
                        logger.error(f"Error computing {metric.name}: {e}")
                        metric_scores[f"{metric.name}_error"] = 1.0
            
            # Compile result
            result = {
                'experiment_name': experiment['name'],
                'model_id': experiment['model_id'],
                'use_case': experiment['use_case'],
                'audio_file': audio_file,
                'response_text': response_text,
                'processing_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat(),
                **response_metadata,
                **metric_scores
            }
            
            # Clean up local file
            Path(local_path).unlink(missing_ok=True)
            
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
        metric_columns = [col for col in df.columns if col.startswith(('transcript_', 'safety_'))]
        for metric_col in metric_columns:
            if df[metric_col].dtype in ['float64', 'int64']:
                summary[f"{metric_col}_avg"] = df[metric_col].mean()
        
        return summary
