import time
import pandas as pd
from typing import List, Dict, Any, Optional, Type
from agents.base_agent import BaseAgent
from evaluation.advanced_metrics_calculator import AdvancedMetricsCalculator
from utils.file_utils import list_gcs_audio_files
from core.gcp_clients import get_llm_config # To get full LLM config for metadata

class ExperimentRunner:
    """
    Runs experiments for a given agent type, dataset, and parameters,
    potentially iterating over multiple LLM configurations.
    """

    def __init__(self, agent_class: Type[BaseAgent],
                       use_advanced_metrics: bool = True,
                       judge_llm_config_name: Optional[str] = None,
                       include_llm_judge: bool = True,
                       include_rouge: bool = True):
        """
        Args:
            agent_class (Type[BaseAgent]): The class of the agent to instantiate (e.g., CallSummarizationAgent).
            use_advanced_metrics (bool): Whether to use advanced metrics (ROUGE + LLM-as-a-Judge)
            judge_llm_config_name (str): LLM config name for LLM-as-a-Judge evaluation
            include_llm_judge (bool): Whether to include LLM-as-a-Judge evaluation
            include_rouge (bool): Whether to include ROUGE evaluation
        """
        self.agent_class = agent_class
        self.use_advanced_metrics = use_advanced_metrics
        self.include_llm_judge = include_llm_judge
        self.include_rouge = include_rouge
        
        if use_advanced_metrics:
            self.metrics_calculator = AdvancedMetricsCalculator(judge_llm_config_name)
        else:
            # Fallback to basic metrics calculator
            from evaluation.metrics_calculator import MetricsCalculator
            self.metrics_calculator = MetricsCalculator()

    def run_experiment(self,
                       gcs_audio_folder_path: str,
                       gcs_bucket_name: str,
                       llm_config_names: List[str], # List of LLM config names from main_config.yaml
                       global_prompt_override: Optional[str] = None,
                       global_system_prompt_override: Optional[str] = None,
                       global_llm_parameters_override: Optional[Dict[str, Any]] = None
                       ) -> List[Dict[str, Any]]:
        """
        Runs the experiment for all audio files against specified LLM configurations.

        Args:
            gcs_audio_folder_path (str): GCS folder path for audio files.
            gcs_bucket_name (str): Name of the GCS bucket.
            llm_config_names (List[str]): List of LLM configuration names to test.
            global_prompt_override (Optional[str]): Optional user prompt template to apply to all LLM configs.
            global_system_prompt_override (Optional[str]): Optional system prompt to apply to all LLM configs.
            global_llm_parameters_override (Optional[Dict[str, Any]]): Optional LLM generation parameters
                                                                      to apply to all LLM configs.
        Returns:
            List[Dict[str, Any]]: A list of dictionaries with results and metrics.
        """
        all_experiment_results = []
        audio_files = list_gcs_audio_files(gcs_bucket_name, gcs_audio_folder_path)

        if not audio_files:
            print(f"No audio files found in gs://{gcs_bucket_name}/{gcs_audio_folder_path}")
            return []
        print(f"Found {len(audio_files)} audio files. Will test against {len(llm_config_names)} LLM configurations.")

        for i, gcs_uri in enumerate(audio_files):
            print(f"\nProcessing audio file {i+1}/{len(audio_files)}: {gcs_uri}")
            for llm_config_name in llm_config_names:
                print(f"  Using LLM configuration: {llm_config_name}")
                try:
                    # Instantiate agent for each LLM config to ensure fresh state
                    # Pass llm_config_name to agent constructor
                    agent = self.agent_class(
                        custom_prompt_template=global_prompt_override,
                        custom_system_prompt=global_system_prompt_override,
                        llm_config_name=llm_config_name
                    )
                    
                    # Apply global LLM parameter overrides if any
                    # This will update the agent's current LLM generation config
                    if global_llm_parameters_override:
                        agent.update_llm_configuration(
                            llm_config_name=llm_config_name, # Ensure it re-bases on the correct config
                            generation_config_override=global_llm_parameters_override
                        )


                    start_time = time.time()
                    agent_output = agent.process(gcs_uri) # Agent now uses its configured LLM
                    end_time = time.time()
                    total_processing_time = round(end_time - start_time, 3)

                    # Calculate metrics using appropriate calculator
                    if self.use_advanced_metrics:
                        all_calculated_metrics = self.metrics_calculator.calculate_comprehensive_metrics(
                            transcript=agent_output.get("transcript", ""),
                            llm_output=agent_output.get("llm_output", ""),
                            agent_type=agent_output.get("agent_type", "unknown"),
                            audio_duration_seconds=agent_output.get("audio_duration_seconds"),
                            stt_processing_time_seconds=agent_output.get("stt_processing_time_seconds"),
                            llm_processing_time_seconds=agent_output.get("llm_processing_time_seconds"),
                            llm_metadata=agent_output.get("llm_metadata"),
                            include_llm_judge=self.include_llm_judge,
                            include_rouge=self.include_rouge
                        )
                    else:
                        all_calculated_metrics = self.metrics_calculator.get_all_metrics(
                            transcript=agent_output.get("transcript", ""),
                            llm_output=agent_output.get("llm_output", ""),
                            audio_duration_seconds=agent_output.get("audio_duration_seconds"),
                            stt_processing_time_seconds=agent_output.get("stt_processing_time_seconds"),
                            llm_processing_time_seconds=agent_output.get("llm_processing_time_seconds"),
                            llm_metadata=agent_output.get("llm_metadata")
                        )

                    combined_result = {**agent_output, **all_calculated_metrics}
                    combined_result["total_file_processing_time_seconds"] = total_processing_time
                    combined_result["experiment_timestamp"] = pd.Timestamp.now(tz='UTC').isoformat()
                    
                    # Add any global overrides to the result for tracking
                    if global_llm_parameters_override:
                        combined_result["global_llm_parameters_override"] = global_llm_parameters_override
                    if global_prompt_override:
                        combined_result["global_prompt_override"] = global_prompt_override
                    if global_system_prompt_override:
                        combined_result["global_system_prompt_override"] = global_system_prompt_override
                    
                    # llm_metadata from agent_output already contains llm_config_name and gen_config_used

                    all_experiment_results.append(combined_result)
                    print(f"    Successfully processed with {llm_config_name}")

                except Exception as e:
                    print(f"    Error processing {gcs_uri} with LLM config {llm_config_name}: {e}")
                    error_result = {
                        "run_id": agent_output.get("run_id", "error_run") if 'agent_output' in locals() else "error_run_pre_agent",
                        "gcs_audio_path": gcs_uri,
                        "agent_type": agent.agent_type if 'agent' in locals() else self.agent_class.__name__,
                        "llm_config_name_attempted": llm_config_name,
                        "transcript": agent_output.get("transcript", "ERROR") if 'agent_output' in locals() else "ERROR_NO_TRANSCRIPT",
                        "llm_output": f"Error: {str(e)}",
                        "error_message": str(e),
                        "experiment_timestamp": pd.Timestamp.now(tz='UTC').isoformat()
                    }
                    all_experiment_results.append(error_result)
        return all_experiment_results