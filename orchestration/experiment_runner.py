import time
import pandas as pd
from typing import List, Dict, Any, Optional, Type
from agents.base_agent import BaseAgent
from evaluation import (
    SummaryLength, SummarySentenceCount, AverageSentenceLength, SummaryRepetitiveness,
    HedgingLanguageCount, FleschReadingEaseScore, LLMJudgeEvaluator,
    LabelProperties, ResponseLengthChars, PotentialExplanationPresence
)
from evaluation.base_metric import BaseMetric
from utils.file_utils import list_gcs_audio_files, load_config

class ExperimentRunner:
    """
    Runs experiments for a given agent type, dataset, and parameters,
    potentially iterating over multiple LLM configurations.
    """

    def __init__(self, 
                 agent_class: Type[BaseAgent],
                 ):
        """
        Args:
            agent_class (Type[BaseAgent]): The class of the agent to instantiate (e.g., CallSummarizationAgent).
        """
        self.agent_class = agent_class
        self.config = load_config() # Load global config
        self.evaluation_config = self.config.get("evaluation", {})
        self.judge_llm_config_name = self.evaluation_config.get("judge_llm_config_name")
        self.metric_params = self.evaluation_config.get("metric_parameters", {})

        # Dynamically instantiate selected metric calculators
        self.metric_calculators: Dict[str, BaseMetric] = {}
        self._initialize_metric_calculators()

    def _initialize_metric_calculators(self):
        """Initializes metric calculator instances based on the config."""
        # Determine if agent is for summarization or classification to pick the right metric list
        agent_name_lower = self.agent_class.__name__.lower()
        is_summarization_agent = "summary" in agent_name_lower or "summarization" in agent_name_lower
        is_classification_agent = "classification" in agent_name_lower or "analysis" in agent_name_lower or "label" in agent_name_lower

        metric_names_to_load: List[str] = []
        if is_summarization_agent:
            metric_names_to_load = self.evaluation_config.get("summarization_metrics_enabled", [])
        elif is_classification_agent:
            metric_names_to_load = self.evaluation_config.get("classification_metrics_enabled", [])
        else:
            print(f"Warning: Could not determine metric set for agent {self.agent_class.__name__}. No specific metrics loaded by default.")

        available_metrics_map = {
            "SummaryLength": (SummaryLength, None),
            "SummarySentenceCount": (SummarySentenceCount, None),
            "AverageSentenceLength": (AverageSentenceLength, None),
            "SummaryRepetitiveness": (SummaryRepetitiveness, None),
            "HedgingLanguageCount": (HedgingLanguageCount, None),
            "FleschReadingEaseScore": (FleschReadingEaseScore, None),
            "LLMJudge_Summarization": (LLMJudgeEvaluator, "summarization"),
            "LLMJudge_Grounde_QA": (LLMJudgeEvaluator, "summary_groundedness"),
            "LabelProperties": (LabelProperties, None),
            "ResponseLengthChars": (ResponseLengthChars, None),
            "PotentialExplanationPresence": (PotentialExplanationPresence, None),
            "LLMJudge_Classification": (LLMJudgeEvaluator, "classification"),
        }

        for metric_name_from_config in metric_names_to_load:
            if metric_name_from_config in available_metrics_map:
                MetricClass, _ = available_metrics_map[metric_name_from_config]
                # Use metric_name_from_config as the key for self.metric_calculators
                # This ensures that if LLMJudgeEvaluator is listed multiple times for different tasks,
                # it's treated as distinct metric calculations in the loop.
                
                # Get parameters specific to this metric class (e.g. SummaryRepetitiveness.min_trigram_length)
                class_specific_config_params = self.metric_params.get(MetricClass.__name__, {})

                try:
                    if MetricClass == LLMJudgeEvaluator:
                        # For LLM Judge, we use a single instance if judge_llm_config_name is the same,
                        # but we will differentiate its tasks later during the _calculate_all_metrics call.
                        # The key in self.metric_calculators will be the specific task, e.g., "LLMJudge_Summarization".
                        if metric_name_from_config not in self.metric_calculators:
                             self.metric_calculators[metric_name_from_config] = LLMJudgeEvaluator(judge_llm_config_name=self.judge_llm_config_name)
                    elif metric_name_from_config not in self.metric_calculators: # For non-LLM judge metrics
                        self.metric_calculators[metric_name_from_config] = MetricClass(**class_specific_config_params)
                except Exception as e:
                    print(f"Error initializing metric {metric_name_from_config} (class {MetricClass.__name__}) with params {class_specific_config_params}: {e}")            
            else:
                print(f"Warning: Metric '{metric_name_from_config}' defined in config is not recognized.")

    def _calculate_all_metrics(self, agent_output: Dict[str, Any]) -> Dict[str, Any]:
        """Calculates all configured metrics for a given agent output."""
        all_results = {}
        llm_response = agent_output.get("llm_output", "")
        transcript = agent_output.get("transcript", "") 
        # Potentially add other fields from agent_output if metrics need them, e.g. 'confidence' for LabelProperties
        # agent_confidence = agent_output.get("confidence_score") 

        # This map is needed again to link the config key (metric_name_from_config) to the judge_task_type
        metric_to_judge_task_map = {
            "LLMJudge_Summarization": "summarization",
            "LLMJudge_Grounde_QA": "summary_groundedness",
            "LLMJudge_Classification": "classification",
        }

        for metric_key_from_config, calculator_instance in self.metric_calculators.items():
            calculate_kwargs = {} # Parameters for the .calculate() method
            
            # Get class-level parameters (e.g. for SummaryRepetitiveness from metric_params in config)
            class_name_for_params = calculator_instance.__class__.__name__
            class_specific_config_params = self.metric_params.get(class_name_for_params, {})
            calculate_kwargs.update(class_specific_config_params)

            if isinstance(calculator_instance, LLMJudgeEvaluator):
                judge_task_type = metric_to_judge_task_map.get(metric_key_from_config)
                if judge_task_type:
                    calculate_kwargs["task_type"] = judge_task_type
                    if judge_task_type == "summarization" or judge_task_type == "summary_groundedness":
                        calculate_kwargs["transcript"] = transcript
                    # Example: if classification judge needed original input text for context
                    # if judge_task_type == "classification":
                    #     calculate_kwargs["input_text"] = agent_output.get("original_input_for_classification")
                else:
                    print(f"Warning: Could not determine task_type for LLM Judge metric key '{metric_key_from_config}'. Skipping.")
                    continue
            elif isinstance(calculator_instance, LabelProperties):
                # Example: If LabelProperties needs 'allowed_labels' or 'confidence' from agent_output or experiment setup
                # This would be more robust if agent_output consistently provided these when available.
                # For now, we rely on them being in metric_params if globally set, or passed via agent_output if dynamic.
                # if agent_output.get("predicted_label_confidence") is not None:
                #    calculate_kwargs["confidence"] = agent_output.get("predicted_label_confidence")
                # if agent_output.get("applicable_allowed_labels") is not None:
                #    calculate_kwargs["allowed_labels"] = agent_output.get("applicable_allowed_labels")
                pass # Assuming metric_params in config can set allowed_labels if static

            try:
                # print(f"Calculating metric: {metric_key_from_config} with instance {calculator_instance} and kwargs {calculate_kwargs}")
                metric_result = calculator_instance.calculate(llm_response, **calculate_kwargs)
                all_results.update(metric_result)
            except Exception as e:
                print(f"Error calculating metric {metric_key_from_config} with {calculator_instance.__class__.__name__}: {e}")
                all_results[f"{metric_key_from_config}_error"] = str(e)
        return all_results

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

                    # Calculate metrics using the new dynamic system
                    all_calculated_metrics = self._calculate_all_metrics(agent_output)

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
                        "metrics_calculation_status": "SKIPPED_DUE_TO_AGENT_ERROR",
                        "experiment_timestamp": pd.Timestamp.now(tz='UTC').isoformat()
                    }
                    all_experiment_results.append(error_result)
        return all_experiment_results