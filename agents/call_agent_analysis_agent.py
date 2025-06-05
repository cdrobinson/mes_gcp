from typing import Dict, Any, Optional
from agents.base_agent import BaseAgent
import yaml
import os

def load_prompt_config():
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'prompt_config.yaml')
    if not os.path.exists(config_path):
        print(f"Warning: Prompt configuration file {config_path} not found. Using default prompts.")
        return {
            'call_agent_analysis': {
                'system_prompt': "You are an expert in analyzing call agent performance.",
                'user_prompt': "Analyze the agent's performance in this call transcript: {transcript}"
            }
        }
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

prompt_config = load_prompt_config()

class CallAgentAnalysisAgent(BaseAgent):
    """Agent for call agent analysis."""

    def __init__(self,
                 custom_prompt_template: Optional[str] = None,
                 custom_system_prompt: Optional[str] = None,
                 llm_config_name: Optional[str] = None): # Added llm_config_name
        agent_type = "call_agent_analysis"

        default_prompts = prompt_config.get(agent_type, {})
        prompt_template_to_use = custom_prompt_template if custom_prompt_template is not None else default_prompts.get('user_prompt', "Analyze agent performance: {transcript}")
        system_prompt_to_use = custom_system_prompt if custom_system_prompt is not None else default_prompts.get('system_prompt', None)

        super().__init__(
            agent_type=agent_type,
            prompt_template=prompt_template_to_use,
            system_prompt=system_prompt_to_use,
            llm_config_name=llm_config_name # Pass to base
        )

    def process(self, gcs_audio_path: str) -> Dict[str, Any]:
        """Processes the audio file for call agent analysis."""
        return self._common_process_steps(gcs_audio_path)