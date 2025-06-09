from abc import ABC, abstractmethod
import time
import uuid
from typing import Dict, Any, Tuple, Optional
from core.gcp_client import (
    get_gemini_model,
    get_gemini_generation_config,
    config as global_main_config
)
from core.retry_handler import default_retry_decorator
from vertexai.generative_models import Part, GenerationConfig, GenerativeModel

class BaseAgent(ABC):
    """Abstract Base Class for GenAI Agents."""

    def __init__(self,
                 agent_type: str,
                 prompt_template: str,
                 system_prompt: Optional[str] = None,
                 llm_config_name: Optional[str] = None):
        self.agent_type = agent_type
        self.prompt_template = prompt_template
        self.system_prompt = system_prompt

        self.update_llm_configuration(llm_config_name)


    def update_llm_configuration(self, llm_config_name: Optional[str] = None,
                                 generation_config_override: Optional[Dict[str, Any]] = None):
        """
        Updates the LLM model and its generation configuration.

        Args:
            llm_config_name (Optional[str]): The name of the LLM configuration from main_config.yaml.
                                            If None, uses the default.
            generation_config_override (Optional[Dict[str, Any]]): Specific generation parameters to override.
        """
        # Determine the LLM configuration to use
        if llm_config_name:
            self.llm_config_name = llm_config_name
        else:
            # Fallback to default if no specific name is provided during agent init or update
            self.llm_config_name = global_main_config['gcp']['vertex_ai']['default_llm_config_name']

        # Fetch the base model and generation config for the chosen llm_config_name
        self.gemini_model: GenerativeModel = get_gemini_model(self.llm_config_name)
        base_gen_config_dict = get_gemini_generation_config(self.llm_config_name).to_dict()

        # Apply overrides if any
        if generation_config_override:
            base_gen_config_dict.update(generation_config_override)

        self.gemini_gen_config: GenerationConfig = GenerationConfig(**base_gen_config_dict)
        
        # Store the actual model name and full generation config used for metadata
        self.current_llm_model_name = self.gemini_model._model_name
        self.current_llm_generation_config_dict = self.gemini_gen_config.to_dict()


    @abstractmethod
    def _transcribe_audio(self, gcs_uri: str) -> Tuple[str, float, float, Dict[str, Any]]:
        """
        Transcribes audio from GCS.
        Subclasses MUST override this to provide a transcription implementation
        (e.g., using Gemini, Speech-to-Text API, or another service).
        """
        pass

    @default_retry_decorator()
    def _call_llm(self, transcript: str) -> Tuple[str, float, Dict[str, Any]]:
        """Calls the configured LLM with the provided transcript and prompt."""
        llm_start_time = time.time()

        prompt = self.prompt_template.format(transcript=transcript)
        contents = []

        if self.system_prompt:
            contents.append(Part.from_text(self.system_prompt))
        contents.append(Part.from_text(prompt))

        response = self.gemini_model.generate_content(
            contents=contents,
            generation_config=self.gemini_gen_config
        )

        llm_end_time = time.time()
        llm_processing_time = llm_end_time - llm_start_time

        llm_output = ""
        if response.candidates:
            for part in response.candidates[0].content.parts:
                if part.text:
                    llm_output += part.text

        usage_metadata = response.usage_metadata if hasattr(response, 'usage_metadata') else None
        llm_metadata = {
            "llm_config_name": self.llm_config_name, # Track which named config was used
            "actual_model_name": self.current_llm_model_name,
            "generation_config_used": self.current_llm_generation_config_dict, # Log the actual config
            "prompt_token_count": usage_metadata.prompt_token_count if usage_metadata else 0,
            "candidates_token_count": usage_metadata.candidates_token_count if usage_metadata else 0,
            "total_token_count": usage_metadata.total_token_count if usage_metadata else 0,
        }
        return llm_output.strip(), llm_processing_time, llm_metadata

    @abstractmethod
    def process(self, gcs_audio_path: str) -> Dict[str, Any]:
        """Processes the audio file: transcribes and then calls the LLM."""
        pass

    def _common_process_steps(self, gcs_audio_path: str) -> Dict[str, Any]:
        """Helper for common processing steps."""
        run_id = str(uuid.uuid4())
        print(f"Processing {gcs_audio_path} for agent {self.agent_type} (LLM config: {self.llm_config_name}) with run_id {run_id}")

        transcript, stt_time, audio_duration, stt_meta = self._transcribe_audio(gcs_audio_path) # MODIFIED: Calls self._transcribe_audio

        if not transcript:
            print(f"Transcription failed or returned empty for {gcs_audio_path}")
            return {
                "run_id": run_id,
                "gcs_audio_path": gcs_audio_path,
                "agent_type": self.agent_type,
                "transcript": "",
                "llm_output": "Error: Transcription failed or empty.",
                "stt_processing_time_seconds": stt_time,
                "llm_processing_time_seconds": 0,
                "audio_duration_seconds": audio_duration,
                "error": "Transcription failed or empty.",
                "stt_metadata": stt_meta,
                "llm_metadata": {"llm_config_name": self.llm_config_name, "error": "No LLM call due to transcription failure."}
            }

        llm_output, llm_time, llm_meta = self._call_llm(transcript)

        return {
            "run_id": run_id,
            "gcs_audio_path": gcs_audio_path,
            "agent_type": self.agent_type,
            "transcript": transcript,
            "llm_output": llm_output,
            "stt_processing_time_seconds": round(stt_time, 3),
            "llm_processing_time_seconds": round(llm_time, 3),
            "audio_duration_seconds": round(audio_duration, 2) if audio_duration is not None else None,
            "stt_metadata": stt_meta,
            "llm_metadata": llm_meta
        }