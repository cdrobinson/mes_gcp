from typing import Dict, Any, Tuple, Optional
import time
import uuid

from agents.base_agent import BaseAgent
from core.gcp_client import get_gemini_model, get_gemini_generation_config, config as global_main_config
from vertexai.generative_models import Part, GenerationConfig, GenerativeModel, Audio

class GeminiTranscriptionAgent(BaseAgent):
    """Agent that uses Gemini for transcription."""

    def __init__(self,
                 agent_type: str = "gemini_transcription_agent",
                 prompt_template: str = "Transcribe the following audio: {transcript}",
                 system_prompt: Optional[str] = "You are a helpful assistant that transcribes audio.",
                 llm_config_name: Optional[str] = None,
                 transcription_llm_config_name: Optional[str] = "gemini_1_5_flash_transcription"
                 ):
        super().__init__(agent_type, prompt_template, system_prompt, llm_config_name)
        self.transcription_llm_config_name = transcription_llm_config_name
        self.transcription_gemini_model: GenerativeModel = get_gemini_model(self.transcription_llm_config_name)
        self.transcription_gemini_gen_config: GenerationConfig = get_gemini_generation_config(self.transcription_llm_config_name)
        self.current_transcription_llm_model_name = self.transcription_gemini_model._model_name
        self.current_transcription_llm_generation_config_dict = self.transcription_gemini_gen_config.to_dict()


    def _transcribe_audio(self, gcs_uri: str) -> Tuple[str, float, float, Dict[str, Any]]:
        """
        Transcribes audio from GCS using a specified Gemini model.
        Overrides the base class method.
        """
        stt_start_time = time.time()
        print(f"Starting Gemini transcription for {gcs_uri} using model {self.current_transcription_llm_model_name}")

        audio_part = Part.from_uri(gcs_uri, mime_type="audio/wav")

        contents = [
            "Please transcribe the following audio recording accurately.",
            audio_part
        ]

        try:
            response = self.transcription_gemini_model.generate_content(
                contents=contents,
                generation_config=self.transcription_gemini_gen_config # Use transcription-specific config
            )
            full_transcript = ""
            if response.candidates:
                for part in response.candidates[0].content.parts:
                    if part.text:
                        full_transcript += part.text
            full_transcript = full_transcript.strip()

        except Exception as e:
            print(f"Error during Gemini transcription for {gcs_uri}: {e}")
            full_transcript = "" # Ensure empty transcript on error

        stt_end_time = time.time()
        stt_processing_time = stt_end_time - stt_start_time

        transcription_metadata = {
            "stt_model_used": self.current_transcription_llm_model_name,
            "generation_config_used": self.current_transcription_llm_generation_config_dict,
            "recognizer_used": "Gemini API",
        }

        if not full_transcript:
            print(f"Gemini transcription returned empty for {gcs_uri}")

        return full_transcript, stt_processing_time, transcription_metadata

    def process(self, gcs_audio_path: str) -> Dict[str, Any]:
        """
        Processes the audio file: transcribes using Gemini and then calls the LLM
        (which could be another Gemini model or a different one as configured in BaseAgent).
        """
        return self._common_process_steps(gcs_audio_path)


    def process_transcription_only(self, gcs_audio_path: str) -> Dict[str, Any]:
        """Processes the audio file: transcribes using Gemini and returns."""
        run_id = str(uuid.uuid4())
        print(f"Processing transcription for {gcs_audio_path} with agent {self.agent_type} (Run ID: {run_id})")

        transcript, stt_time, stt_meta = self._transcribe_audio(gcs_audio_path)

        if not transcript:
            print(f"Transcription failed or returned empty for {gcs_audio_path}")
            return {
                "run_id": run_id,
                "gcs_audio_path": gcs_audio_path,
                "agent_type": self.agent_type,
                "transcript": "",
                "stt_processing_time_seconds": round(stt_time, 3),
                "error": "Transcription failed or empty.",
                "stt_metadata": stt_meta,
            }

        return {
            "run_id": run_id,
            "gcs_audio_path": gcs_audio_path,
            "agent_type": self.agent_type,
            "transcript": transcript,
            "stt_processing_time_seconds": round(stt_time, 3),
            "stt_metadata": stt_meta,
        }