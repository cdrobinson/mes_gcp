"""Gemini client for LLM inference"""

import logging
import time
from typing import Dict, Any
import google.genai as genai
from google.genai import types

from utils.retry import RetryableClient, retry_with_backoff
from clients.base_client import BaseLLMClient

logger = logging.getLogger(__name__)


class GeminiClient(BaseLLMClient, RetryableClient):
    """Client for interacting with Gemini models"""

    def __init__(self, model_id: str, config: Dict[str, str] = None, **retry_kwargs):
        """
        Initialise Gemini client

        Args:
            model_id: The Gemini model to use
            config: Dictionary with 'project_id' and 'location'
            **retry_kwargs: Additional arguments for retry configuration
        """
        super().__init__(**retry_kwargs)
        self.model_id = model_id
        if config is None:
            raise ValueError("config dictionary with 'project_id' and 'location' is required")
        self.client = genai.Client(
            vertexai=True,
            project=config["project_id"],
            location=config["location"]
        )
        logger.info(f"Initialised Gemini client with model: {model_id}")

    @retry_with_backoff(max_attempts=3)
    def generate_from_audio(self,
                           audio_data: bytes,
                           prompt: str,
                           generation_config: Dict[str, Any],
                           mime_type: str = "audio/wav") -> Dict[str, Any]:
        """
        Generate content from audio using Gemini

        Args:
            audio_data: The audio file bytes
            prompt: The text prompt to send with the audio
            generation_config: Generation configuration parameters
            mime_type: MIME type of the audio file

        Returns:
            Dictionary containing the response and metadata
        """
        start_time = time.time()
        try:
            gen_config = types.GenerateContentConfig(
                temperature=generation_config.get("temperature", 0.2),
                top_k=generation_config.get("top_k", 40),
                top_p=generation_config.get("top_p", 0.8),
                max_output_tokens=generation_config.get("max_output_tokens", 4096),
                )

            audio_part = types.Part.from_bytes(
                    data=audio_data,
                    mime_type=mime_type
                )

            content = [
                prompt,
                audio_part
            ]
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=content,
                config=gen_config
            )

            end_time = time.time()
            response_text = response.text if response.text else ""
            metadata = self._extract_response_metadata(response, start_time, end_time)
            logger.info(f"Generated response in {metadata['latency_seconds']:.2f}s")
            return {
                "response_text": response_text,
                "metadata": metadata,
                "raw_response": response
            }
        except Exception as e:
            end_time = time.time()
            logger.error(f"Error generating content from audio: {e}")
            return {
                "response_text": "",
                "metadata": {
                    "error": str(e),
                    "latency_seconds": end_time - start_time,
                    "model_id": self.model_id,
                    "timestamp": start_time
                },
                "raw_response": None
            }

    @retry_with_backoff(max_attempts=3)
    def generate_from_text(self,
                           prompt: str,
                           generation_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate content from text using Gemini

        Args:
            prompt: The text prompt
            generation_config: Generation configuration parameters

        Returns:
            Dictionary containing the response and metadata
        """
        try:
            gen_config = types.GenerateContentConfig(
                temperature=generation_config.get("temperature", 0.2),
                top_k=generation_config.get("top_k", 40),
                top_p=generation_config.get("top_p", 0.8),
                max_output_tokens=generation_config.get("max_output_tokens", 4096),
            )

            content = [prompt]

            start_time = time.time()
            
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=content,
                config=gen_config
            )

            end_time = time.time()
            response_text = response.text if response.text else ""
            metadata = self._extract_response_metadata(response, start_time, end_time)
            logger.info(f"Generated text response in {metadata['latency_seconds']:.2f}s")
            return {
                "response_text": response_text,
                "metadata": metadata,
                "raw_response": response
            }
        except Exception as e:
            end_time = time.time()
            logger.error(f"Error generating content from text: {e}")
            return {
                "response_text": "",
                "metadata": {
                    "error": str(e),
                    "latency_seconds": end_time - start_time,
                    "model_id": self.model_id,
                    "timestamp": start_time
                },
                "raw_response": None
            }

    def _extract_response_metadata(self, response, start_time: float, end_time: float) -> Dict[str, Any]:
        """
        Extract metadata from the Gemini response

        Args:
            response: The Gemini response object
            start_time: Request start timestamp
            end_time: Request end timestamp

        Returns:
            Dictionary with extracted metadata
        """
        metadata = {
            "model_id": self.model_id,
            "latency_seconds": end_time - start_time,
            "timestamp": start_time
        }
        try:
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage = response.usage_metadata
                metadata["input_tokens"] = getattr(usage, 'prompt_token_count', None)
                metadata["output_tokens"] = getattr(usage, 'candidates_token_count', None)
                metadata["total_tokens"] = getattr(usage, 'total_token_count', None)
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                metadata["avg_logprobs"] = getattr(candidate, 'avg_logprobs', None)
        except Exception as e:
            logger.warning(f"Error extracting response metadata: {e}")
            metadata["metadata_extraction_error"] = str(e)
        return metadata
