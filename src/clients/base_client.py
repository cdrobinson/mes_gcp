"""Base client for LLM inference"""

from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseLLMClient(ABC):
    """Abstract base class for LLM clients"""

    @abstractmethod
    def generate_from_text(self, prompt: str, generation_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate content from text"""
        pass

    @abstractmethod
    def generate_from_audio(self, audio_data: bytes, prompt: str, generation_config: Dict[str, Any], mime_type: str) -> Dict[str, Any]:
        """Generate content from audio"""
        pass
