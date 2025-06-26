"""VertexAI client for Model Armour and Evaluation services"""

import logging
import requests
from typing import Dict, Any
from google.auth import default
from google.auth.transport.requests import Request

from utils.retry import RetryableClient, retry_with_backoff

logger = logging.getLogger(__name__)


class VertexAIClient(RetryableClient):
    """Client for interacting with VertexAI services including Model Armour"""

    def __init__(self, project_id: str, location: str = "europe-west2", **retry_kwargs):
        """
        Initialise VertexAI client
        
        Args:
            project_id: GCP project ID
            location: GCP location
            **retry_kwargs: Additional arguments for retry configuration
        """
        super().__init__(**retry_kwargs)
        self.project_id = project_id
        self.location = location
        
        self.credentials, _ = default()
        
        self.model_armour_base_url = f"https://modelarmor.{location}.rep.googleapis.com/v1"
        
        logger.info(f"Initialised VertexAI client for project: {project_id}, location: {location}")

    def _get_access_token(self) -> str:
        """Get fresh access token"""
        if not self.credentials.valid:
            self.credentials.refresh(Request())
        return self.credentials.token

    @retry_with_backoff(max_attempts=3)
    def sanitise_model_response(self, response_text: str, template_id: str) -> Dict[str, Any]:
        """
        Sanitise model response using Model Armour
        
        Args:
            response_text: The model response to sanitize
            template_id: Model Armour template ID
            
        Returns:
            Dictionary containing sanitisation results
        """
        try:
            url = f"{self.model_armour_base_url}/projects/{self.project_id}/locations/{self.location}/templates/{template_id}:sanitizeModelResponse"
            
            headers = {
                "Authorization": f"Bearer {self._get_access_token()}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model_response_data": {
                    "text": response_text
                }
            }
            
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"Successfully sanitised response with template: {template_id}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error sanitising model response: {e}")
            raise
