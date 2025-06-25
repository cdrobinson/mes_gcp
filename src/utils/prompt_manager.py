""" Prompt management utilities using Vertex AI Prompt Management service. """

import vertexai
from vertexai.preview import prompts
from vertexai.preview.prompts import Prompt

class PromptManager:
    """
    Manages Vertex AI PromptTemplate lifecycle:
      - initialise with project & location
      - create, load, update prompt versions
    """

    def __init__(self, project: str, location: str):
        """
        Args:
           project: GCP project ID
           locationL GCP region
        """
        self.project = project
        self.location = location
        vertexai.init(project=project, location=location)

    def create(self, prompt_name: str, prompt_text: str) -> str:
        """
        Create a new PromptTemplate resource in Vertex AI.
        
        Args:
            prompt_name: Display name for the prompt
            prompt_text: Raw prompt content
            
        Returns:
            The prompt_id of the created prompt resource
        """
        local_prompt = Prompt(
            prompt_name=prompt_name,
            prompt_data=prompt_text,
            model_name="gemini-2.0-flash-001" # required field
        )

        created = prompts.create_version(prompt=local_prompt)
        return created.prompt_id
    
    def load(self, prompt_id: str) -> str:
        """
        Load a saved PromptTemplate resource.
        
        Args:
            prompt_id: The ID of the prompt resource to load

        Returns:
            The prompt text (prompt_data) of the saved resource
        """
        prompt = prompts.get(prompt_id=prompt_id)
        return prompt.prompt_data
    
    def update(self, prompt_id: str, prompt_text: str) -> str:
        """
        Create a new version of an existing PromptTemplate with updated text

        Args:
            prompt_id: The ID of the prompt resource to update
            prompt_text: The new prompt content

        Returns:
            The prompt_id of the updated prompt resource
        """
        prompt = prompts.get(prompt_id=prompt_id)

        prompt.prompt_data = prompt_text

        new_version = prompts.create_version(prompt=prompt)
        return new_version.prompt_id