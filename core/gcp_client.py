import os
from functools import lru_cache
import yaml
from google.cloud import aiplatform
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech
from google.cloud import bigquery
from google.cloud import storage
from vertexai.generative_models import GenerativeModel, GenerationConfig
from typing import Dict, Any

# --- Configuration Loading ---
def get_config_path():
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'main_config.yaml')

@lru_cache(maxsize=None)
def load_config():
    """Loads the main configuration file."""
    config_path = get_config_path()
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config()

@lru_cache(maxsize=None)
def get_vertex_ai_client():
    """Initializes and returns the Vertex AI client."""
    aiplatform.init(
        project=config['gcp']['project_id'],
        location=config['gcp']['location']
    )
    return aiplatform

def get_llm_config(llm_config_name: str) -> Dict[str, Any]:
    """Fetches a specific LLM configuration by name."""
    if llm_config_name is None:
        llm_config_name = config['gcp']['vertex_ai']['default_llm_config_name']

    llm_conf = config['gcp']['vertex_ai']['llm_configurations'].get(llm_config_name)
    if not llm_conf:
        raise ValueError(f"LLM configuration '{llm_config_name}' not found in main_config.yaml.")
    return llm_conf

@lru_cache(maxsize=10)
def get_gemini_model(llm_config_name: str) -> GenerativeModel:
    """Initializes and returns a Gemini model instance based on the named configuration."""
    get_vertex_ai_client()
    specific_llm_config = get_llm_config(llm_config_name)
    model_name = specific_llm_config['gemini_model_name']
    model = GenerativeModel(model_name)
    return model

def get_gemini_generation_config(llm_config_name: str) -> GenerationConfig:
    """Returns the generation configuration for the specified Gemini LLM config."""
    specific_llm_config = get_llm_config(llm_config_name)
    return GenerationConfig(**specific_llm_config['generation_config'])


@lru_cache(maxsize=None)
def get_speech_to_text_client() -> SpeechClient:
    """Initializes and returns the Speech-to-Text client."""
    return SpeechClient()

def get_speech_to_text_config() -> cloud_speech.RecognitionConfig:
    """Builds and returns the Speech-to-Text recognition configuration."""
    stt_config = config['gcp']['speech_to_text']
    features_config = stt_config.get('features', {})

    recognition_features = cloud_speech.RecognitionFeatures()
    if features_config.get('enable_automatic_punctuation'):
        recognition_features.enable_automatic_punctuation = True
    if features_config.get('enable_word_time_offsets'):
        recognition_features.enable_word_time_offsets = True

    diarization_config_dict = features_config.get('diarization_config')
    if diarization_config_dict and diarization_config_dict.get('enable_speaker_diarization'):
        recognition_features.diarization_config = cloud_speech.SpeakerDiarizationConfig(
            min_speaker_count=diarization_config_dict.get('min_speaker_count', 1),
            max_speaker_count=diarization_config_dict.get('max_speaker_count', 2)
        )

    return cloud_speech.RecognitionConfig(
        auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
        model=stt_config.get('model', 'telephony'),
        language_codes=stt_config.get('language_codes', ['en-US']),
        features=recognition_features
    )

def get_recognizer_path(project_id: str, location: str, recognizer_id: str) -> str:
    """Constructs the full recognizer path for Speech-to-Text v2."""
    return f"projects/{project_id}/locations/{location}/recognizers/{recognizer_id}"


@lru_cache(maxsize=None)
def get_bigquery_client() -> bigquery.Client:
    """Initializes and returns the BigQuery client."""
    return bigquery.Client(project=config['gcp']['project_id'])

@lru_cache(maxsize=None)
def get_storage_client() -> storage.Client:
    """Initializes and returns the Cloud Storage client."""
    return storage.Client(project=config['gcp']['project_id'])

# --- Helper functions to get specific config values ---
def get_gcp_project_id() -> str:
    return config['gcp']['project_id']

def get_gcp_location() -> str:
    return config['gcp']['location']

def get_bq_dataset_id() -> str:
    return config['gcp']['bigquery']['dataset_id']

def get_bq_summarisation_table_id() -> str:
    return config['gcp']['bigquery']['summarisation_table_id']

def get_bq_analysis_table_id() -> str:
    return config['gcp']['bigquery']['analysis_table_id']

def get_gcs_audio_bucket_name() -> str:
    return config['gcp']['gcs']['audio_data_bucket']

def get_gcs_audio_dataset_paths() -> dict:
    return config['gcp']['gcs']['audio_dataset_paths']

def get_retry_config() -> dict:
    return config.get('retry_settings', {
        'attempts': 3, 'wait_initial': 1, 'wait_multiplier': 2, 'wait_max': 10
    })