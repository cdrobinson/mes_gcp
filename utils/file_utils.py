from core.gcp_client import get_storage_client
from typing import List

def list_gcs_audio_files(bucket_name: str, folder_path: str) -> List[str]:
    """
    Lists all audio files (common extensions) in a specified GCS folder.
    """
    storage_client = get_storage_client()
    bucket = storage_client.bucket(bucket_name)
    prefix = folder_path
    if prefix and not prefix.endswith('/'):
        prefix += '/'
    if prefix == '/':
        prefix = ""

    blobs = bucket.list_blobs(prefix=prefix)
    audio_files = []
    audio_extensions = ('.wav', '.mp3', '.flac', '.ogg', '.aac', '.m4a', '.opus', '.raw', '.amr')

    for blob in blobs:
        if blob.name.endswith(audio_extensions) and not blob.name.endswith('/'):
            audio_files.append(f"gs://{bucket_name}/{blob.name}")
    return audio_files