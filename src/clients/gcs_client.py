"""Google Cloud Storage client for audio file management"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import fnmatch
from google.cloud import storage
from google.cloud.exceptions import NotFound, Forbidden

from utils.retry import RetryableClient, retry_with_backoff

logger = logging.getLogger(__name__)


class GCSClient(RetryableClient):
    """Client for interacting with Google Cloud Storage"""
    
    def __init__(self, bucket_name: str, **retry_kwargs):
        """
        Initialise GCS client
        
        Args:
            bucket_name: Name of the GCS bucket
            **retry_kwargs: Additional arguments for retry configuration
        """
        super().__init__(**retry_kwargs)
        self.bucket_name = bucket_name

        self.client = storage.Client()
        
        self.bucket = self.client.bucket(bucket_name)
    
    @retry_with_backoff(max_attempts=3)
    def list_audio_files(self, patterns: List[str]) -> List[str]:
        """
        List audio files in the bucket matching given patterns
        
        Args:
            patterns: List of glob patterns to match (e.g., ['calls/claim_*.wav'])
            
        Returns:
            List of GCS URIs for matching files
        """
        matching_files = []
        
        try:
            blobs = self.client.list_blobs(self.bucket)
            
            for blob in blobs:
                blob_name = blob.name
                
                # Check if the blob matches any of the patterns
                for pattern in patterns:
                    if fnmatch.fnmatch(blob_name, pattern):
                        # Construct GCS URI
                        gcs_uri = f"gs://{self.bucket_name}/{blob_name}"
                        matching_files.append(gcs_uri)
                        break
            
            logger.info(f"Found {len(matching_files)} audio files matching patterns: {patterns}")
            return matching_files
            
        except Exception as e:
            logger.error(f"Error listing audio files: {e}")
            raise
    
    @retry_with_backoff(max_attempts=3)
    def upload_file(self, local_path: str, gcs_path: str) -> str:
        """
        Upload a local file to GCS
        
        Args:
            local_path: Path to the local file
            gcs_path: Destination path in GCS (without gs:// prefix)
            
        Returns:
            GCS URI of the uploaded file
        """
        try:
            blob = self.bucket.blob(gcs_path)
            
            with open(local_path, 'rb') as file_obj:
                blob.upload_from_file(file_obj)
            
            gcs_uri = f"gs://{self.bucket_name}/{gcs_path}"
            logger.info(f"Uploaded {local_path} to {gcs_uri}")
            return gcs_uri
            
        except Exception as e:
            logger.error(f"Error uploading file {local_path} to {gcs_path}: {e}")
            raise
    
    @retry_with_backoff(max_attempts=3) 
    def download_file(self, gcs_path: str, local_path: str) -> str:
        """
        Download a file from GCS to local storage
        
        Args:
            gcs_path: Path in GCS (without gs:// prefix)
            local_path: Local destination path
            
        Returns:
            Path to the downloaded file
        """
        try:
            blob = self.bucket.blob(gcs_path)
            
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            
            blob.download_to_filename(local_path)
            
            logger.info(f"Downloaded {gcs_path} to {local_path}")
            return local_path
            
        except Exception as e:
            logger.error(f"Error downloading file {gcs_path} to {local_path}: {e}")
            raise
    
    @retry_with_backoff(max_attempts=3)
    def download_bytes(self, gcs_path: str) -> bytes:
        """
        Download a file from GCS and return its contents as bytes

        Args:
            gcs_path: Path in GCS (without gs:// prefix)

        Returns:
            File contents as bytes
        """
        try:
            blob = self.bucket.blob(gcs_path)
            data = blob.download_as_bytes()
            logger.info(f"Downloaded {gcs_path} as bytes")
            return data
        except Exception as e:
            logger.error(f"Error downloading file {gcs_path} as bytes: {e}")
            raise

    def get_file_metadata(self, gcs_path: str) -> Dict[str, Any]:
        """
        Get metadata for a file in GCS
        
        Args:
            gcs_path: Path in GCS (without gs:// prefix)
            
        Returns:
            Dictionary with file metadata
        """
        try:
            blob = self.bucket.blob(gcs_path)
            blob.reload()
            
            return {
                'name': blob.name,
                'size': blob.size,
                'created': blob.time_created,
                'updated': blob.updated,
                'content_type': blob.content_type,
                'md5_hash': blob.md5_hash,
                'gcs_uri': f"gs://{self.bucket_name}/{gcs_path}"
            }
            
        except Exception as e:
            logger.error(f"Error getting metadata for {gcs_path}: {e}")
            return {'error': str(e)}
    
    @staticmethod
    def parse_gcs_uri(gcs_uri: str) -> tuple[str, str]:
        """
        Parse a GCS URI into bucket and path components
        
        Args:
            gcs_uri: GCS URI in format gs://bucket/path
            
        Returns:
            Tuple of (bucket_name, path)
        """
        if not gcs_uri.startswith('gs://'):
            raise ValueError(f"Invalid GCS URI: {gcs_uri}")
        
        # Remove gs:// prefix and split
        path_part = gcs_uri[5:]  # Remove 'gs://'
        bucket_name, path = path_part.split('/', 1)
        
        return bucket_name, path
