"""Audio file loading and processing utilities"""

import logging
from typing import List, Optional, Dict, Any
from pathlib import Path
import mimetypes

logger = logging.getLogger(__name__)


class AudioLoader:
    """Utility class for loading and validating audio files"""
    
    SUPPORTED_FORMATS = {
        '.wav': 'audio/wav',
        '.mp3': 'audio/mpeg', 
        '.m4a': 'audio/mp4',
        '.flac': 'audio/flac',
        '.ogg': 'audio/ogg'
    }
    
    def __init__(self):
        pass
    
    def validate_audio_file(self, file_path: str) -> bool:
        """
        Validate that a file is a supported audio format
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            True if the file is a valid audio file
        """
        try:
            path = Path(file_path)
            
            # Check if file exists
            if not path.exists():
                logger.error(f"Audio file does not exist: {file_path}")
                return False
            
            # Check file extension
            extension = path.suffix.lower()
            if extension not in self.SUPPORTED_FORMATS:
                logger.error(f"Unsupported audio format: {extension}")
                return False
            
            # Check MIME type
            mime_type, _ = mimetypes.guess_type(file_path)
            expected_mime = self.SUPPORTED_FORMATS[extension]
            
            if mime_type and not mime_type.startswith('audio/'):
                logger.error(f"File is not an audio file: {mime_type}")
                return False
            
            # Basic file size check (audio files should be > 1KB)
            if path.stat().st_size < 1024:
                logger.warning(f"Audio file seems too small: {path.stat().st_size} bytes")
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating audio file {file_path}: {e}")
            return False
    
    def get_audio_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Get basic metadata about an audio file
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Dictionary with file metadata
        """
        try:
            path = Path(file_path)
            
            metadata = {
                'file_path': str(path.absolute()),
                'file_name': path.name,
                'file_size_bytes': path.stat().st_size,
                'file_extension': path.suffix.lower(),
                'mime_type': mimetypes.guess_type(file_path)[0]
            }
            
            # Try to get duration using basic file inspection
            # For more detailed audio analysis, you might want to use libraries like librosa or pydub
            # but keeping dependencies minimal for now
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error getting audio metadata for {file_path}: {e}")
            return {'error': str(e)}
    
    def prepare_audio_for_api(self, file_path: str) -> Optional[bytes]:
        """
        Prepare audio file data for API submission
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Audio file bytes if successful, None otherwise
        """
        try:
            if not self.validate_audio_file(file_path):
                return None
            
            with open(file_path, 'rb') as f:
                audio_data = f.read()
            
            logger.info(f"Loaded audio file: {file_path} ({len(audio_data)} bytes)")
            return audio_data
            
        except Exception as e:
            logger.error(f"Error loading audio file {file_path}: {e}")
            return None
    
    def batch_validate_audio_files(self, file_paths: List[str]) -> Dict[str, bool]:
        """
        Validate multiple audio files
        
        Args:
            file_paths: List of audio file paths
            
        Returns:
            Dictionary mapping file paths to validation results
        """
        results = {}
        
        for file_path in file_paths:
            results[file_path] = self.validate_audio_file(file_path)
        
        valid_count = sum(results.values())
        logger.info(f"Validated {len(file_paths)} audio files: {valid_count} valid, {len(file_paths) - valid_count} invalid")
        
        return results
