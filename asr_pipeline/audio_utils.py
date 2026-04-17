"""Audio processing utilities."""

import logging
from pathlib import Path
from typing import Tuple, Optional
import librosa
import soundfile as sf
import numpy as np
from pydub import AudioSegment


logger = logging.getLogger(__name__)


class AudioProcessor:
    """Handles audio loading, preprocessing, and segmentation."""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
    
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file and convert to the specified sample rate.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        audio_file = Path(audio_path)
        
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        try:
            # Try loading with librosa first (handles most formats)
            audio_data, sr = librosa.load(
                audio_path,
                sr=self.sample_rate,
                mono=True
            )
            effective_sr = int(sr)
            logger.info(f"Loaded audio: {audio_file.name} ({len(audio_data)/effective_sr:.2f}s)")
            return audio_data, effective_sr
            
        except Exception as e:
            logger.warning(f"librosa failed to load {audio_path}: {e}")
            
            try:
                # Fallback to pydub for other formats
                audio_segment = AudioSegment.from_file(audio_path)
                audio_segment = audio_segment.set_frame_rate(self.sample_rate).set_channels(1)
                
                # Convert to numpy array
                audio_data = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
                audio_data = audio_data / (2**15)  # Normalize to [-1, 1]
                
                logger.info(f"Loaded audio with pydub: {audio_file.name} ({len(audio_data)/self.sample_rate:.2f}s)")
                return audio_data, self.sample_rate
                
            except Exception as e2:
                raise RuntimeError(f"Failed to load audio file {audio_path}: {e2}")
    
    def extract_segment(
        self,
        audio_data: np.ndarray,
        start_time: float,
        end_time: float,
        sample_rate: int
    ) -> np.ndarray:
        """
        Extract a segment from audio data.
        
        Args:
            audio_data: Audio data array
            start_time: Start time in seconds
            end_time: End time in seconds
            sample_rate: Sample rate of the audio
            
        Returns:
            Audio segment as numpy array
        """
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        
        # Ensure we don't go out of bounds
        start_sample = max(0, start_sample)
        end_sample = min(len(audio_data), end_sample)
        
        return audio_data[start_sample:end_sample]
    
    def save_segment(
        self,
        audio_segment: np.ndarray,
        output_path: str,
        sample_rate: int
    ) -> None:
        """
        Save audio segment to file.
        
        Args:
            audio_segment: Audio data to save
            output_path: Output file path
            sample_rate: Sample rate of the audio
        """
        sf.write(output_path, audio_segment, sample_rate)
        logger.debug(f"Saved audio segment: {output_path}")
    
    def get_audio_duration(self, audio_path: str) -> float:
        """
        Get duration of audio file in seconds.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Duration in seconds
        """
        try:
            duration = librosa.get_duration(path=audio_path)
            return duration
        except Exception:
            # Fallback to pydub
            audio_segment = AudioSegment.from_file(audio_path)
            return len(audio_segment) / 1000.0
    
    def preprocess_for_whisper(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Preprocess audio data for Whisper model.
        
        Args:
            audio_data: Raw audio data
            
        Returns:
            Preprocessed audio data
        """
        # Whisper expects audio normalized to [-1, 1]
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Normalize if needed
        if np.max(np.abs(audio_data)) > 1.0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        return audio_data
    
    def validate_audio_format(self, audio_path: str) -> bool:
        """
        Validate if audio file can be processed.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            True if valid, False otherwise
        """
        try:
            audio_file = Path(audio_path)
            
            # Check if file exists
            if not audio_file.exists():
                return False
            
            # Check file extension
            valid_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.aac', '.wma'}
            if audio_file.suffix.lower() not in valid_extensions:
                logger.warning(f"Unsupported audio format: {audio_file.suffix}")
                return False
            
            # Try to get duration (quick validation)
            duration = self.get_audio_duration(audio_path)
            if duration <= 0:
                return False
            
            logger.info(f"Audio validation passed: {audio_file.name} ({duration:.2f}s)")
            return True
            
        except Exception as e:
            logger.error(f"Audio validation failed: {e}")
            return False