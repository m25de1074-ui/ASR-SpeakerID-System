"""Configuration management for the speech pipeline."""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv


@dataclass
class Config:
    """Configuration settings for the speech pipeline."""
    
    # API Keys and tokens
    huggingface_token: Optional[str] = None
    openai_api_key: Optional[str] = None
    
    # Model configurations
    whisper_model: str = "base"
    pyannote_model: str = "pyannote/speaker-diarization-3.1"
    
    # Processing settings
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    
    # Output settings
    output_format: str = "srt"
    speaker_labels: str = "Speaker"
    
    # Audio processing settings
    sample_rate: int = 16000
    chunk_length: float = 30.0  # seconds
    
    @classmethod
    def from_env(cls, env_file: Optional[str] = None) -> "Config":
        """Load configuration from environment variables."""
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()
        min_speakers_str = os.getenv("MIN_SPEAKERS")
        min_speakers = int(min_speakers_str) if min_speakers_str else None
        max_speakers_str = os.getenv("MAX_SPEAKERS")
        max_speakers = int(max_speakers_str) if max_speakers_str else None
        return cls(
            huggingface_token=os.getenv("HUGGINGFACE_TOKEN"),
            openai_api_key=os.getenv("OPENAI_API_KEY", None),
            whisper_model=os.getenv("WHISPER_MODEL", "base"),
            pyannote_model=os.getenv("PYANNOTE_MODEL", "pyannote/speaker-diarization-3.1"),
            output_format=os.getenv("OUTPUT_FORMAT", "srt"),
            speaker_labels=os.getenv("SPEAKER_LABELS", "Speaker"),
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
    
    def validate(self) -> None:
        """Validate configuration settings."""
        if not self.huggingface_token:
            raise ValueError(
                "Hugging Face token is required. Please set HUGGINGFACE_TOKEN "
                "in your .env file or environment variables."
            )
        
        valid_whisper_models = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
        if self.whisper_model not in valid_whisper_models:
            raise ValueError(
                f"Invalid Whisper model: {self.whisper_model}. "
                f"Valid models: {', '.join(valid_whisper_models)}"
            )
        
        valid_formats = ["srt", "vtt", "json"]
        if self.output_format not in valid_formats:
            raise ValueError(
                f"Invalid output format: {self.output_format}. "
                f"Valid formats: {', '.join(valid_formats)}"
            )