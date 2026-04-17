"""Speech recognition and speaker diarization pipeline."""

from .pipeline import SpeechPipeline
from .models import TranscriptionResult, SpeakerSegment
from .config import Config

__version__ = "0.1.0"
__all__ = ["SpeechPipeline", "TranscriptionResult", "SpeakerSegment", "Config"]