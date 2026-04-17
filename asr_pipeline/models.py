"""Data models for the speech pipeline."""

import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import timedelta


@dataclass
class SpeakerSegment:
    """Represents a segment of audio with speaker and timing information."""
    
    start: float  # Start time in seconds
    end: float    # End time in seconds
    speaker: str  # Speaker identifier
    text: Optional[str] = None  # Transcribed text
    confidence: Optional[float] = None  # Confidence score
    audio_data: Optional[Any] = None  # Audio data for the segment
    
    @property
    def duration(self) -> float:
        """Duration of the segment in seconds."""
        return self.end - self.start
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert segment to dictionary."""
        return {
            "start": self.start,
            "end": self.end,
            "speaker": self.speaker,
            "text": self.text,
            "confidence": self.confidence,
            "duration": self.duration
        }


@dataclass
class TranscriptionResult:
    """Contains the complete transcription result with speaker diarization."""
    
    segments: List[SpeakerSegment]
    total_duration: float
    speakers: List[str]
    
    def to_srt(self) -> str:
        """Convert result to SRT subtitle format."""
        srt_content = []
        
        for i, segment in enumerate(self.segments, 1):
            if not segment.text:
                continue
                
            start_time = self._seconds_to_srt_time(segment.start)
            end_time = self._seconds_to_srt_time(segment.end)
            
            srt_content.append(f"{i}")
            srt_content.append(f"{start_time} --> {end_time}")
            srt_content.append(f"{segment.speaker}: {segment.text}")
            srt_content.append("")
        
        return "\n".join(srt_content)
    
    def to_vtt(self) -> str:
        """Convert result to WebVTT format."""
        vtt_content = ["WEBVTT", ""]
        
        for segment in self.segments:
            if not segment.text:
                continue
                
            start_time = self._seconds_to_vtt_time(segment.start)
            end_time = self._seconds_to_vtt_time(segment.end)
            
            vtt_content.append(f"{start_time} --> {end_time}")
            vtt_content.append(f"{segment.speaker}: {segment.text}")
            vtt_content.append("")
        
        return "\n".join(vtt_content)
    
    def to_json(self, indent: int = 2) -> str:
        """Convert result to JSON format."""
        data = {
            "total_duration": self.total_duration,
            "speakers": self.speakers,
            "segments": [segment.to_dict() for segment in self.segments]
        }
        return json.dumps(data, indent=indent)
    
    def get_speaker_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for each speaker."""
        stats = {}
        
        for speaker in self.speakers:
            speaker_segments = [s for s in self.segments if s.speaker == speaker]
            total_time = sum(s.duration for s in speaker_segments)
            word_count = sum(len(s.text.split()) if s.text else 0 for s in speaker_segments)
            
            stats[speaker] = {
                "total_time": total_time,
                "percentage": (total_time / self.total_duration) * 100,
                "segments_count": len(speaker_segments),
                "word_count": word_count,
                "average_confidence": sum(s.confidence for s in speaker_segments if s.confidence) / len(speaker_segments) if speaker_segments else 0
            }
        
        return stats
    
    @staticmethod
    def _seconds_to_srt_time(seconds: float) -> str:
        """Convert seconds to SRT time format (HH:MM:SS,mmm)."""
        td = timedelta(seconds=seconds)
        hours, remainder = divmod(td.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = int((seconds % 1) * 1000)
        
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d},{milliseconds:03d}"
    
    @staticmethod
    def _seconds_to_vtt_time(seconds: float) -> str:
        """Convert seconds to WebVTT time format (HH:MM:SS.mmm)."""
        td = timedelta(seconds=seconds)
        hours, remainder = divmod(td.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = int((seconds % 1) * 1000)
        
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}.{milliseconds:03d}"