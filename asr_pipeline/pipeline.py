"""Main speech recognition and speaker diarization pipeline."""

import logging
import math
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


def _mps_available() -> bool:
    """Return True if PyTorch MPS backend is available."""

    return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())

try:
    import torch
    # disable tf32 to suppress pyannote warnings
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    from faster_whisper import WhisperModel
    from pyannote.audio import Pipeline as PyannnotePipeline
    from pyannote.core import Annotation, Segment

    # Filter out repetitive deprecation warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
except ImportError as e:
    raise ImportError(
        "Required dependencies not installed. Please run: "
        "uv sync"
    ) from e

from .config import Config
from .models import SpeakerSegment, TranscriptionResult
from .audio_utils import AudioProcessor


logger = logging.getLogger(__name__)


class SpeechPipeline:
    """Complete pipeline for speech recognition and speaker diarization."""
    
    def __init__(
        self,
        config: Optional[Config] = None,
        whisper_model: Optional[str] = None,
        pyannote_model: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the speech pipeline.
        
        Args:
            config: Configuration object
            whisper_model: Whisper model name override
            pyannote_model: Pyannote model name override
            device: Device to run models on ('cpu', 'cuda', 'mps')
        """
        self.config = config or Config.from_env()

        # Override model names if provided
        if whisper_model:
            self.config.whisper_model = whisper_model
        if pyannote_model:
            self.config.pyannote_model = pyannote_model

        # Validate configuration
        self.config.validate()

        # Set device
        if device is None:
            if torch.cuda.is_available():
                resolved_device = "cuda"
            elif _mps_available():
                resolved_device = "mps"
            else:
                resolved_device = "cpu"
        else:
            resolved_device = device.lower()

        if resolved_device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA device requested but torch.cuda.is_available() returned False.")
        if resolved_device == "mps" and not _mps_available():
            raise ValueError("MPS device requested but torch.backends.mps.is_available() returned False.")
        if resolved_device not in {"cpu", "cuda", "mps"}:
            raise ValueError(f"Unsupported device '{resolved_device}'. Choose from 'cpu', 'cuda', or 'mps'.")

        self.device = resolved_device
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self.audio_processor = AudioProcessor(sample_rate=self.config.sample_rate)
        self.whisper_model: Optional[Any] = None
        self.diarization_pipeline: Optional[Any] = None
        
        # Load models
        self._load_models()
    
    def _load_models(self) -> None:
        """Load Whisper and pyannote models."""
        logger.info("Loading models...")
        
        # Load Faster Whisper model
        try:
            logger.info(f"Loading Faster Whisper model: {self.config.whisper_model}")
            # faster-whisper uses different device specification
            if self.device == "cuda":
                whisper_device = "cuda"
                compute_type = "float16"
            else:
                whisper_device = "cpu"
                compute_type = "int8"
                if self.device == "mps":
                    logger.info("MPS selected: Whisper will run on CPU backend (faster-whisper currently lacks native MPS support).")

            self.whisper_model = WhisperModel(
                self.config.whisper_model,
                device=whisper_device,
                compute_type=compute_type
            )
            logger.info("Faster Whisper model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load Faster Whisper model: {e}")
        
        # Load pyannote diarization pipeline
        try:
            logger.info(f"Loading pyannote model: {self.config.pyannote_model}")
            self.diarization_pipeline = PyannnotePipeline.from_pretrained(
                self.config.pyannote_model,
                use_auth_token=self.config.huggingface_token
            )
            
            # Move to device if CUDA is available
            if self.diarization_pipeline is not None and hasattr(self.diarization_pipeline, "to"):
                if self.device == "cuda":
                    self.diarization_pipeline = self.diarization_pipeline.to(torch.device("cuda"))
                elif self.device == "mps":
                    self.diarization_pipeline = self.diarization_pipeline.to(torch.device("mps"))
            
            logger.info("Pyannote model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load pyannote model: {e}")
    
    def process(
        self,
        audio_path: str,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        output_path: Optional[str] = None,
        output_format: Optional[str] = None,
        diarization_output_path: Optional[str] = "",  # disable diarization saving by default
        diarization_output_format: str = "txt"
    ) -> TranscriptionResult:
        """
        Process audio file with speaker diarization and speech recognition.
        
        Args:
            audio_path: Path to input audio file
            min_speakers: Minimum number of speakers (optional)
            max_speakers: Maximum number of speakers (optional)
            output_path: Path to save transcription output file (optional)
            output_format: Transcription output format ('srt', 'vtt', 'json')
            diarization_output_path: Directory or base path for diarization output (if None, automatically saved next to audio)
            diarization_output_format: 'txt' (default), 'rttm', or 'json'
            
        Returns:
            TranscriptionResult object
        """
        logger.info(f"Processing audio file: {audio_path}")
        
        # Validate audio file
        if not self.audio_processor.validate_audio_format(audio_path):
            raise ValueError(f"Invalid or unsupported audio file: {audio_path}")
        
        # Load audio
        total_duration = self.audio_processor.get_audio_duration(audio_path)
        
        # Perform speaker diarization
        logger.info("Performing speaker diarization...")
        diarization = self._perform_diarization(
            audio_path, min_speakers, max_speakers
        )

        # Always save diarization (default behavior) unless explicitly disabled by passing empty string
        if diarization_output_path == "":
            logger.info("Diarization saving disabled by empty path")
        else:
            # Derive default path if not provided
            if diarization_output_path is None:
                audio_file = Path(audio_path)
                diarization_output_path = str(audio_file.parent / f"{audio_file.stem}_diarization")
            try:
                logger.info(
                    "Saving diarization to %s (format=%s)",
                    diarization_output_path,
                    diarization_output_format,
                )
                self._save_diarization(
                    diarization,
                    audio_path=audio_path,
                    output_path=diarization_output_path,
                    fmt=diarization_output_format,
                )
            except Exception as e:  # pragma: no cover - non critical
                logger.warning(f"Failed to save diarization output: {e}")
        
        # Perform speech recognition over full audio
        logger.info("Performing speech recognition...")
        whisper_segments = self._transcribe_audio(audio_path)

        # Word-level speaker attribution & regrouping
        # (Falls back to segment-level merge if no word timestamps are available)
        transcribed_segments = self._word_level_diarization_merge(
            diarization, whisper_segments
        )
        if not transcribed_segments:  # Fallback safety net
            transcribed_segments = self._merge_diarization_and_transcription(
                diarization, whisper_segments
            )
        
        # Create result
        speakers = list(set(segment.speaker for segment in transcribed_segments))
        speakers.sort()  # Sort for consistent ordering
        
        result = TranscriptionResult(
            segments=transcribed_segments,
            total_duration=total_duration,
            speakers=speakers
        )
        
        # Save output if requested
        if output_path:
            self._save_result(result, output_path, output_format)
        
        logger.info(f"Processing completed. Found {len(speakers)} speakers in {total_duration:.2f}s audio")
        return result
    
    def _perform_diarization(
        self,
        audio_path: str,
        min_speakers: Optional[int],
        max_speakers: Optional[int]
    ) -> Annotation:
        """Perform speaker diarization on audio file."""
        import tempfile
        import soundfile as sf
        import numpy as np

        # Set speaker constraints
        diarization_params = {}
        if min_speakers is not None:
            diarization_params["min_speakers"] = min_speakers
        if max_speakers is not None:
            diarization_params["max_speakers"] = max_speakers
        
        # Run diarization
        if self.diarization_pipeline is None:
            raise RuntimeError("Diarization pipeline not loaded")
        # Pre-process audio to avoid chunking issues with pyannote
        # Use silence padding instead of resampling
        temp_file_path = None
        try:
            audio_info = sf.info(audio_path)
            original_sr = audio_info.samplerate
            original_duration = audio_info.duration
            original_frames = audio_info.frames
            logger.info(f"Audio file info: {original_sr}Hz, {original_duration:.2f}s, {original_frames} samples")
            
            # Detect sample rate mismatch
            mismatch = self._detect_sample_mismatch(
                audio_path=audio_path,
                claimed_sr=original_sr,
                duration=original_duration,
                frames=original_frames
            )
                        # Load audio data
            audio_data, sr = sf.read(audio_path, always_2d=False)
            
            # Ensure mono
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1)
                logger.debug("Converted stereo audio to mono")
            
            # Calculate expected samples for clean chunking
            duration = len(audio_data) / sr
            # Round duration to nearest 0.01 second to avoid fractional samples
            clean_duration = round(duration, 2)
            expected_samples = int(clean_duration * sr)
            actual_samples = len(audio_data)
            
            # Pad or trim to exact expected length
            if actual_samples < expected_samples:
                # Pad with silence at the end
                pad_samples = expected_samples - actual_samples
                audio_data = np.pad(audio_data, (0, pad_samples), mode='constant', constant_values=0)
                logger.info(f"Padded audio with {pad_samples} silent samples ({pad_samples/sr*1000:.2f}ms) to reach {expected_samples} total samples")
            elif actual_samples > expected_samples:
                # Trim excess samples from the end
                trim_samples = actual_samples - expected_samples
                audio_data = audio_data[:expected_samples]
                logger.info(f"Trimmed {trim_samples} samples ({trim_samples/sr*1000:.2f}ms) to reach {expected_samples} total samples")
            else:
                logger.debug(f"No padding needed: audio already has exact sample count ({actual_samples} samples)")
            
            # Create temporary WAV file with exact sample count
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_file_path = temp_file.name
            temp_file.close()
            
            # Write audio with original sample rate and exact sample count
            sf.write(temp_file_path, audio_data, sr, subtype='PCM_16')
            
            # Verify the written file
            verify_info = sf.info(temp_file_path)
            logger.debug(
                f"Created padded audio file: {temp_file_path} "
                f"({verify_info.samplerate}Hz, {verify_info.duration:.3f}s, {verify_info.frames} samples)"
            )
            
            # Run diarization on the padded audio
            if diarization_params:
                diarization = self.diarization_pipeline(temp_file_path, **diarization_params)
            else:
                diarization = self.diarization_pipeline(temp_file_path)
            
            # Convert to Annotation (pyannote 4.0 compatibility)
            if hasattr(diarization, "itertracks"):
                result = diarization
            else:
                result = diarization.speaker_diarization
            
            return result
            
        finally:
            # Clean up temporary file
            if temp_file_path is not None:
                try:
                    Path(temp_file_path).unlink()
                    logger.debug(f"Cleaned up temporary file: {temp_file_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary file {temp_file_path}: {e}")
    
    def _detect_sample_mismatch(
        self,
        audio_path: str,
        claimed_sr: int,
        duration: float,
        frames: int
    ) -> bool:
        """
        Detect sample rate mismatches by comparing claimed vs actual sample counts.
        
        Args:
            audio_path: Path to the audio file
            claimed_sr: Sample rate reported by file metadata
            duration: Duration in seconds reported by file metadata
            frames: Total number of frames/samples in the file
        """
        # Calculate expected samples based on duration and claimed sample rate
        expected_samples = int(duration * claimed_sr)
        actual_samples = frames
        
        # Calculate the difference
        sample_diff = actual_samples - expected_samples
        if sample_diff == 0:
            logger.debug(f"✓ No sample mismatch detected: {actual_samples} samples match expected")
            return False
        
        # Calculate percentage difference
        diff_pct = (abs(sample_diff) / expected_samples) * 100 if expected_samples > 0 else 0
        
        # Log the mismatch with appropriate severity
        if abs(sample_diff) <= 10:
            # Tiny mismatch - likely rounding, very common and safe to ignore
            logger.debug(
                f"Negligible sample mismatch: expected {expected_samples}, "
                f"got {actual_samples} (diff: {sample_diff:+d}, {diff_pct:.4f}%)"
            )
        elif diff_pct < 0.1:
            # Small mismatch - less than 0.1% difference
            logger.debug(
                f"Minor sample mismatch: expected {expected_samples}, "
                f"got {actual_samples} (diff: {sample_diff:+d}, {diff_pct:.3f}%)"
            )
        elif diff_pct < 0.5:
            # Moderate mismatch - could cause issues with strict chunking
            logger.info(
                f"⚠ Sample mismatch detected: expected {expected_samples}, "
                f"got {actual_samples} (diff: {sample_diff:+d}, {diff_pct:.2f}%)"
            )
        else:
            # Significant mismatch - likely to cause chunking errors
            logger.warning(
                f"⚠⚠ Significant sample mismatch: expected {expected_samples}, "
                f"got {actual_samples} (diff: {sample_diff:+d}, {diff_pct:.2f}%)"
            )
            # raise error
            raise ValueError("Significant sample mismatch detected")
        return True
    
    def _transcribe_audio(self, audio_path: str) -> List[Dict[str, Any]]:
        """Run Whisper transcription over the full audio file."""

        if self.whisper_model is None:
            raise RuntimeError("Whisper model not loaded")

        try:
            whisper_segments, _ = self.whisper_model.transcribe(
                audio_path,
                language=None,
                word_timestamps=True,
            )
        except Exception as exc:  # pragma: no cover - safety belt
            raise RuntimeError(f"Failed to transcribe audio: {exc}") from exc

        processed_segments: List[Dict[str, Any]] = []
        for raw_segment in whisper_segments:
            if raw_segment.text is None:
                continue

            text = raw_segment.text.strip()
            if not text:
                continue

            words_payload: List[Dict[str, Any]] = []
            if hasattr(raw_segment, "words") and raw_segment.words:
                for word in raw_segment.words:
                    if word is None:
                        continue
                    word_prob = getattr(word, "probability", None)
                    words_payload.append(
                        {
                            "start": getattr(word, "start", None),
                            "end": getattr(word, "end", None),
                            "word": getattr(word, "word", ""),
                            "probability": word_prob,
                        }
                    )

            processed_segments.append(
                {
                    "segment": Segment(raw_segment.start, raw_segment.end),
                    "text": text,
                    "words": words_payload,
                    "avg_logprob": getattr(raw_segment, "avg_logprob", None),
                }
            )

        return processed_segments

    def _merge_diarization_and_transcription(
        self,
        diarization: Annotation,
        whisper_segments: List[Dict[str, Any]],
    ) -> List[SpeakerSegment]:
        """Merge diarization labels with Whisper transcription segments."""

        if not whisper_segments:
            return []

        speaker_text: List[Dict[str, Any]] = []
        for entry in whisper_segments:
            segment: Segment = entry["segment"]
            speaker_label = self._infer_speaker_for_segment(diarization, segment)
            speaker_text.append({"segment": segment, "speaker": speaker_label, **entry})

        merged_entries = self._merge_sentences(speaker_text)

        speaker_mapping: Dict[str, str] = {}
        formatted_segments: List[SpeakerSegment] = []

        for merged in merged_entries:
            raw_label = merged["speaker"]
            display_label = self._format_speaker_label(raw_label, speaker_mapping)
            confidence = self._compute_confidence(merged["items"])

            formatted_segments.append(
                SpeakerSegment(
                    start=merged["start"],
                    end=merged["end"],
                    speaker=display_label,
                    text=merged["text"],
                    confidence=confidence,
                )
            )

        formatted_segments.sort(key=lambda seg: seg.start)
        return formatted_segments

    # ---------------------------------------------------------------------
    # Word-level diarization + regrouping
    # ---------------------------------------------------------------------
    def _word_level_diarization_merge(
        self,
        diarization: Annotation,
        whisper_segments: List[Dict[str, Any]],
        max_gap: float = 10.0,
    ) -> List[SpeakerSegment]:
        """
        Perform word-level speaker attribution, then regroup contiguous
        same-speaker words into new segments.

        Steps:
          1. Iterate over Whisper segments & their words.
          2. For each word (with timestamps), infer speaker via diarization.
          3. Accumulate a flat list of word tokens: (start, end, word, speaker, prob).
          4. Regroup consecutive tokens sharing the same speaker where the
             inter-word gap <= max_gap seconds.
          5. Compute per-group confidence as the average of available word probs.

        Fallbacks:
          * If a Whisper segment has no word timestamps, we assign a single
            speaker to the whole segment (segment-level inference) and use
            its avg_logprob (if present) as a proxy confidence.
          * If no tokens produced overall, returns empty list.
        """
        if not whisper_segments:
            return []

        word_tokens: List[Dict[str, Any]] = []

        # Collect word-level tokens
        for seg in whisper_segments:
            segment: Segment = seg["segment"]
            words = seg.get("words") or []
            if words:
                for w in words:
                    w_start = w.get("start")
                    w_end = w.get("end")
                    if w_start is None or w_end is None:
                        continue
                    # Construct a tiny segment for speaker inference
                    w_segment = Segment(float(w_start), float(w_end))
                    spk = self._infer_speaker_for_segment(diarization, w_segment)
                    token_text = (w.get("word") or "").strip()
                    if not token_text:
                        continue
                    word_tokens.append(
                        {
                            "start": float(w_start),
                            "end": float(w_end),
                            "speaker": spk,
                            "text": token_text,
                            "prob": w.get("probability"),
                        }
                    )
            else:
                # Segment-level fallback
                spk = self._infer_speaker_for_segment(diarization, segment)
                confidence = None
                avg_lp = seg.get("avg_logprob")
                if avg_lp is not None:
                    try:
                        confidence = math.exp(float(avg_lp))
                    except Exception:  # pragma: no cover - safety belt
                        confidence = None
                word_tokens.append(
                    {
                        "start": float(segment.start),
                        "end": float(segment.end),
                        "speaker": spk,
                        "text": seg.get("text", "").strip(),
                        "prob": confidence,
                        "_segment_level": True,
                    }
                )

        if not word_tokens:
            return []

        # Sort by time to ensure proper ordering
        word_tokens.sort(key=lambda t: (t["start"], t["end"]))

        # Regroup contiguous same-speaker tokens
        grouped: List[SpeakerSegment] = []
        buffer: List[Dict[str, Any]] = []

        def flush_buffer():
            if not buffer:
                return
            start = buffer[0]["start"]
            end = buffer[-1]["end"]
            speaker_raw = buffer[0]["speaker"]
            
            # Smart text joining: handle segment-level vs word-level tokens differently
            text_fragments = []
            for tok in buffer:
                if not tok["text"]:
                    continue
                text_fragments.append(tok["text"])
            
            # Join text intelligently
            if not text_fragments:
                text = None
            elif len(text_fragments) == 1:
                # Single fragment - use as-is
                text = text_fragments[0].strip()
            else:
                # Multiple fragments - join with smart spacing
                result = []
                for i, fragment in enumerate(text_fragments):
                    fragment = fragment.strip()
                    if not fragment:
                        continue
                    
                    # Add space before fragment unless:
                    # 1. It's the first fragment, OR
                    # 2. Previous fragment ends with whitespace, OR
                    # 3. Current fragment starts with punctuation that doesn't need leading space
                    if result:
                        prev_ends_with_space = result[-1][-1:].isspace()
                        cur_starts_with_no_space_punct = fragment[0] in ".,;:!?)]}\"'"
                        
                        if not prev_ends_with_space and not cur_starts_with_no_space_punct:
                            result.append(" ")
                    
                    result.append(fragment)
                
                text = "".join(result)
            
            probs = [t["prob"] for t in buffer if t.get("prob") is not None]
            confidence = sum(probs) / len(probs) if probs else None
            grouped.append(
                SpeakerSegment(
                    start=start,
                    end=end,
                    speaker=speaker_raw,
                    text=text,
                    confidence=confidence,
                )
            )
            buffer.clear()

        prev_end: Optional[float] = None
        prev_speaker: Optional[str] = None

        for tok in word_tokens:
            cur_start = tok["start"]
            cur_speaker = tok["speaker"]
            gap = (cur_start - prev_end) if (prev_end is not None) else 0.0
            new_group = False
            if not buffer:
                new_group = True
            else:
                # Speaker change triggers flush
                if cur_speaker != prev_speaker:
                    new_group = True
                # Excessive gap triggers flush even if same speaker
                elif gap > max_gap:
                    new_group = True

            if new_group:
                flush_buffer()
            buffer.append(tok)
            prev_end = tok["end"]
            prev_speaker = cur_speaker

        flush_buffer()

        # Map raw diarization speakers to user-friendly labels
        mapping: Dict[str, str] = {}
        for seg in grouped:
            seg.speaker = self._format_speaker_label(seg.speaker, mapping)

        grouped.sort(key=lambda s: s.start)
        return grouped

    def _infer_speaker_for_segment(
        self, diarization: Annotation, segment: Segment
    ) -> str:
        """Infer dominant speaker label for a given time segment."""

        cropped = diarization.crop(segment)

        durations: Dict[str, float] = defaultdict(float)
        for track in cropped.itertracks(yield_label=True):
            seg: Segment = track[0]
            label = track[-1]
            overlap = max(0.0, min(segment.end, seg.end) - max(segment.start, seg.start))
            if overlap > 0:
                durations[str(label)] += overlap

        if not durations:
            return "Unknown"

        return max(durations.items(), key=lambda item: item[1])[0]

    def _merge_sentences(self, speaker_text: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge consecutive segments when speaker remains the same or sentence ends."""

        merged: List[Dict[str, Any]] = []
        buffer: List[Dict[str, Any]] = []
        previous_speaker: Optional[str] = None

        for entry in speaker_text:
            speaker = entry["speaker"]
            text = entry["text"]

            def flush_buffer() -> None:
                if not buffer:
                    return
                merged.append(self._collapse_buffer(buffer))
                buffer.clear()

            if previous_speaker is not None and speaker != previous_speaker and buffer:
                flush_buffer()

            buffer.append(entry)
            previous_speaker = speaker

            if text and text[-1] in {".", "?", "!", "。", "？", "！", "…"}:
                flush_buffer()
                previous_speaker = None

        if buffer:
            merged.append(self._collapse_buffer(buffer))

        return merged

    @staticmethod
    def _collapse_buffer(buffer: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collapse buffered segments into a single merged entry."""

        start = buffer[0]["segment"].start
        end = buffer[-1]["segment"].end
        speaker = buffer[0]["speaker"]
        text = " ".join(item["text"] for item in buffer).strip()

        return {
            "start": start,
            "end": end,
            "speaker": speaker,
            "text": text,
            "items": list(buffer),
        }

    @staticmethod
    def _compute_confidence(items: List[Dict[str, Any]]) -> Optional[float]:
        """Compute aggregated confidence from word probabilities or avg logprob."""

        word_probs: List[float] = []
        for entry in items:
            for word in entry.get("words", []):
                prob = word.get("probability")
                if prob is not None:
                    word_probs.append(prob)

        if word_probs:
            return sum(word_probs) / len(word_probs)

        logprobs: List[float] = []
        for entry in items:
            avg_logprob = entry.get("avg_logprob")
            if avg_logprob is not None:
                logprobs.append(float(avg_logprob))
        if logprobs:
            return sum(math.exp(lp) for lp in logprobs) / len(logprobs)

        return None

    def _format_speaker_label(
        self, raw_label: str, mapping: Dict[str, str]
    ) -> str:
        """Convert raw diarization labels into user-friendly speaker names."""

        if raw_label == "Unknown":
            return "Unknown"

        if raw_label not in mapping:
            mapping[raw_label] = f"{self.config.speaker_labels} {len(mapping) + 1}"

        return mapping[raw_label]
    
    def _save_result(
        self,
        result: TranscriptionResult,
        output_path: str,
        output_format: Optional[str]
    ) -> None:
        """Save transcription result to file."""
        format_name = output_format or self.config.output_format
        output_file = Path(output_path)
        
        # Ensure correct file extension
        if format_name == "srt" and output_file.suffix != ".srt":
            output_file = output_file.with_suffix(".srt")
        elif format_name == "vtt" and output_file.suffix != ".vtt":
            output_file = output_file.with_suffix(".vtt")
        elif format_name == "json" and output_file.suffix != ".json":
            output_file = output_file.with_suffix(".json")
        
        # Generate content
        if format_name == "srt":
            content = result.to_srt()
        elif format_name == "vtt":
            content = result.to_vtt()
        elif format_name == "json":
            content = result.to_json()
        else:
            raise ValueError(f"Unsupported output format: {format_name}")
        
        # Save to file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(content, encoding="utf-8")
        
        logger.info(f"Results saved to: {output_file}")
    
    def _save_diarization(
        self,
        diarization: Annotation,
        audio_path: str,
        output_path: str,
        fmt: str = "rttm"
    ) -> None:
        """Save raw diarization annotation.

        Supported formats:
          rttm  - Standard Rich Transcription Time Marked format
          json  - Simple JSON list of segments {start, end, speaker}
          txt   - Human-readable plain text
        """
        fmt = fmt.lower()
        path = Path(output_path)
        if fmt == "rttm" and path.suffix.lower() != ".rttm":
            path = path.with_suffix(".rttm")
        elif fmt == "json" and path.suffix.lower() != ".json":
            path = path.with_suffix(".json")
        elif fmt == "txt" and path.suffix.lower() != ".txt":
            path = path.with_suffix(".txt")

        path.parent.mkdir(parents=True, exist_ok=True)

        # Build segment list once
        segments: List[Dict[str, Any]] = []
        for track in diarization.itertracks(yield_label=True):
            # itertracks may yield (segment, track_id) or (segment, track_id, label)
            if len(track) == 3:
                segment, _track_id, label = track  # type: ignore[misc]
            else:
                segment, _track_id = track  # type: ignore[misc]
                label = "Unknown"
            segments.append(
                {
                    "start": float(segment.start),
                    "end": float(segment.end),
                    "duration": float(segment.duration),
                    "speaker": str(label),
                }
            )

        if fmt == "rttm":
            # Compose RTTM lines (SPEAKER <file_id> 1 <start> <duration> <NA> <NA> <speaker> <NA> <NA>)
            file_id = Path(audio_path).stem
            lines = []
            for seg in segments:
                lines.append(
                    f"SPEAKER {file_id} 1 {seg['start']:.3f} {seg['duration']:.3f} <NA> <NA> {seg['speaker']} <NA> <NA>"
                )
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        elif fmt == "json":
            import json
            path.write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")
        elif fmt == "txt":
            lines = []
            for seg in segments:
                lines.append(
                    f"[{seg['start']:.2f}-{seg['end']:.2f}] {seg['speaker']}"
                )
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        else:
            raise ValueError(f"Unsupported diarization output format: {fmt}")

        logger.info(f"Diarization saved to: {path}")
    
    def get_model_info(self) -> dict:
        """Get information about loaded models."""
        return {
            "whisper_model": self.config.whisper_model,
            "pyannote_model": self.config.pyannote_model,
            "device": self.device,
            "sample_rate": self.config.sample_rate
        }