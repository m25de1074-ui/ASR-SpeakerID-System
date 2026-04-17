# ASR-SpeakerID-System
Speech understanding project
# Automatic Speech Recognition & Speaker Identification using Deep Learning

##  Overview
This project implements an end-to-end **Speech Understanding System** that performs:

-  Automatic Speech Recognition (ASR)
-  Speaker Identification (Speaker Diarization)

Given an input audio file, the system:
1. Transcribes speech into text
2. Identifies different speakers
3. Generates timestamped subtitles in `.srt` format

---

##  Features
-  Speech-to-text using Whisper (Faster-Whisper)
-  Speaker diarization using Pyannote Audio
-  Timestamped subtitle generation (SRT)
-  GPU support (CUDA)
-  Demo mode for quick testing



## Project Structure
```
ASR-SpeakerID-System/
│
├── asr_pipeline/
│ ├── __init__.py
│ ├── cli.py
│ ├── pipeline.py
│ ├── models.py
│ ├── config.py
│ └── audio_utils.py
│
├── tests/
│ ├── __init__.py
│ ├── test_pipeline.py
├── .gitignore
├── .python-version
├── run_pipeline.sh
├── pyproject.toml
├── uv.lock
├── .env.example
├── sample.wav
├── output.srt
├── output1.srt
├── output2.srt
└── README.md
```


##  Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/m25de1074-ui/ASR-SpeakerID-System
cd ASR-SpeakerID-System
```
### 2. Install dependencies
```bash
uv sync
```

###3. Configure environment variables

Create a .env file:

```bash
HUGGINGFACE_TOKEN=your_token_here
WHISPER_MODEL=base
PYANNOTE_MODEL=pyannote/speaker-diarization-3.1
OUTPUT_FORMAT=srt
SPEAKER_LABELS=Speaker
```
###4.Accept model access:
-  https://huggingface.co/pyannote/speaker-diarization-3.1
-  https://huggingface.co/pyannote/segmentation-3.0

## Quickstart Demo
```bash
uv run python quickstart.py -demo
```

## Run on Custom Audio
```bash
uv run python -m asr_pipeline.cli process sample.wav --output output.srt
```

## Output Example

1
00:00:00,000 --> 00:00:03,520
Speaker 1: To all these inquiries, the count responded in the affirmative.

2
00:00:04,580 --> 00:00:04,580
Unknown: I

3
00:00:04,580 --> 00:00:09,480
Speaker 2: neglected to tell you that you're not the borrower dream about him last night. That?



## Methodology
### Speech Recognition
    Model: Whisper (Faster-Whisper)
    Converts audio → text
### Speaker Identification
    Model: Pyannote Audio
    Segments speakers in audio
### Pipeline Flow
    Audio preprocessing
    Speaker diarization
    Speech transcription
    Alignment of speakers with text
    Subtitle generation

## Limitations
### Overlapping speech may lead to:
-  Incorrect speaker labels
-  “Unknown” segments
-  Performance depends on audio clarity
-  Short audio clips reduce accuracy

## Future Work
-  Real-time speech processing
-  Improved speaker embeddings
-  Noise reduction techniques
-  Multi-language support

## Technologies Used
-  Python
-  PyTorch
-  Whisper (Faster-Whisper)
-  Pyannote Audio
-  Hugging Face

### Authors
-  Arun Kumar P -- M25DE1020
-  Chandu Srinivas -- M25DE1074
-  Akash Bhardwaj -- M25DE1015
