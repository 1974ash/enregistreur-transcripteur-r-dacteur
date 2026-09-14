Summary of app.py
This is a Streamlit web application for audio/video transcription and summarization using local AI models (Whisper for transcription and Ollama for text analysis).

Key Features:
Audio Input

Record audio directly in the browser via microphone
Upload audio/video files (WAV, MP3, M4A, OGG, FLAC, WebM, MP4, MOV, AVI, MKV)
Media Processing

Converts audio/video to WAV mono at 16 kHz using FFmpeg
Handles multiple media formats
Transcription (using Whisper)

Configurable model sizes (tiny to large-v3)
Supports multiple languages or auto-detection
Generates timestamped transcripts
Outputs segments with start/end times
Summarization & Analysis (using Ollama)

Three output types:
Short summary
Detailed report
Structured analysis
Handles long transcripts by chunking them and merging results
Export Options

Download transcript with timestamps (Markdown format)
Download transcription in SRT format (subtitles)
Display results in organized tabs
Configuration

Sidebar settings for Whisper model, compute device (CPU/CUDA), language selection
Ollama model and timeout configuration
Environment variable support for defaults
Tech Stack:
Streamlit - UI framework
Faster-Whisper - Local speech-to-text
Ollama - Local LLM for summarization
FFmpeg - Media format conversion

To launch this app :
In one terminal type : ollama serve 
In a second one type :  streamlit run app.py 
You'll see the app in this local website : http://localhost:8501
