"""
SpacemiT ASR (Automatic Speech Recognition) Python Module

A Python interface for the SenseVoice ASR engine.

Usage:
    import spacemit_asr

    # Quick recognition
    text = spacemit_asr.recognize_file("audio.wav")
    print(text)

    # Full control
    engine = spacemit_asr.Engine()
    engine.initialize()
    result = engine.recognize_file("audio.wav")
    print(result.text)

    # Streaming with microphone
    with spacemit_asr.MicrophoneStream() as stream:
        for text in stream.recognize():
            print(text)
"""

from .engine import Engine, Config, Language, Result
from .stream import AudioStream, MicrophoneStream, Resampler
from .callback import AsrCallback, PrintCallback, CollectCallback
from .utils import recognize_file, recognize_audio, list_devices

__version__ = "1.0.0"
__author__ = "muggle"

__all__ = [
    # Main classes
    "Engine",
    "Config",
    "Language",
    "Result",
    # Streaming
    "AudioStream",
    "MicrophoneStream",
    "Resampler",
    # Callbacks
    "AsrCallback",
    "PrintCallback",
    "CollectCallback",
    # Quick functions
    "recognize_file",
    "recognize_audio",
    "list_devices",
]
