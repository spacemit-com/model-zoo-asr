"""
ASR Engine - Main recognition interface
"""

from enum import Enum
from typing import Optional, Union
from pathlib import Path
import numpy as np

# Import C++ bindings
try:
    from . import _spacemit_asr as _asr
except ImportError:
    raise ImportError(
        "_spacemit_asr module not found. Please build the C++ bindings first:\n"
        "  cd build && cmake .. && make _spacemit_asr"
    )


class Language(Enum):
    """Supported languages for ASR"""
    AUTO = _asr.Language.AUTO
    ZH = _asr.Language.ZH      # Chinese
    EN = _asr.Language.EN      # English
    JA = _asr.Language.JA      # Japanese
    KO = _asr.Language.KO      # Korean
    YUE = _asr.Language.YUE    # Cantonese

    def to_native(self):
        return self.value


class Config:
    """ASR Configuration"""

    def __init__(self, model_dir: str = "~/.cache/models/asr/sensevoice"):
        """
        Create ASR configuration.

        Args:
            model_dir: Path to SenseVoice model directory
        """
        self._config = _asr.ASRConfig.sensevoice(str(Path(model_dir).expanduser()))
        self._model_dir = model_dir

    @property
    def language(self) -> Language:
        """Get/set recognition language"""
        return Language(self._config.language)

    @language.setter
    def language(self, value: Language):
        self._config.language = value.to_native()

    @property
    def sample_rate(self) -> int:
        """Get/set sample rate (Hz)"""
        return self._config.sample_rate

    @sample_rate.setter
    def sample_rate(self, value: int):
        self._config.sample_rate = value

    @property
    def punctuation_enabled(self) -> bool:
        """Get/set automatic punctuation"""
        return self._config.punctuation_enabled

    @punctuation_enabled.setter
    def punctuation_enabled(self, value: bool):
        self._config.punctuation_enabled = value

    def with_language(self, language: Language) -> "Config":
        """Set language (chainable)"""
        self._config.language = language.to_native()
        return self

    def with_punctuation(self, enabled: bool = True) -> "Config":
        """Set punctuation (chainable)"""
        self._config.punctuation_enabled = enabled
        return self

    @property
    def provider(self) -> str:
        """Get/set execution provider: 'cpu' or 'spacemit'"""
        return self._config.extra_params.get("provider", "spacemit")

    @provider.setter
    def provider(self, value: str):
        self._config.extra_params["provider"] = value


class Result:
    """Recognition result wrapper"""

    def __init__(self, native_result):
        self._result = native_result

    @property
    def text(self) -> str:
        """Full recognized text"""
        return self._result.get_text()

    @property
    def sentences(self) -> list:
        """List of sentence results"""
        return self._result.sentences

    @property
    def audio_duration_ms(self) -> int:
        """Audio duration in milliseconds"""
        return self._result.audio_duration_ms

    @property
    def processing_time_ms(self) -> int:
        """Processing time in milliseconds"""
        return self._result.processing_time_ms

    @property
    def rtf(self) -> float:
        """Real-Time Factor (processing_time / audio_duration)"""
        return self._result.rtf

    @property
    def is_empty(self) -> bool:
        """Check if result is empty"""
        return self._result.is_empty()

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return f"Result('{self.text}', rtf={self.rtf:.3f})"

    def __bool__(self) -> bool:
        return not self.is_empty


class Engine:
    """
    ASR Engine - Main speech recognition interface

    Example:
        engine = Engine()
        engine.initialize()

        # Recognize file
        result = engine.recognize_file("audio.wav")
        print(result.text)

        # Recognize numpy array
        audio = np.array([...], dtype=np.float32)
        result = engine.recognize(audio)
        print(result.text)

        engine.shutdown()

    Context manager:
        with Engine() as engine:
            result = engine.recognize_file("audio.wav")
            print(result.text)
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Create ASR engine.

        Args:
            config: ASR configuration (optional, uses default if not provided)
        """
        self._engine = _asr.ASREngine()
        self._config = config
        self._initialized = False

    def initialize(self, config: Optional[Config] = None) -> "Engine":
        """
        Initialize the engine.

        Args:
            config: ASR configuration (uses constructor config if not provided)

        Returns:
            self for method chaining
        """
        if config:
            self._config = config
        if self._config is None:
            self._config = Config()

        error = self._engine.initialize(self._config._config)
        if not error.is_ok():
            raise RuntimeError(f"Failed to initialize ASR engine: {error.message}")

        self._initialized = True
        return self

    def shutdown(self):
        """Shutdown the engine and release resources"""
        if self._initialized:
            self._engine.shutdown()
            self._initialized = False

    @property
    def is_initialized(self) -> bool:
        """Check if engine is initialized"""
        return self._initialized

    @property
    def backend_name(self) -> str:
        """Get backend name"""
        return self._engine.backend_name

    def recognize(self, audio: np.ndarray) -> Result:
        """
        Recognize audio data.

        Args:
            audio: Audio samples as numpy array (float32 or int16)

        Returns:
            Recognition result
        """
        if not self._initialized:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        # Ensure correct dtype
        if audio.dtype == np.float32:
            pass
        elif audio.dtype == np.int16:
            pass
        elif audio.dtype == np.float64:
            audio = audio.astype(np.float32)
        else:
            audio = audio.astype(np.float32)

        # Ensure 1D array
        if audio.ndim > 1:
            audio = audio.flatten()

        result = self._engine.recognize(audio)
        return Result(result)

    def recognize_file(self, file_path: Union[str, Path]) -> Result:
        """
        Recognize audio from file.

        Args:
            file_path: Path to audio file (WAV, MP3, etc.)

        Returns:
            Recognition result
        """
        if not self._initialized:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        path = str(Path(file_path).expanduser().resolve())
        result = self._engine.recognize_file(path)
        return Result(result)

    def set_language(self, language: Language):
        """
        Set recognition language.

        Args:
            language: Target language
        """
        self._engine.set_language(language.to_native())

    def set_punctuation(self, enabled: bool):
        """
        Set automatic punctuation.

        Args:
            enabled: True to enable, False to disable
        """
        # Note: May require re-initialization depending on backend
        if self._config:
            self._config.punctuation_enabled = enabled

    # -------------------------------------------------------------------------
    # Streaming API
    # -------------------------------------------------------------------------

    def start(self, callback=None, **kwargs):
        """
        Start streaming recognition session.

        Args:
            callback: AsrCallback instance for receiving events
            **kwargs: Additional options (reserved)

        Example:
            class MyCallback(asr.AsrCallback):
                def on_event(self, result):
                    print(result.text)

            engine.start(callback=MyCallback())
            for chunk in audio_stream:
                engine.send_audio_frame(chunk)
            engine.stop()
        """
        if not self._initialized:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        if callback is not None:
            # Import here to avoid circular import
            from .callback import AsrCallback
            if isinstance(callback, AsrCallback):
                native_cb = callback._get_native_callback()
                self._engine.set_callback(native_cb)
            else:
                raise TypeError("callback must be an AsrCallback instance")

        self._engine.start()

    def send_audio_frame(self, buffer: bytes):
        """
        Send audio frame during streaming.

        Args:
            buffer: PCM S16LE audio data (16kHz, mono)

        Note:
            Recommended chunk size: 100ms of audio (~3200 bytes at 16kHz)
        """
        if not self._initialized:
            raise RuntimeError("Engine not initialized")

        self._engine.send_audio_frame(buffer)

    def stop(self):
        """
        Stop streaming recognition session.

        Blocks until all audio is processed and final results are returned.
        """
        if not self._initialized:
            return

        self._engine.stop()

    def flush(self):
        """
        Flush buffer and recognize immediately (keeps session active).

        Use this when your VAD detects end of sentence. The recognition result
        will be delivered via callback, and you can continue sending audio
        for the next sentence.

        Example:
            engine.start(callback=my_callback)
            while recording:
                engine.send_audio_frame(chunk)
                if vad.is_sentence_end():
                    engine.flush()  # Recognize current buffer
            engine.stop()
        """
        if not self._initialized:
            raise RuntimeError("Engine not initialized")

        self._engine.flush()

    @property
    def is_streaming(self) -> bool:
        """Check if streaming session is active."""
        return self._engine.is_streaming() if self._initialized else False

    def __enter__(self) -> "Engine":
        if not self._initialized:
            self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        return False

    @staticmethod
    def get_version() -> str:
        """Get library version"""
        return _asr.ASREngine.get_version()

    @staticmethod
    def get_available_backends() -> list:
        """Get list of available backends"""
        return _asr.ASREngine.get_available_backends()
