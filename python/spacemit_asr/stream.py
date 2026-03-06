"""
ASR Audio Streaming - Real-time audio capture and recognition
"""

from typing import Optional, Callable, Generator, List, Tuple
from threading import Thread, Event
from queue import Queue, Empty
import numpy as np
import time

try:
    from . import _spacemit_asr as _asr
except ImportError:
    raise ImportError("_spacemit_asr module not found. Build the C++ bindings first.")

from .engine import Engine, Config, Result, Language


class AudioStream:
    """
    Low-level audio input stream.

    Captures audio from microphone and provides it via callback or queue.

    Example:
        stream = AudioStream(sample_rate=16000, channels=1)

        def callback(audio, frames, channels):
            print(f"Got {frames} frames")

        stream.set_callback(callback)
        stream.open()
        stream.start()
        time.sleep(5)
        stream.stop()
        stream.close()
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        channels: int = 2,
        frames_per_buffer: int = 1024,
        device_index: int = -1
    ):
        """
        Create audio stream.

        Args:
            sample_rate: Sample rate in Hz (default: 48000)
            channels: Number of channels (default: 2)
            frames_per_buffer: Frames per callback (default: 1024)
            device_index: Audio device index (-1 for default)
        """
        self._stream = _asr.AudioInputStream()
        self._sample_rate = sample_rate
        self._channels = channels
        self._frames_per_buffer = frames_per_buffer
        self._device_index = device_index
        self._callback = None
        self._is_open = False

    def __del__(self):
        """Destructor - ensure stream is closed"""
        try:
            if self._is_open:
                self._stream.stop()
                self._stream.close()
        except Exception:
            pass

    def set_callback(self, callback: Callable[[np.ndarray, int, int], None]):
        """
        Set audio callback.

        Args:
            callback: Function called with (audio_array, frames, channels)
        """
        self._callback = callback
        self._stream.set_callback(callback)

    def open(self) -> bool:
        """Open the audio stream"""
        result = self._stream.open(
            self._sample_rate,
            self._channels,
            self._frames_per_buffer,
            self._device_index
        )
        self._is_open = result
        return result

    def close(self):
        """Close the audio stream"""
        if self._is_open:
            self._stream.close()
            self._is_open = False

    def start(self) -> bool:
        """Start audio capture"""
        return self._stream.start()

    def stop(self) -> bool:
        """Stop audio capture"""
        return self._stream.stop()

    @property
    def is_running(self) -> bool:
        """Check if stream is running"""
        return self._stream.is_running()

    @property
    def is_open(self) -> bool:
        """Check if stream is open"""
        return self._stream.is_open()

    @property
    def sample_rate(self) -> int:
        """Actual sample rate"""
        return self._stream.sample_rate

    @property
    def channels(self) -> int:
        """Actual number of channels"""
        return self._stream.channels

    def __enter__(self):
        self.open()
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        self.close()
        return False

    @staticmethod
    def list_devices() -> List[Tuple[int, str]]:
        """
        List available audio input devices.

        Returns:
            List of (device_index, device_name) tuples
        """
        return _asr.AudioInputStream.list_devices()


class Resampler:
    """Audio resampler for converting sample rates"""

    def __init__(self, input_rate: int, output_rate: int, channels: int = 1):
        """
        Create resampler.

        Args:
            input_rate: Input sample rate
            output_rate: Output sample rate
            channels: Number of channels
        """
        self._resampler = _asr.Resampler(input_rate, output_rate, channels)

    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        Resample audio data.

        Args:
            audio: Input audio (float32)

        Returns:
            Resampled audio
        """
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        return self._resampler.process(audio)


class MicrophoneStream:
    """
    High-level microphone stream with real-time ASR.

    Captures audio from microphone, resamples to 16kHz, and performs
    speech recognition.

    Example:
        # Basic usage
        with MicrophoneStream() as stream:
            print("Listening... (Ctrl+C to stop)")
            for result in stream.recognize():
                print(f"Recognized: {result.text}")

        # With configuration
        stream = MicrophoneStream(
            language=Language.ZH,
            device_index=0,
            chunk_duration=3.0  # Recognize every 3 seconds
        )
        stream.start()
        while True:
            result = stream.get_result(timeout=5.0)
            if result:
                print(result.text)
    """

    def __init__(
        self,
        language: Language = Language.ZH,
        model_dir: str = "~/.cache/models/asr/sensevoice",
        device_index: int = -1,
        input_sample_rate: int = 48000,
        input_channels: int = 2,
        chunk_duration: float = 5.0,  # seconds
        frames_per_buffer: int = 2048,
    ):
        """
        Create microphone stream.

        Args:
            language: Recognition language
            model_dir: Path to SenseVoice model
            device_index: Audio device index (-1 for default)
            input_sample_rate: Input sample rate (default: 48000)
            input_channels: Input channels (default: 2)
            chunk_duration: Duration to accumulate before recognition (seconds)
        """
        self._language = language
        self._model_dir = model_dir
        self._device_index = device_index
        self._input_sample_rate = input_sample_rate
        self._input_channels = input_channels
        self._chunk_duration = chunk_duration
        self._frames_per_buffer = frames_per_buffer

        # Components
        self._audio_stream: Optional[AudioStream] = None
        self._resampler: Optional[Resampler] = None
        self._engine: Optional[Engine] = None

        # State
        self._audio_buffer: List[np.ndarray] = []
        self._result_queue: Queue = Queue()
        self._stop_event = Event()
        self._worker_thread: Optional[Thread] = None
        self._running = False

    def __del__(self):
        """Destructor - ensure resources are released"""
        try:
            self.close()
        except Exception:
            pass

    def start(self):
        """Start the microphone stream and recognition"""
        if self._running:
            return

        # Initialize ASR engine
        config = Config(self._model_dir)
        config.language = self._language

        self._engine = Engine(config)
        self._engine.initialize()

        # Initialize audio stream first to get actual sample rate
        self._audio_stream = AudioStream(
            sample_rate=self._input_sample_rate,
            channels=self._input_channels,
            frames_per_buffer=self._frames_per_buffer,
            device_index=self._device_index
        )

        # Set callback
        self._audio_buffer = []
        self._audio_stream.set_callback(self._audio_callback)

        # Open and start
        if not self._audio_stream.open():
            raise RuntimeError("Failed to open audio stream")

        # Get actual sample rate from device and create resampler accordingly
        actual_sample_rate = self._audio_stream.sample_rate
        if actual_sample_rate != 16000:
            self._resampler = Resampler(actual_sample_rate, 16000, channels=1)
        else:
            self._resampler = None  # No resampling needed

        if not self._audio_stream.start():
            raise RuntimeError("Failed to start audio stream")

        # Start worker thread
        self._stop_event.clear()
        self._worker_thread = Thread(target=self._recognition_worker, daemon=True)
        self._worker_thread.start()

        self._running = True

    def stop(self):
        """Stop the microphone stream"""
        if not self._running:
            return

        self._stop_event.set()

        if self._audio_stream:
            self._audio_stream.stop()
            self._audio_stream.close()
            self._audio_stream = None

        if self._worker_thread:
            self._worker_thread.join(timeout=2.0)
            self._worker_thread = None

        if self._engine:
            self._engine.shutdown()
            self._engine = None

        self._running = False

    def _audio_callback(self, audio: np.ndarray, frames: int, channels: int):
        """Internal callback for audio data"""
        # Convert to mono if stereo
        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1).astype(np.float32)
        else:
            audio = audio.flatten().astype(np.float32)

        # Resample to 16kHz
        if self._resampler:
            audio = self._resampler.process(audio)

        self._audio_buffer.append(audio)

    def _recognition_worker(self):
        """Background worker for recognition"""
        target_samples = int(16000 * self._chunk_duration)

        while not self._stop_event.is_set():
            # Check if we have enough audio
            total_samples = sum(len(buf) for buf in self._audio_buffer)

            if total_samples >= target_samples:
                # Concatenate buffer
                audio = np.concatenate(self._audio_buffer)
                self._audio_buffer = []

                # Recognize
                if self._engine and len(audio) > 0:
                    try:
                        result = self._engine.recognize(audio)
                        if result:
                            self._result_queue.put(result)
                    except Exception as e:
                        print(f"Recognition error: {e}")

            time.sleep(0.1)

        # Final recognition with remaining audio
        if self._audio_buffer and self._engine:
            audio = np.concatenate(self._audio_buffer)
            self._audio_buffer = []
            if len(audio) > 0:
                try:
                    result = self._engine.recognize(audio)
                    if result:
                        self._result_queue.put(result)
                except Exception:
                    pass

    def get_result(self, timeout: Optional[float] = None) -> Optional[Result]:
        """
        Get next recognition result.

        Args:
            timeout: Timeout in seconds (None for blocking)

        Returns:
            Recognition result or None if timeout
        """
        try:
            return self._result_queue.get(timeout=timeout)
        except Empty:
            return None

    def recognize(self) -> Generator[Result, None, None]:
        """
        Generator for recognition results.

        Yields:
            Recognition results as they become available
        """
        while self._running or not self._result_queue.empty():
            result = self.get_result(timeout=0.5)
            if result:
                yield result

    def listen(self, duration: float = 5.0) -> Optional[Result]:
        """
        Listen for a fixed duration and return ONE recognition result.
        Suitable for dialogue/conversation scenarios.

        Args:
            duration: Recording duration in seconds

        Returns:
            Recognition result, or None if no speech detected

        Example:
            stream = MicrophoneStream(device_index=0)

            while True:
                print("Listening...")
                result = stream.listen(duration=5.0)
                if result:
                    print(f"You said: {result.text}")
                    # Process result, call LLM, play TTS, etc.
        """
        # Start if not running
        was_running = self._running
        if not self._running:
            self._start_recording_only()

        # Clear old audio
        self._audio_buffer = []

        # Record for specified duration
        time.sleep(duration)

        # Get accumulated audio
        if not self._audio_buffer:
            return None

        audio = np.concatenate(self._audio_buffer)
        self._audio_buffer = []

        # Stop if we started it
        if not was_running:
            self._stop_recording_only()

        # Recognize
        if self._engine and len(audio) > 0:
            try:
                return self._engine.recognize(audio)
            except Exception as e:
                print(f"Recognition error: {e}")
                return None

        return None

    def _start_recording_only(self):
        """Start recording without the recognition worker thread"""
        if self._running:
            return

        # Initialize ASR engine
        if not self._engine:
            config = Config(self._model_dir)
            config.language = self._language
            self._engine = Engine(config)
            self._engine.initialize()

        # Initialize audio stream first to get actual sample rate
        if not self._audio_stream:
            self._audio_stream = AudioStream(
                sample_rate=self._input_sample_rate,
                channels=self._input_channels,
                frames_per_buffer=self._frames_per_buffer,
                device_index=self._device_index
            )
            self._audio_stream.set_callback(self._audio_callback)

        # Open and start
        if not self._audio_stream.is_open:
            if not self._audio_stream.open():
                raise RuntimeError("Failed to open audio stream")

        # Get actual sample rate from device and create resampler accordingly
        if not self._resampler:
            actual_sample_rate = self._audio_stream.sample_rate
            if actual_sample_rate != 16000:
                self._resampler = Resampler(actual_sample_rate, 16000, channels=1)
            # else: No resampling needed, _resampler stays None

        if not self._audio_stream.is_running:
            if not self._audio_stream.start():
                raise RuntimeError("Failed to start audio stream")

        self._audio_buffer = []
        self._running = True

    def _stop_recording_only(self):
        """Stop recording but keep engine initialized for next listen()"""
        if self._audio_stream:
            self._audio_stream.stop()
            self._audio_stream.close()
            self._audio_stream = None
        self._running = False

    def close(self):
        """Release all resources. Call when done with dialogue."""
        self._stop_recording_only()
        if self._engine:
            self._engine.shutdown()
            self._engine = None
        self._resampler = None

    @property
    def is_running(self) -> bool:
        """Check if stream is running"""
        return self._running

    def __enter__(self) -> "MicrophoneStream":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False


def list_devices() -> List[Tuple[int, str]]:
    """
    List available audio input devices.

    Returns:
        List of (device_index, device_name) tuples
    """
    return AudioStream.list_devices()
