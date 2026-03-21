from __future__ import annotations

from collections import deque
import threading

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None


class AudioMonitor:
    def __init__(self, sample_rate: int, block_size: int, vad_threshold: float) -> None:
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.vad_threshold = vad_threshold
        self._stream = None
        self._latest = np.zeros((block_size,), dtype=np.float32)
        self._buffer: deque[np.ndarray] = deque()
        self._buffer_samples = 0
        self._max_buffer = sample_rate * 8
        self._lock = threading.Lock()
        self._noise_floor = max(1e-5, vad_threshold * 0.5)

    def _callback(self, indata, frames, callback_time, status) -> None:  # noqa: ANN001
        del frames, callback_time, status
        mono = indata[:, 0].astype(np.float32).copy()
        with self._lock:
            self._latest = mono
            self._buffer.append(mono)
            self._buffer_samples += mono.size
            while self._buffer and self._buffer_samples > self._max_buffer:
                dropped = self._buffer.popleft()
                self._buffer_samples -= dropped.size

    def start(self) -> None:
        if sd is None:
            raise ImportError("sounddevice is required for real-time audio monitoring.")
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            channels=1,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def latest_seconds(self, seconds: float) -> np.ndarray:
        n = int(max(1, seconds * self.sample_rate))
        with self._lock:
            if not self._buffer:
                return np.empty((0,), dtype=np.float32)
            chunks: list[np.ndarray] = []
            collected = 0
            for block in reversed(self._buffer):
                chunks.append(block)
                collected += block.size
                if collected >= n:
                    break
        arr = np.concatenate(chunks[::-1]) if chunks else np.empty((0,), dtype=np.float32)
        return arr[-n:] if arr.size >= n else arr

    def rms(self) -> float:
        with self._lock:
            chunk = self._latest
        return float(np.sqrt(np.mean(chunk * chunk))) if chunk.size else 0.0

    def analyze_level(self) -> tuple[float, bool]:
        with self._lock:
            chunk = self._latest
            rms = float(np.sqrt(np.mean(chunk * chunk))) if chunk.size else 0.0
            # Adaptive thresholding: tolerate stationary outdoor noise.
            if rms < (self._noise_floor * 1.6):
                self._noise_floor = (0.96 * self._noise_floor) + (0.04 * max(1e-6, rms))
            dynamic_thr = max(self.vad_threshold, self._noise_floor * 2.6)
            return rms, rms >= dynamic_thr

    def vad(self) -> bool:
        _, is_voice = self.analyze_level()
        return is_voice
