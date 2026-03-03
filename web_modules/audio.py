from __future__ import annotations

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
        self._buffer: list[float] = []
        self._max_buffer = sample_rate * 8
        self._lock = threading.Lock()
        self._noise_floor = max(1e-5, vad_threshold * 0.5)

    def _callback(self, indata, frames, callback_time, status) -> None:  # noqa: ANN001
        del frames, callback_time, status
        mono = indata[:, 0].astype(np.float32)
        with self._lock:
            self._latest = mono
            self._buffer.extend(mono.tolist())
            if len(self._buffer) > self._max_buffer:
                self._buffer = self._buffer[-self._max_buffer :]

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
            arr = np.array(self._buffer, dtype=np.float32)
        return arr[-n:] if arr.size >= n else arr

    def rms(self) -> float:
        with self._lock:
            chunk = self._latest.copy()
        return float(np.sqrt(np.mean(chunk * chunk))) if chunk.size else 0.0

    def vad(self) -> bool:
        rms = self.rms()
        # Adaptive thresholding: tolerate stationary outdoor noise.
        if rms < (self._noise_floor * 1.6):
            self._noise_floor = (0.96 * self._noise_floor) + (0.04 * max(1e-6, rms))
        dynamic_thr = max(self.vad_threshold, self._noise_floor * 2.6)
        return rms >= dynamic_thr
