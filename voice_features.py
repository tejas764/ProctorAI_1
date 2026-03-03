from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import librosa
except ImportError:
    librosa = None


TARGET_SAMPLE_RATE = 16000


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-8:
        return v.astype(np.float32)
    return (v / n).astype(np.float32)


def _as_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float32)
    return np.mean(audio.astype(np.float32), axis=1)


def preprocess_audio(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
    if librosa is None:
        raise ImportError("librosa is required for voice feature extraction.")

    y = _as_mono(audio)
    if y.size == 0:
        return np.zeros((0,), dtype=np.float32), TARGET_SAMPLE_RATE
    if sample_rate != TARGET_SAMPLE_RATE:
        y = librosa.resample(y=y, orig_sr=sample_rate, target_sr=TARGET_SAMPLE_RATE)
    y = y - np.mean(y)
    y = y / (np.std(y) + 1e-8)

    # Lightweight spectral denoise.
    stft = librosa.stft(y, n_fft=512, hop_length=160, win_length=400)
    mag = np.abs(stft)
    phase = np.exp(1j * np.angle(stft))
    noise_floor = np.percentile(mag, 20, axis=1, keepdims=True)
    cleaned_mag = np.maximum(mag - noise_floor, 0.0)
    y_clean = librosa.istft(cleaned_mag * phase, hop_length=160, win_length=400, length=len(y))
    return y_clean.astype(np.float32), TARGET_SAMPLE_RATE


@dataclass
class VoiceFeatureBundle:
    embedding: np.ndarray
    mfcc_mean: np.ndarray
    mfcc_std: np.ndarray
    pitch_mean: float
    pitch_std: float
    pitch_min: float
    pitch_max: float
    energy_mean: float
    energy_std: float
    frame_count: int

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "embedding": self.embedding.tolist(),
            "mfcc_mean": self.mfcc_mean.tolist(),
            "mfcc_std": self.mfcc_std.tolist(),
            "pitch_mean": self.pitch_mean,
            "pitch_std": self.pitch_std,
            "pitch_min": self.pitch_min,
            "pitch_max": self.pitch_max,
            "energy_mean": self.energy_mean,
            "energy_std": self.energy_std,
            "frame_count": self.frame_count,
        }

    @staticmethod
    def from_json_dict(data: dict[str, Any]) -> "VoiceFeatureBundle":
        return VoiceFeatureBundle(
            embedding=np.array(data["embedding"], dtype=np.float32),
            mfcc_mean=np.array(data["mfcc_mean"], dtype=np.float32),
            mfcc_std=np.array(data["mfcc_std"], dtype=np.float32),
            pitch_mean=float(data["pitch_mean"]),
            pitch_std=float(data["pitch_std"]),
            pitch_min=float(data["pitch_min"]),
            pitch_max=float(data["pitch_max"]),
            energy_mean=float(data["energy_mean"]),
            energy_std=float(data["energy_std"]),
            frame_count=int(data["frame_count"]),
        )


def extract_voice_features(audio: np.ndarray, sample_rate: int) -> VoiceFeatureBundle:
    y, sr = preprocess_audio(audio, sample_rate)
    if y.size == 0:
        return VoiceFeatureBundle(
            embedding=np.zeros((124,), dtype=np.float32),
            mfcc_mean=np.zeros((60,), dtype=np.float32),
            mfcc_std=np.zeros((60,), dtype=np.float32),
            pitch_mean=0.0,
            pitch_std=0.0,
            pitch_min=0.0,
            pitch_max=0.0,
            energy_mean=0.0,
            energy_std=0.0,
            frame_count=0,
        )

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, n_fft=400, hop_length=160)
    delta = librosa.feature.delta(mfcc, order=1)
    delta2 = librosa.feature.delta(mfcc, order=2)
    full = np.concatenate([mfcc, delta, delta2], axis=0)

    mfcc_mean = np.mean(full, axis=1).astype(np.float32)
    mfcc_std = np.std(full, axis=1).astype(np.float32)

    f0 = librosa.yin(y, fmin=65, fmax=350, sr=sr, frame_length=400, hop_length=160)
    valid_f0 = f0[np.isfinite(f0)]
    if valid_f0.size == 0:
        pitch_mean = pitch_std = pitch_min = pitch_max = 0.0
    else:
        pitch_mean = float(np.mean(valid_f0))
        pitch_std = float(np.std(valid_f0))
        pitch_min = float(np.min(valid_f0))
        pitch_max = float(np.max(valid_f0))

    energy = librosa.feature.rms(y=y, frame_length=400, hop_length=160).flatten()
    energy_mean = float(np.mean(energy)) if energy.size else 0.0
    energy_std = float(np.std(energy)) if energy.size else 0.0

    emb = np.concatenate(
        [
            mfcc_mean,
            mfcc_std,
            np.array([pitch_mean, pitch_std, energy_mean, energy_std], dtype=np.float32),
        ]
    )
    emb = _l2_normalize(emb)

    return VoiceFeatureBundle(
        embedding=emb,
        mfcc_mean=mfcc_mean,
        mfcc_std=mfcc_std,
        pitch_mean=pitch_mean,
        pitch_std=pitch_std,
        pitch_min=pitch_min,
        pitch_max=pitch_max,
        energy_mean=energy_mean,
        energy_std=energy_std,
        frame_count=int(full.shape[1]),
    )


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(_l2_normalize(a), _l2_normalize(b)))

