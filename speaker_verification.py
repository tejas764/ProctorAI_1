from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Optional

import numpy as np

from voice_biometric_store import SpeakerProfile, VoiceBiometricStore
from voice_features import cosine_similarity, extract_voice_features

def _l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-8:
        return v
    return v / n


def _frame_signal(signal: np.ndarray, frame_size: int, hop: int) -> list[np.ndarray]:
    if len(signal) < frame_size:
        return []
    return [signal[i : i + frame_size] for i in range(0, len(signal) - frame_size + 1, hop)]


def simple_speaker_embedding(audio: np.ndarray, sr: int) -> np.ndarray:
    # Lightweight fallback embedding: log-band spectral profile + temporal stats.
    audio = audio.astype(np.float32)
    if audio.size == 0:
        return np.zeros(64, dtype=np.float32)
    audio = audio - np.mean(audio)
    audio = audio / (np.std(audio) + 1e-8)

    frames = _frame_signal(audio, frame_size=int(0.03 * sr), hop=int(0.01 * sr))
    if not frames:
        return np.zeros(64, dtype=np.float32)

    specs = []
    for frame in frames:
        win = frame * np.hanning(len(frame))
        spectrum = np.abs(np.fft.rfft(win))
        specs.append(np.log1p(spectrum))
    spec = np.stack(specs, axis=0)

    # 32 coarse frequency bins.
    bins = np.array_split(spec.mean(axis=0), 32)
    band_means = np.array([float(np.mean(b)) for b in bins], dtype=np.float32)
    band_stds = np.array([float(np.std(b)) for b in bins], dtype=np.float32)
    emb = np.concatenate([band_means, band_stds], axis=0)
    return _l2_normalize(emb.astype(np.float32))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_n = _l2_normalize(a)
    b_n = _l2_normalize(b)
    return float(np.dot(a_n, b_n))


@dataclass
class SpeakerVerificationResult:
    similarity: Optional[float]
    drift: Optional[float]
    is_mismatch: bool
    is_drift: bool
    has_reference: bool
    decision: str
    reason: str
    threshold: Optional[float]
    drift_threshold: Optional[float]
    status_color: str


class SpeakerVerifier:
    def __init__(
        self,
        sample_rate: int,
        similarity_threshold: float = 0.72,
        drift_threshold: float = 0.08,
        window_seconds: float = 2.5,
        user_id: str = "default_user",
        db_path: str = "proctorguard.db",
    ) -> None:
        self.sample_rate = sample_rate
        self.similarity_threshold = similarity_threshold
        self.drift_threshold = drift_threshold
        self.window_seconds = window_seconds
        self.user_id = user_id
        self.store = VoiceBiometricStore(db_path=db_path)
        self.profile: Optional[SpeakerProfile] = self.store.load_profile(user_id)
        self.similarity_history: deque[float] = deque(maxlen=24)

    def _effective_thresholds(self) -> tuple[float, float]:
        if self.profile is None:
            return self.similarity_threshold, self.drift_threshold
        return self.profile.base_threshold, self.profile.drift_threshold

    def reload_profile(self) -> None:
        self.profile = self.store.load_profile(self.user_id)

    def verify(self, audio: np.ndarray, audio_present: bool, timestamp_s: float) -> SpeakerVerificationResult:
        if not audio_present:
            return SpeakerVerificationResult(
                similarity=None,
                drift=None,
                is_mismatch=False,
                is_drift=False,
                has_reference=self.profile is not None and self.profile.enrollment_complete,
                decision="NO_SPEECH",
                reason="audio_not_present",
                threshold=self._effective_thresholds()[0],
                drift_threshold=self._effective_thresholds()[1],
                status_color="green",
            )

        if self.profile is None or not self.profile.enrollment_complete:
            res = SpeakerVerificationResult(
                similarity=None,
                drift=None,
                is_mismatch=True,
                is_drift=False,
                has_reference=False,
                decision="VIOLATION",
                reason="enrollment_incomplete_or_missing",
                threshold=self._effective_thresholds()[0],
                drift_threshold=self._effective_thresholds()[1],
                status_color="red",
            )
            self.store.log_runtime_match(
                user_id=self.user_id,
                timestamp_s=timestamp_s,
                similarity=None,
                drift=None,
                decision=res.decision,
                reason=res.reason,
            )
            return res

        features = extract_voice_features(audio, self.sample_rate)
        sim = cosine_similarity(self.profile.mean_embedding, features.embedding)
        self.similarity_history.append(sim)
        trend = float(np.mean(self.similarity_history)) if self.similarity_history else sim
        drift = float(abs(sim - trend))

        threshold, drift_threshold = self._effective_thresholds()
        # Allow natural short-term voice variation before declaring mismatch.
        mismatch_margin = 0.04
        effective_mismatch_threshold = max(0.0, threshold - mismatch_margin)
        mismatch = sim < effective_mismatch_threshold

        # Drift detection combines temporal instability and tolerant pitch-range checks.
        pitch_span = max(20.0, self.profile.pitch_max - self.profile.pitch_min)
        pitch_margin = max(20.0, 0.25 * pitch_span)
        pitch_out_of_range = bool(
            features.pitch_mean > 0.0
            and (
                features.pitch_mean < (self.profile.pitch_min - pitch_margin)
                or features.pitch_mean > (self.profile.pitch_max + pitch_margin)
            )
        )
        stable_history = len(self.similarity_history) >= 6
        effective_drift_threshold = max(0.12, drift_threshold)
        is_drift = (stable_history and drift > effective_drift_threshold) or pitch_out_of_range

        if mismatch:
            decision = "VIOLATION"
            reason = "different_speaker_detected"
            color = "red"
        elif is_drift:
            decision = "VIOLATION"
            reason = "voice_drift_detected"
            color = "yellow"
        else:
            decision = "MATCH"
            reason = "base_case_match"
            color = "green"

        res = SpeakerVerificationResult(
            similarity=sim,
            drift=drift,
            is_mismatch=mismatch,
            is_drift=is_drift,
            has_reference=True,
            decision=decision,
            reason=reason,
            threshold=threshold,
            drift_threshold=drift_threshold,
            status_color=color,
        )
        self.store.log_runtime_match(
            user_id=self.user_id,
            timestamp_s=timestamp_s,
            similarity=sim,
            drift=drift,
            decision=res.decision,
            reason=res.reason,
        )
        return res
