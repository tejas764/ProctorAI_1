from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import numpy as np


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, x)))


@dataclass
class MultiSpeakerEstimate:
    speaker_count: int
    confidence: float
    reason: str


@dataclass
class WindowDecision:
    similarity_score: float
    drift_score: float
    lip_sync_score: float
    active_speaker_prob: float
    fused_score: float
    state: str
    anomaly: bool
    reason: str


def estimate_speaker_count(audio: np.ndarray, sample_rate: int) -> MultiSpeakerEstimate:
    """Heuristic multi-speaker detector for short windows.

    Returns 2 only on strong evidence to reduce false positives from outdoor noise.
    """
    if audio.size < int(sample_rate * 0.5):
        return MultiSpeakerEstimate(speaker_count=1, confidence=0.0, reason="insufficient_audio")

    y = audio.astype(np.float32)
    y = y - np.mean(y)
    rms = float(np.sqrt(np.mean(y * y)))
    if rms < 0.01:
        return MultiSpeakerEstimate(speaker_count=1, confidence=0.0, reason="low_energy")

    win = np.hanning(y.size)
    mag = np.abs(np.fft.rfft(y * win))
    if mag.size < 64:
        return MultiSpeakerEstimate(speaker_count=1, confidence=0.0, reason="short_fft")

    freqs = np.fft.rfftfreq(y.size, d=1.0 / sample_rate)
    band = (freqs >= 80.0) & (freqs <= 1200.0)
    band_mag = mag[band]
    band_freq = freqs[band]
    if band_mag.size < 16:
        return MultiSpeakerEstimate(speaker_count=1, confidence=0.0, reason="narrow_band")

    # Strong dual-peak detection in speech band.
    idx = np.argsort(band_mag)[-8:]
    top_freqs = np.sort(band_freq[idx])
    top_amps = np.sort(band_mag[idx])[::-1]
    if top_amps.size < 2:
        return MultiSpeakerEstimate(speaker_count=1, confidence=0.0, reason="single_peak")

    amp_ratio = float(top_amps[1] / (top_amps[0] + 1e-8))
    freq_sep = float(np.max(np.diff(top_freqs))) if top_freqs.size > 1 else 0.0

    # Harmonic flatness; multi-speaker tends to flatten formant dominance.
    gm = float(np.exp(np.mean(np.log(band_mag + 1e-8))))
    am = float(np.mean(band_mag) + 1e-8)
    flatness = gm / am

    multi_evidence = 0
    if amp_ratio > 0.72 and freq_sep > 120.0:
        multi_evidence += 1
    if flatness > 0.52:
        multi_evidence += 1

    if multi_evidence >= 2:
        conf = _clamp(0.55 + 0.35 * amp_ratio + 0.2 * min(1.0, freq_sep / 500.0))
        return MultiSpeakerEstimate(speaker_count=2, confidence=conf, reason="dual_voice_pattern")

    return MultiSpeakerEstimate(speaker_count=1, confidence=0.75, reason="single_dominant_voice")


class DriftTracker:
    def __init__(self, maxlen: int = 20) -> None:
        self.history: deque[float] = deque(maxlen=maxlen)

    def update(self, similarity: float, drift_threshold: float) -> tuple[float, str]:
        self.history.append(similarity)
        if len(self.history) < 4:
            return 0.0, "Stable"

        arr = np.array(self.history, dtype=np.float32)
        ema = float(np.mean(arr[-6:]))
        drift = float(abs(similarity - ema))

        if drift <= drift_threshold:
            return drift, "Stable"
        if drift <= (drift_threshold * 1.6):
            return drift, "Drifting"
        return drift, "Unstable"


def distribution_similarity_score(similarity: float, base_threshold: float, embedding_variance: float) -> tuple[float, bool]:
    # Wider acceptance for naturally variable speakers; still bounded for security.
    sigma = math.sqrt(max(1e-6, embedding_variance))
    tol = min(0.12, max(0.03, 0.8 * sigma))
    accept_thr = max(0.0, base_threshold - tol)

    # Scale score around acceptance threshold and 1.0.
    norm = (similarity - accept_thr) / max(1e-6, (1.0 - accept_thr))
    score = _clamp(norm)
    return score, bool(similarity >= accept_thr)


def soft_pitch_match(pitch_hz: float | None, pitch_min: float | None, pitch_max: float | None) -> bool | None:
    if pitch_hz is None or pitch_hz <= 0.0 or pitch_min is None or pitch_max is None:
        return None
    span = max(20.0, pitch_max - pitch_min)
    margin = max(20.0, 0.30 * span)
    return bool((pitch_min - margin) <= pitch_hz <= (pitch_max + margin))


def fuse_window_decision(
    *,
    similarity_score: float,
    drift_score: float,
    lip_sync_score: float,
    active_speaker_prob: float,
    single_face: bool,
    speaker_count: int,
    hard_mismatch: bool,
) -> WindowDecision:
    # Weighted biometric fusion.
    fused = (
        0.45 * similarity_score
        + 0.20 * drift_score
        + 0.20 * lip_sync_score
        + 0.15 * active_speaker_prob
    )

    if speaker_count > 1:
        return WindowDecision(
            similarity_score=similarity_score,
            drift_score=drift_score,
            lip_sync_score=lip_sync_score,
            active_speaker_prob=active_speaker_prob,
            fused_score=fused,
            state="RED",
            anomaly=True,
            reason="multiple_speakers_detected",
        )

    if hard_mismatch or (not single_face):
        return WindowDecision(
            similarity_score=similarity_score,
            drift_score=drift_score,
            lip_sync_score=lip_sync_score,
            active_speaker_prob=active_speaker_prob,
            fused_score=fused,
            state="RED",
            anomaly=True,
            reason="ownership_or_identity_violation",
        )

    if fused >= 0.62:
        return WindowDecision(
            similarity_score=similarity_score,
            drift_score=drift_score,
            lip_sync_score=lip_sync_score,
            active_speaker_prob=active_speaker_prob,
            fused_score=fused,
            state="GREEN",
            anomaly=False,
            reason="verified",
        )
    if fused >= 0.46:
        return WindowDecision(
            similarity_score=similarity_score,
            drift_score=drift_score,
            lip_sync_score=lip_sync_score,
            active_speaker_prob=active_speaker_prob,
            fused_score=fused,
            state="YELLOW",
            anomaly=False,
            reason="voice_drifting",
        )

    return WindowDecision(
        similarity_score=similarity_score,
        drift_score=drift_score,
        lip_sync_score=lip_sync_score,
        active_speaker_prob=active_speaker_prob,
        fused_score=fused,
        state="RED",
        anomaly=True,
        reason="low_fused_confidence",
    )
