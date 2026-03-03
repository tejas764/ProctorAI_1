from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import wave

import numpy as np

from voice_biometric_store import VoiceBiometricStore
from voice_enrollment import VoiceEnrollmentService


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_wav_bytes(raw: bytes) -> tuple[np.ndarray, int]:
    import io

    with wave.open(io.BytesIO(raw), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        frames = wf.getnframes()
        pcm = wf.readframes(frames)

    if sample_width != 2:
        raise ValueError("Only 16-bit PCM WAV is supported.")

    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, sample_rate


def save_wav(audio: np.ndarray, sample_rate: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(audio.astype(np.float32), -1.0, 1.0)
    pcm = (clipped * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


class EnrollmentApi:
    def __init__(self, db_path: str = "proctorguard.db") -> None:
        self.store = VoiceBiometricStore(db_path=db_path)
        self.service = VoiceEnrollmentService(store=self.store)

    def finalize_enrollment(self, user_id: str) -> dict[str, object]:
        features_by_q = self.store.get_user_question_features(user_id)
        questions = self.service.questions()
        question_ids = {q["question_id"] for q in questions}

        if set(features_by_q.keys()) != question_ids:
            self.store.mark_incomplete(user_id)
            return {
                "enrollment_complete": False,
                "error": "All 10 questions must be recorded before completion.",
            }

        profile = self.service._build_profile(user_id, features_by_q)  # noqa: SLF001
        self.store.save_profile(profile)

        quality_score = round(max(0.0, min(100.0, (1.0 - profile.embedding_variance) * 100.0)), 2)
        confidence = round(profile.base_threshold * 100.0, 2)
        return {
            "enrollment_complete": bool(profile.enrollment_complete),
            "voice_quality_score": quality_score,
            "enrollment_confidence": confidence,
            "monitor_url": f"/monitor?user_id={user_id}",
        }
