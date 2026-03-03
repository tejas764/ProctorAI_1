from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import wave

import numpy as np

from enrollment_questions import ENROLLMENT_QUESTIONS
from voice_biometric_store import SpeakerProfile, VoiceBiometricStore
from voice_features import VoiceFeatureBundle, cosine_similarity, extract_voice_features


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _save_wav(audio: np.ndarray, sample_rate: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mono = audio if audio.ndim == 1 else np.mean(audio, axis=1)
    clipped = np.clip(mono.astype(np.float32), -1.0, 1.0)
    pcm = (clipped * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


@dataclass
class EnrollmentAudioSample:
    question_id: str
    audio: np.ndarray
    sample_rate: int
    timestamp_iso: str


@dataclass
class EnrollmentResult:
    user_id: str
    processed_questions: int
    enrollment_complete: bool
    base_threshold: float
    drift_threshold: float
    profile: SpeakerProfile | None


class VoiceEnrollmentService:
    def __init__(
        self,
        store: VoiceBiometricStore,
        audio_dir: str = "proctor_data/enrollment_audio",
    ) -> None:
        self.store = store
        self.audio_dir = Path(audio_dir)

    def questions(self) -> list[dict[str, str]]:
        return self.store.get_questions()

    def enroll_user(self, user_id: str, samples: list[EnrollmentAudioSample]) -> EnrollmentResult:
        question_ids = {q.question_id for q in ENROLLMENT_QUESTIONS}
        by_question = {s.question_id: s for s in samples if s.question_id in question_ids}

        features_by_q: dict[str, VoiceFeatureBundle] = {}
        for q in ENROLLMENT_QUESTIONS:
            sample = by_question.get(q.question_id)
            if sample is None:
                continue
            file_path = self.audio_dir / user_id / f"{q.question_id}_{sample.timestamp_iso.replace(':', '-')}.wav"
            _save_wav(sample.audio, sample.sample_rate, file_path)
            feats = extract_voice_features(sample.audio, sample.sample_rate)
            features_by_q[q.question_id] = feats
            self.store.save_question_sample(
                user_id=user_id,
                question_id=q.question_id,
                audio_path=str(file_path),
                recorded_at=sample.timestamp_iso,
                features=feats,
            )

        if len(features_by_q) != len(ENROLLMENT_QUESTIONS):
            self.store.mark_incomplete(user_id)
            return EnrollmentResult(
                user_id=user_id,
                processed_questions=len(features_by_q),
                enrollment_complete=False,
                base_threshold=1.0,
                drift_threshold=0.0,
                profile=None,
            )

        profile = self._build_profile(user_id, features_by_q)
        self.store.save_profile(profile)
        return EnrollmentResult(
            user_id=user_id,
            processed_questions=len(features_by_q),
            enrollment_complete=profile.enrollment_complete,
            base_threshold=profile.base_threshold,
            drift_threshold=profile.drift_threshold,
            profile=profile,
        )

    def _build_profile(self, user_id: str, features_by_q: dict[str, VoiceFeatureBundle]) -> SpeakerProfile:
        ordered = [features_by_q[q.question_id] for q in ENROLLMENT_QUESTIONS]
        embeddings = np.stack([f.embedding for f in ordered], axis=0)
        mean_embedding = np.mean(embeddings, axis=0).astype(np.float32)

        sims = np.array([cosine_similarity(mean_embedding, e) for e in embeddings], dtype=np.float32)
        sim_mean = float(np.mean(sims))
        sim_std = float(np.std(sims))

        # Base threshold is conservative enough to detect speaker changes, while tolerating natural variation.
        base_threshold = float(max(0.60, min(0.95, sim_mean - 2.0 * sim_std)))
        embedding_variance = float(np.var(sims))
        drift_threshold = float(max(0.04, min(0.20, 1.5 * sim_std + 0.03)))

        pitch_min = float(min(f.pitch_min for f in ordered))
        pitch_max = float(max(f.pitch_max for f in ordered))

        return SpeakerProfile(
            user_id=user_id,
            mean_embedding=mean_embedding,
            embedding_variance=embedding_variance,
            pitch_min=pitch_min,
            pitch_max=pitch_max,
            base_threshold=base_threshold,
            drift_threshold=drift_threshold,
            enrollment_complete=True,
            completed_at=_utc_now_iso(),
        )

