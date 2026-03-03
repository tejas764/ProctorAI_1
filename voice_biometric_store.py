from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Optional

import numpy as np

from enrollment_questions import ENROLLMENT_QUESTIONS
from voice_features import VoiceFeatureBundle


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dumps(data: dict[str, Any]) -> str:
    return json.dumps(data, separators=(",", ":"), ensure_ascii=True)


@dataclass
class SpeakerProfile:
    user_id: str
    mean_embedding: np.ndarray
    embedding_variance: float
    pitch_min: float
    pitch_max: float
    base_threshold: float
    drift_threshold: float
    enrollment_complete: bool
    completed_at: Optional[str]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "mean_embedding": self.mean_embedding.tolist(),
            "embedding_variance": self.embedding_variance,
            "pitch_min": self.pitch_min,
            "pitch_max": self.pitch_max,
            "base_threshold": self.base_threshold,
            "drift_threshold": self.drift_threshold,
            "enrollment_complete": self.enrollment_complete,
            "completed_at": self.completed_at,
        }

    @staticmethod
    def from_json_dict(data: dict[str, Any]) -> "SpeakerProfile":
        return SpeakerProfile(
            user_id=str(data["user_id"]),
            mean_embedding=np.array(data["mean_embedding"], dtype=np.float32),
            embedding_variance=float(data["embedding_variance"]),
            pitch_min=float(data["pitch_min"]),
            pitch_max=float(data["pitch_max"]),
            base_threshold=float(data["base_threshold"]),
            drift_threshold=float(data["drift_threshold"]),
            enrollment_complete=bool(data["enrollment_complete"]),
            completed_at=data.get("completed_at"),
        )


class VoiceBiometricStore:
    def __init__(self, db_path: str = "proctorguard.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self.seed_questions()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS enrollment_questions (
                    question_id TEXT PRIMARY KEY,
                    question_text TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS enrollment_samples (
                    sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    question_id TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    audio_path TEXT NOT NULL,
                    feature_json TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(user_id),
                    FOREIGN KEY(question_id) REFERENCES enrollment_questions(question_id),
                    UNIQUE(user_id, question_id)
                );

                CREATE TABLE IF NOT EXISTS speaker_profiles (
                    user_id TEXT PRIMARY KEY,
                    profile_json TEXT NOT NULL,
                    enrollment_complete INTEGER NOT NULL,
                    completed_at TEXT,
                    FOREIGN KEY(user_id) REFERENCES users(user_id)
                );

                CREATE TABLE IF NOT EXISTS runtime_voice_matches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    timestamp_s REAL NOT NULL,
                    similarity REAL,
                    drift REAL,
                    decision TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    def seed_questions(self) -> None:
        with self._connect() as conn:
            for q in ENROLLMENT_QUESTIONS:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO enrollment_questions(question_id, question_text, created_at)
                    VALUES (?, ?, ?)
                    """,
                    (q.question_id, q.text, _utc_now_iso()),
                )

    def upsert_user(self, user_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO users(user_id, created_at)
                VALUES (?, ?)
                """,
                (user_id, _utc_now_iso()),
            )

    def get_questions(self) -> list[dict[str, str]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT question_id, question_text FROM enrollment_questions ORDER BY question_id"
            ).fetchall()
            return [{"question_id": str(r["question_id"]), "text": str(r["question_text"])} for r in rows]

    def save_question_sample(
        self,
        user_id: str,
        question_id: str,
        audio_path: str,
        recorded_at: str,
        features: VoiceFeatureBundle,
    ) -> None:
        self.upsert_user(user_id)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO enrollment_samples(user_id, question_id, recorded_at, audio_path, feature_json)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(user_id, question_id) DO UPDATE SET
                    recorded_at=excluded.recorded_at,
                    audio_path=excluded.audio_path,
                    feature_json=excluded.feature_json
                """,
                (user_id, question_id, recorded_at, audio_path, _json_dumps(features.to_json_dict())),
            )

    def get_user_question_features(self, user_id: str) -> dict[str, VoiceFeatureBundle]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT question_id, feature_json
                FROM enrollment_samples
                WHERE user_id = ?
                ORDER BY question_id
                """,
                (user_id,),
            ).fetchall()
        out: dict[str, VoiceFeatureBundle] = {}
        for r in rows:
            out[str(r["question_id"])] = VoiceFeatureBundle.from_json_dict(json.loads(str(r["feature_json"])))
        return out

    def save_profile(self, profile: SpeakerProfile) -> None:
        self.upsert_user(profile.user_id)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO speaker_profiles(user_id, profile_json, enrollment_complete, completed_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    profile_json=excluded.profile_json,
                    enrollment_complete=excluded.enrollment_complete,
                    completed_at=excluded.completed_at
                """,
                (
                    profile.user_id,
                    _json_dumps(profile.to_json_dict()),
                    int(profile.enrollment_complete),
                    profile.completed_at,
                ),
            )

    def load_profile(self, user_id: str) -> Optional[SpeakerProfile]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT profile_json, enrollment_complete, completed_at
                FROM speaker_profiles
                WHERE user_id = ?
                """,
                (user_id,),
            ).fetchone()
        if row is None:
            return None
        data = json.loads(str(row["profile_json"]))
        data["enrollment_complete"] = bool(row["enrollment_complete"])
        data["completed_at"] = row["completed_at"]
        return SpeakerProfile.from_json_dict(data)

    def mark_incomplete(self, user_id: str) -> None:
        profile = SpeakerProfile(
            user_id=user_id,
            mean_embedding=np.zeros((124,), dtype=np.float32),
            embedding_variance=1.0,
            pitch_min=0.0,
            pitch_max=0.0,
            base_threshold=1.0,
            drift_threshold=0.0,
            enrollment_complete=False,
            completed_at=None,
        )
        self.save_profile(profile)

    def log_runtime_match(
        self,
        user_id: str,
        timestamp_s: float,
        similarity: Optional[float],
        drift: Optional[float],
        decision: str,
        reason: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO runtime_voice_matches(user_id, timestamp_s, similarity, drift, decision, reason, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (user_id, float(timestamp_s), similarity, drift, decision, reason, _utc_now_iso()),
            )

