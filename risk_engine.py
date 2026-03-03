from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import wave
from typing import Optional

import cv2
import numpy as np


RISK_WEIGHTS = {
    "AUDIO_ONLY": 3,
    "AUDIO_WITHOUT_LIP_MOTION": 3,
    "AUDIO_ONLY_SPEECH": 3,
    "AV_DESYNC": 2,
    "WHISPER_DETECTED": 2,
    "PHONEME_VISEME_MISMATCH": 3,
    "POSSIBLE_AUDIO_PLAYBACK": 3,
    "AUDIO_SYNC_LOW": 2,
    "HAND_OCCLUSION": 2,
    "VOICE_MISMATCH": 5,
    "LIPSYNC_VERIFICATION_FAIL": 4,
    "MULTIPLE_FACES": 5,
    "ENROLLMENT_MISSING": 6,
    "VOICE_DRIFT": 4,
    "VOICE_POLICY_WARNING": 2,
    "VOICE_POLICY_SOFT_FLAG": 4,
    "CHEATING_ALERT": 8,
}


@dataclass
class EventRecord:
    timestamp_s: float
    reason: str
    risk_delta: int
    risk_total: int
    frame_path: Optional[str]
    audio_path: Optional[str]
    details: dict


class RiskEngine:
    def __init__(self, log_dir: str = "proctor_logs") -> None:
        self.risk_score = 0
        self.events: list[EventRecord] = []
        self.log_dir = Path(log_dir)
        self.frame_dir = self.log_dir / "frames"
        self.audio_dir = self.log_dir / "audio"
        self.frame_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)

    def level(self) -> str:
        if self.risk_score >= 8:
            return "CHEATING_LIKELY"
        if self.risk_score >= 4:
            return "WARNING"
        return "NORMAL"

    def _save_frame(self, frame: Optional[np.ndarray], key: str) -> Optional[str]:
        if frame is None:
            return None
        path = self.frame_dir / f"{key}.jpg"
        cv2.imwrite(str(path), frame)
        return str(path)

    def _save_audio(self, audio: Optional[np.ndarray], sample_rate: int, key: str) -> Optional[str]:
        if audio is None or audio.size == 0:
            return None
        wav_path = self.audio_dir / f"{key}.wav"
        clipped = np.clip(audio, -1.0, 1.0)
        pcm = (clipped * 32767).astype(np.int16)
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm.tobytes())
        return str(wav_path)

    def add_event(
        self,
        reason: str,
        timestamp_s: float,
        frame: Optional[np.ndarray],
        audio: Optional[np.ndarray],
        sample_rate: int,
        details: Optional[dict] = None,
    ) -> EventRecord:
        delta = int(RISK_WEIGHTS.get(reason, 1))
        self.risk_score += delta
        key = f"{int(timestamp_s * 1000)}_{reason.lower()}"
        frame_path = self._save_frame(frame, key)
        audio_path = self._save_audio(audio, sample_rate, key)
        rec = EventRecord(
            timestamp_s=timestamp_s,
            reason=reason,
            risk_delta=delta,
            risk_total=self.risk_score,
            frame_path=frame_path,
            audio_path=audio_path,
            details=details or {},
        )
        self.events.append(rec)
        return rec

    def export_json(self, filename: str = "events.json") -> str:
        output = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "risk_score": self.risk_score,
            "risk_level": self.level(),
            "events": [asdict(e) for e in self.events],
        }
        path = self.log_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        return str(path)
