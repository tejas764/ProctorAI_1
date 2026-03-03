from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import threading
import time

import cv2
import numpy as np

from audio_sync_verification import AudioSyncResult, AudioSyncVerifier
from av_correlation import AVCorrelationEngine
from lip_sync_verification import LipSyncVerifier
from risk_engine import RiskEngine
from speaker_verification import SpeakerVerificationResult, SpeakerVerifier
from voice_biometric_store import VoiceBiometricStore
from voice_features import extract_voice_features
from web_modules.audio import AudioMonitor
from web_modules.verification_logic import (
    DriftTracker,
    MultiSpeakerEstimate,
    WindowDecision,
    distribution_similarity_score,
    estimate_speaker_count,
    fuse_window_decision,
    soft_pitch_match,
)
from chunks_modules.shared import create_face_mesh_backend


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def compute_mar(face_landmarks: object) -> float:
    pts = face_landmarks.landmark
    upper_inner = pts[13]
    lower_inner = pts[14]
    left_corner = pts[78]
    right_corner = pts[308]
    vertical = abs(lower_inner.y - upper_inner.y)
    horizontal = max(abs(right_corner.x - left_corner.x), 1e-6)
    return float(vertical / horizontal)


class MonitoringWorker:
    def __init__(
        self,
        store: VoiceBiometricStore,
        camera_index: int = 0,
        sample_rate: int = 16000,
        block_size: int = 1024,
        vad_threshold: float = 0.015,
        audio_sync_low_score_threshold: float = 0.40,
        suspicious_streak_for_verify: int = 8,
    ) -> None:
        self.store = store
        self.camera_index = camera_index
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.vad_threshold = vad_threshold
        self.audio_sync_low_score_threshold = audio_sync_low_score_threshold
        self.suspicious_streak_for_verify = suspicious_streak_for_verify

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._flag_cooldown: dict[str, float] = {}
        self._user_id = ""
        self._latest_jpeg: bytes = b""
        self._state: dict[str, object] = {
            "running": False,
            "user_id": "",
            "flags": 0,
            "faces": 0,
            "audio_present": False,
            "lip_sync_status": "IDLE",
            "speaker_decision": "UNKNOWN",
            "speaker_reason": "not_started",
            "speaker_similarity": None,
            "speaker_threshold": None,
            "voice_drift": None,
            "drift_threshold": None,
            "pitch_hz": None,
            "pitch_min_hz": None,
            "pitch_max_hz": None,
            "pitch_match": None,
            "frequency_match": None,
            "audio_sync_score": None,
            "audio_sync_flags": [],
            "av_offset_ms": None,
            "verify_score": None,
            "risk_score": 0,
            "risk_level": "NORMAL",
            "multiple_voice_suspected": False,
            "multiple_voice_reason": "",
            "speaker_count": 1,
            "speaker_count_confidence": 0.0,
            "speaker_similarity_bar": 0.0,
            "voice_stability": "Stable",
            "active_speaker_status": "UNKNOWN",
            "face_count_status": "UNKNOWN",
            "verification_state": "YELLOW",
            "escalation_level": "NORMAL",
            "anomaly_streak": 0,
            "last_flag_reason": "",
            "updated_at": _utc_now_iso(),
            "error": "",
        }

    def _update_state(self, **kwargs: object) -> None:
        with self._lock:
            self._state.update(kwargs)
            self._state["updated_at"] = _utc_now_iso()

    def get_state(self) -> dict[str, object]:
        with self._lock:
            return dict(self._state)

    def get_latest_jpeg(self) -> bytes:
        with self._lock:
            return self._latest_jpeg

    def _flag_event(
        self,
        reason: str,
        risk: RiskEngine,
        frame: np.ndarray,
        audio: np.ndarray,
        details: dict[str, object] | None = None,
        cooldown_s: float = 2.0,
    ) -> None:
        now = time.time()
        last = self._flag_cooldown.get(reason, -999.0)
        if (now - last) < cooldown_s:
            return
        self._flag_cooldown[reason] = now
        risk.add_event(
            reason=reason,
            timestamp_s=now,
            frame=frame,
            audio=audio,
            sample_rate=self.sample_rate,
            details=details or {},
        )
        with self._lock:
            self._state["flags"] = int(self._state.get("flags", 0)) + 1
            self._state["last_flag_reason"] = reason
            self._state["risk_score"] = int(risk.risk_score)
            self._state["risk_level"] = risk.level()
            self._state["updated_at"] = _utc_now_iso()

    @staticmethod
    def _update_counter(counters: dict[str, int], key: str, active: bool) -> int:
        value = counters.get(key, 0) + 1 if active else 0
        counters[key] = value
        return value

    def start(self, user_id: str) -> tuple[bool, str]:
        if self._thread and self._thread.is_alive():
            return False, "Monitoring already running."

        profile = self.store.load_profile(user_id)
        if profile is None or not profile.enrollment_complete:
            return False, "Enrollment incomplete. Complete voice enrollment first."

        self._user_id = user_id
        self._flag_cooldown = {}
        self._stop.clear()
        self._update_state(
            running=True,
            user_id=user_id,
            flags=0,
            error="",
            last_flag_reason="",
            risk_score=0,
            risk_level="NORMAL",
        )
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return True, "Monitoring started."

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._update_state(running=False)

    def _run(self) -> None:
        try:
            cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(self.camera_index)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open camera index {self.camera_index}")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
            cap.set(cv2.CAP_PROP_FPS, 30)

            mic = AudioMonitor(self.sample_rate, self.block_size, self.vad_threshold)
            mic.start()

            speaker = SpeakerVerifier(
                sample_rate=self.sample_rate,
                user_id=self._user_id,
                db_path=str(self.store.db_path),
            )
            av_engine = AVCorrelationEngine()
            audio_sync = AudioSyncVerifier(sample_rate=self.sample_rate)
            lip_verifier = LipSyncVerifier()
            risk = RiskEngine(log_dir="proctor_logs")
            face_mesh, _ = create_face_mesh_backend()
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            face_cascade_alt = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
            motion_series: deque[float] = deque(maxlen=120)
            audio_series: deque[float] = deque(maxlen=120)
            suspicious_streak = 0
            stable_faces = 0
            stable_face_streak = 0
            last_face_count = -1
            last_verify_t = 0.0
            verify_interval_s = 1.0
            spk_cache = SpeakerVerificationResult(
                similarity=None,
                drift=None,
                is_mismatch=False,
                is_drift=False,
                has_reference=True,
                decision="NO_SPEECH",
                reason="warming_up",
                threshold=None,
                drift_threshold=None,
                status_color="green",
            )
            drift_tracker = DriftTracker(maxlen=24)
            multi_cache = MultiSpeakerEstimate(speaker_count=1, confidence=0.0, reason="warming_up")
            decision_cache = WindowDecision(
                similarity_score=0.0,
                drift_score=0.0,
                lip_sync_score=0.0,
                active_speaker_prob=0.0,
                fused_score=0.0,
                state="YELLOW",
                anomaly=False,
                reason="warming_up",
            )
            anomaly_streak = 0
            pitch_hz = None
            pitch_match = None
            frequency_match = None
            pitch_min_hz = None
            pitch_max_hz = None

            try:
                while not self._stop.is_set():
                    ok, frame = cap.read()
                    if not ok:
                        time.sleep(0.05)
                        continue

                    frame = cv2.flip(frame, 1)
                    faces = []
                    num_faces = 0
                    if face_mesh is not None:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        res = face_mesh.process(rgb)
                        faces = res.multi_face_landmarks if res.multi_face_landmarks else []
                        num_faces = len(faces)
                    else:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        gray_eq = cv2.equalizeHist(gray)
                        # Faster fallback: detect on downscaled frame.
                        small = cv2.resize(gray_eq, (0, 0), fx=0.5, fy=0.5)
                        det = face_cascade.detectMultiScale(small, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
                        if len(det) == 0:
                            det = face_cascade_alt.detectMultiScale(small, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
                        num_faces = int(len(det))

                    if num_faces == last_face_count:
                        stable_face_streak += 1
                    else:
                        stable_face_streak = 1
                        last_face_count = num_faces
                    # Require a few stable frames before switching face-count state.
                    if stable_face_streak >= 3:
                        stable_faces = num_faces

                    audio_present = mic.vad()
                    audio_rms = mic.rms()
                    now_t = time.time()
                    lip_sync_status = "NO_FACE"
                    mar = 0.0
                    mar_delta = 0.0
                    corr_score = None
                    if stable_faces == 1:
                        if face_mesh is not None and faces:
                            mar = compute_mar(faces[0])
                            av = av_engine.update(audio_present, audio_rms, mar)
                            lip_sync_status = av.status
                            mar_delta = float(av.mar_delta)
                            corr_score = av.corr_score
                        else:
                            # Without facial landmarks, lip sync should stay uncertain to avoid false flags.
                            lip_sync_status = "UNCERTAIN_NO_LANDMARKS"
                    elif stable_faces > 1:
                        lip_sync_status = "MULTIPLE_FACES"

                    motion_series.append(mar_delta)
                    audio_series.append(audio_rms)

                    audio_chunk = mic.latest_seconds(0.35)
                    if face_mesh is not None and stable_faces == 1 and audio_present:
                        sync_res = audio_sync.update(
                            timestamp_s=now_t,
                            audio_chunk=audio_chunk,
                            audio_present=audio_present,
                            mar_value=mar,
                            mouth_occluded=False,
                        )
                    else:
                        sync_res = AudioSyncResult(
                            score=1.0,
                            flags=[],
                            energy_mar_corr=0.0,
                            offset_ms=None,
                            whisper=False,
                            viseme_mismatch_count=0,
                            playback_suspected=False,
                        )

                    if speaker.profile is not None:
                        pitch_min_hz = float(speaker.profile.pitch_min)
                        pitch_max_hz = float(speaker.profile.pitch_max)

                    if (now_t - last_verify_t) >= verify_interval_s:
                        # Overlapping sliding window verification.
                        window_audio = mic.latest_seconds(2.5)
                        spk_cache = speaker.verify(audio=window_audio, audio_present=audio_present, timestamp_s=now_t)
                        spk = spk_cache

                        if audio_present:
                            multi_cache = estimate_speaker_count(window_audio, self.sample_rate)
                        else:
                            multi_cache = MultiSpeakerEstimate(speaker_count=1, confidence=0.0, reason="no_speech")

                        if audio_present and window_audio.size >= int(self.sample_rate * 0.4):
                            feats = extract_voice_features(window_audio, self.sample_rate)
                            pitch_hz = float(feats.pitch_mean)
                            pitch_match = soft_pitch_match(pitch_hz, pitch_min_hz, pitch_max_hz)
                            frequency_match = pitch_match
                        else:
                            pitch_hz = None
                            pitch_match = None
                            frequency_match = None
                        if spk.similarity is not None and spk.threshold is not None:
                            emb_var = float(speaker.profile.embedding_variance) if speaker.profile is not None else 0.02
                            similarity_score, _ = distribution_similarity_score(spk.similarity, spk.threshold, emb_var)
                            drift_val, stability_label = drift_tracker.update(
                                spk.similarity,
                                max(0.10, spk.drift_threshold or 0.10),
                            )
                            drift_score = max(0.0, min(1.0, 1.0 - (drift_val / max(0.2, 2.0 * (spk.drift_threshold or 0.10)))))
                        else:
                            similarity_score = 0.0
                            drift_score = 0.5
                            stability_label = "Stable"

                        lip_sync_score = float(sync_res.score) if stable_faces == 1 else 0.4
                        active_prob = 1.0 if (audio_present and stable_faces == 1 and lip_sync_score >= 0.45) else 0.2

                        decision_cache = fuse_window_decision(
                            similarity_score=similarity_score,
                            drift_score=drift_score,
                            lip_sync_score=lip_sync_score,
                            active_speaker_prob=active_prob,
                            single_face=(stable_faces == 1),
                            speaker_count=multi_cache.speaker_count,
                            hard_mismatch=bool(spk.is_mismatch),
                        )

                        if multi_cache.speaker_count > 1:
                            self._flag_event(
                                "MULTIPLE_FACES",
                                risk,
                                frame,
                                window_audio,
                                {"reason": "multiple_speakers_detected", "confidence": multi_cache.confidence},
                            )

                        anomaly_streak = anomaly_streak + 1 if decision_cache.anomaly else max(0, anomaly_streak - 1)
                        escalation_level = "ALERT" if anomaly_streak >= 3 else ("WARNING" if anomaly_streak >= 2 else "NORMAL")

                        # Temporal escalation policy.
                        if anomaly_streak == 2:
                            self._flag_event(
                                "VOICE_POLICY_WARNING",
                                risk,
                                frame,
                                window_audio,
                                {"reason": decision_cache.reason, "streak": anomaly_streak},
                            )
                        elif anomaly_streak >= 3:
                            self._flag_event(
                                "CHEATING_ALERT",
                                risk,
                                frame,
                                window_audio,
                                {"reason": decision_cache.reason, "streak": anomaly_streak},
                            )

                        suspicious = decision_cache.anomaly or (sync_res.score < self.audio_sync_low_score_threshold) or (len(sync_res.flags) >= 2)
                        suspicious_streak = suspicious_streak + 1 if suspicious else 0
                        verify_score = None
                        if suspicious_streak >= self.suspicious_streak_for_verify and len(motion_series) >= 30:
                            vr = lip_verifier.verify_segment(
                                np.array(motion_series, dtype=np.float32),
                                np.array(audio_series, dtype=np.float32),
                            )
                            verify_score = float(vr.score)
                            if not vr.passed:
                                self._flag_event("LIPSYNC_VERIFICATION_FAIL", risk, frame, window_audio, {"score": vr.score})
                            suspicious_streak = 0
                        last_verify_t = now_t
                    else:
                        spk = spk_cache
                        stability_label = self._state.get("voice_stability", "Stable")
                        escalation_level = self._state.get("escalation_level", "NORMAL")
                        verify_score = self._state.get("verify_score")

                    multiple_voice_suspected = bool(multi_cache.speaker_count > 1)
                    multiple_voice_reason = multi_cache.reason if multiple_voice_suspected else ""
                    face_count_status = "SINGLE_FACE" if stable_faces == 1 else ("MULTIPLE_FACES" if stable_faces > 1 else "NO_FACE")
                    active_speaker_status = "BOUND" if (audio_present and stable_faces == 1 and lip_sync_status in ("SYNC_OK", "UNCERTAIN_NO_LANDMARKS")) else "UNBOUND"

                    ok_jpg, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 78])
                    if ok_jpg:
                        with self._lock:
                            self._latest_jpeg = encoded.tobytes()

                    self._update_state(
                        running=True,
                        user_id=self._user_id,
                        faces=stable_faces,
                        audio_present=audio_present,
                        lip_sync_status=lip_sync_status,
                        speaker_decision=spk.decision,
                        speaker_reason=spk.reason,
                        speaker_similarity=None if spk.similarity is None else float(spk.similarity),
                        speaker_threshold=None if spk.threshold is None else float(spk.threshold),
                        voice_drift=None if spk.drift is None else float(spk.drift),
                        drift_threshold=None if spk.drift_threshold is None else float(spk.drift_threshold),
                        pitch_hz=pitch_hz,
                        pitch_min_hz=pitch_min_hz,
                        pitch_max_hz=pitch_max_hz,
                        pitch_match=pitch_match,
                        frequency_match=frequency_match,
                        audio_sync_score=float(sync_res.score),
                        audio_sync_flags=list(sync_res.flags),
                        av_offset_ms=None if sync_res.offset_ms is None else float(sync_res.offset_ms),
                        verify_score=verify_score,
                        risk_score=int(risk.risk_score),
                        risk_level=risk.level(),
                        multiple_voice_suspected=multiple_voice_suspected,
                        multiple_voice_reason=multiple_voice_reason,
                        speaker_count=int(multi_cache.speaker_count),
                        speaker_count_confidence=float(multi_cache.confidence),
                        speaker_similarity_bar=float(decision_cache.similarity_score),
                        voice_stability=str(stability_label),
                        active_speaker_status=active_speaker_status,
                        face_count_status=face_count_status,
                        verification_state=str(decision_cache.state),
                        escalation_level=escalation_level,
                        anomaly_streak=int(anomaly_streak),
                    )

                    time.sleep(0.01)
            finally:
                mic.stop()
                cap.release()
                if face_mesh is not None:
                    face_mesh.close()
                risk.export_json()
        except Exception as exc:  # noqa: BLE001
            self._update_state(error=str(exc))
        finally:
            self._update_state(running=False)
