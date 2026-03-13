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
from web_modules.gaze_bridge import GazeEngine, GazeReading, _reading_kwargs
from web_modules.phone_detection import PhoneDetector  # Import PhoneDetector
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


def _human_flag_detail(reason: str, details: dict[str, object] | None = None) -> str:
    info = details or {}
    if reason == "GAZE_OUTSIDE":
        conf = info.get("gaze_confidence")
        return f"Gaze moved outside calibrated screen region (confidence={conf})."
    if reason == "VOICE_POLICY_WARNING":
        why = str(info.get("reason", "voice anomaly"))
        streak = info.get("streak")
        return f"Voice policy warning: {why}. Anomaly streak={streak}."
    if reason == "CHEATING_ALERT":
        why = str(info.get("reason", "high-risk behavior"))
        streak = info.get("streak")
        return f"Escalated cheating alert: {why}. Anomaly streak={streak}."
    if reason == "MULTIPLE_FACES":
        why = str(info.get("reason", "multiple speakers/faces detected"))
        conf = info.get("confidence")
        return f"Multiple faces detected: {why} (confidence={conf})."
    if reason == "PHONE_DETECTED":
        return "Cell phone detected in frame."
    return f"Flag raised: {reason}. {str(details or '')}"


def compute_mar(face_landmarks: object) -> float:
    pts = getattr(face_landmarks, "landmark")
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
            "gaze_enabled": False,
            "gaze_status": "DISABLED",
            "gaze_confidence": 0.0,
            "gaze_calibrated": False,
            "gaze_progress": 0.0,
            "gaze_step": "",
            "gaze_step_index": 0,
            "gaze_total_steps": 0,
            "gaze_step_progress": 0.0,
            "gaze_prompt": "",
            "gaze_awaiting_start": False,
            "gaze_countdown_active": False,
            "gaze_countdown_remaining": 0.0,
            "gaze_capturing": False,
            "speaker_similarity_bar": 0.0,
            "voice_stability": "Stable",
            "active_speaker_status": "UNKNOWN",
            "face_count_status": "UNKNOWN",
            "verification_state": "YELLOW",
            "escalation_level": "NORMAL",
            "anomaly_streak": 0,
            "last_flag_reason": "",
            "last_flag_details": "",
            "updated_at": _utc_now_iso(),
            "error": "",
        }
        self._gaze_engine: GazeEngine | None = None

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
            self._state["last_flag_details"] = _human_flag_detail(reason, details)
            self._state["risk_score"] = int(risk.risk_score)
            self._state["risk_level"] = risk.level()
            self._state["updated_at"] = _utc_now_iso()

    def _apply_gaze_reading(self, reading: GazeReading) -> None:
        self._update_state(
            gaze_enabled=bool(self._gaze_engine and self._gaze_engine.ready),
            gaze_status=reading.status,
            gaze_confidence=reading.confidence,
            gaze_calibrated=reading.calibrated,
            gaze_progress=reading.progress,
            gaze_step=reading.step,
            gaze_step_index=reading.step_index,
            gaze_total_steps=reading.total_steps,
            gaze_step_progress=reading.step_progress,
            gaze_prompt=reading.prompt,
            gaze_awaiting_start=reading.awaiting_start,
            gaze_error=reading.error,
        )

    def get_gaze_state(self) -> dict[str, object]:
        state = self.get_state()
        return {
            "enabled": state.get("gaze_enabled", False),
            "status": state.get("gaze_status", "DISABLED"),
            "confidence": state.get("gaze_confidence", 0.0),
            "calibrated": state.get("gaze_calibrated", False),
            "progress": state.get("gaze_progress", 0.0),
            "step": state.get("gaze_step", ""),
            "step_index": state.get("gaze_step_index", 0),
            "total_steps": state.get("gaze_total_steps", 0),
            "step_progress": state.get("gaze_step_progress", 0.0),
            "prompt": state.get("gaze_prompt", ""),
            "awaiting_start": state.get("gaze_awaiting_start", False),
            "countdown_active": state.get("gaze_countdown_active", False),
            "countdown_remaining": state.get("gaze_countdown_remaining", 0.0),
            "capturing": state.get("gaze_capturing", False),
            "error": state.get("error", ""),
        }

    def begin_gaze_calibration_step(self) -> tuple[bool, str, dict[str, object]]:
        if self._gaze_engine is None:
            state = self.get_gaze_state()
            state["error"] = self.get_state().get("error", "Gaze engine unavailable")
            return False, "Gaze engine unavailable.", state
        ok, message, state = self._gaze_engine.begin_calibration_step()
        self._update_state(
            gaze_step=state.get("step", ""),
            gaze_step_index=state.get("step_index", 0),
            gaze_total_steps=state.get("total_steps", 0),
            gaze_step_progress=state.get("step_progress", 0.0),
            gaze_progress=state.get("progress", 0.0),
            gaze_prompt=state.get("prompt", ""),
            gaze_awaiting_start=state.get("awaiting_start", False),
            gaze_countdown_active=state.get("countdown_active", False),
            gaze_countdown_remaining=state.get("countdown_remaining", 0.0),
            gaze_capturing=state.get("capturing", False),
        )
        return ok, message, self.get_gaze_state()

    def reset_gaze_calibration(self) -> tuple[bool, str, dict[str, object]]:
        if self._gaze_engine is None:
            state = self.get_gaze_state()
            state["error"] = self.get_state().get("error", "Gaze engine unavailable")
            return False, "Gaze engine unavailable.", state
        ok, message, state = self._gaze_engine.reset_calibration(delete_saved=True)
        self._update_state(
            gaze_status="CALIBRATING",
            gaze_calibrated=False,
            gaze_confidence=0.0,
            gaze_step=state.get("step", ""),
            gaze_step_index=state.get("step_index", 0),
            gaze_total_steps=state.get("total_steps", 0),
            gaze_step_progress=state.get("step_progress", 0.0),
            gaze_progress=state.get("progress", 0.0),
            gaze_prompt=state.get("prompt", ""),
            gaze_awaiting_start=state.get("awaiting_start", False),
            gaze_countdown_active=state.get("countdown_active", False),
            gaze_countdown_remaining=state.get("countdown_remaining", 0.0),
            gaze_capturing=state.get("capturing", False),
        )
        return ok, message, self.get_gaze_state()

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
            last_flag_details="",
            risk_score=0,
            risk_level="NORMAL",
            gaze_status="DISABLED",
            gaze_confidence=0.0,
            gaze_calibrated=False,
            gaze_progress=0.0,
            gaze_step="",
            gaze_step_index=0,
            gaze_total_steps=0,
            gaze_step_progress=0.0,
            gaze_prompt="",
            gaze_awaiting_start=False,
            gaze_countdown_active=False,
            gaze_countdown_remaining=0.0,
            gaze_capturing=False,
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
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
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

            # Initialize phone detector
            phone_detector = PhoneDetector(model_dir="web_modules")
            last_phone_check = 0

            gaze_engine = GazeEngine(store=self.store, user_id=self._user_id)
            self._gaze_engine = gaze_engine
            gaze_ok, gaze_message = gaze_engine.start()
            if gaze_ok:
                # Load per-user calibration (or reset for fresh calibration)
                gaze_engine.set_user(self._user_id)
            gaze_cache = GazeReading(
                status="DISABLED" if not gaze_ok else ("INSIDE" if gaze_engine.calibrated else "CALIBRATING"),
                confidence=0.0,
                calibrated=gaze_engine.calibrated,
                error="" if gaze_ok else gaze_message,
                **_reading_kwargs(gaze_engine.calibration_state()),
            )
            self._apply_gaze_reading(gaze_cache)
            self._update_state(error="" if gaze_ok else gaze_message)
            outside_gaze_streak = 0
            frame_index = 0
            motion_series: deque[float] = deque(maxlen=120)
            audio_series: deque[float] = deque(maxlen=120)
            suspicious_streak = 0
            stable_faces = 0
            stable_face_streak = 0
            last_face_count = -1
            last_faces: list[object] = []
            last_num_faces = 0
            last_verify_t = 0.0
            verify_interval_s = 1.0
            min_voice_policy_rms = 0.012
            min_reliable_pitch_hz = 70.0
            max_reliable_pitch_hz = 420.0
            face_mesh_stride = 1
            gaze_stride = 2
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
                    frame_index += 1
                    audio_present = mic.vad()
                    audio_rms = mic.rms()
                    faces = []
                    num_faces = 0
                    if face_mesh is not None:
                        unstable_face_window = stable_faces != 1 or stable_face_streak < 6
                        if unstable_face_window:
                            face_mesh_stride = 1
                        elif audio_present:
                            face_mesh_stride = 3
                        else:
                            face_mesh_stride = 8

                        run_face_mesh = (frame_index % face_mesh_stride == 0) or (frame_index <= 2)
                        if run_face_mesh:
                            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            res = face_mesh.process(rgb)
                            last_faces = res.multi_face_landmarks if res.multi_face_landmarks else []
                            last_num_faces = len(last_faces)
                        faces = last_faces
                        num_faces = last_num_faces
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

                    if not gaze_cache.calibrated or gaze_cache.status in {"CALIBRATING", "OUTSIDE", "ERROR"}:
                        gaze_stride = 1
                    elif stable_faces == 1 and stable_face_streak >= 10 and gaze_cache.status == "INSIDE":
                        gaze_stride = 6
                    else:
                        gaze_stride = 2

                    if gaze_ok and (frame_index % gaze_stride == 0):
                        gaze_cache = gaze_engine.process(frame)
                        self._apply_gaze_reading(gaze_cache)
                        gaze_state = gaze_engine.calibration_state()
                        self._update_state(
                            gaze_countdown_active=bool(gaze_state.get("countdown_active", False)),
                            gaze_countdown_remaining=float(gaze_state.get("countdown_remaining", 0.0)),
                            gaze_capturing=bool(gaze_state.get("capturing", False)),
                        )
                    elif not gaze_ok:
                        gaze_cache = GazeReading(
                            status="DISABLED",
                            confidence=0.0,
                            calibrated=False,
                            progress=0.0,
                            error=gaze_message,
                        )

                    if gaze_cache.status == "OUTSIDE":
                        outside_gaze_streak += 1
                    else:
                        outside_gaze_streak = 0
                    if outside_gaze_streak >= 2 and audio_present:
                        self._flag_event(
                            "GAZE_OUTSIDE",
                            risk,
                            frame,
                            mic.latest_seconds(1.0),
                            {
                                "gaze_confidence": float(gaze_cache.confidence),
                                "gaze_status": gaze_cache.status,
                            },
                        )
                        outside_gaze_streak = 0

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
                        speech_window_active = bool(
                            audio_present
                            and window_audio.size >= int(self.sample_rate * 0.4)
                            and audio_rms >= min_voice_policy_rms
                        )
                        spk_cache = speaker.verify(audio=window_audio, audio_present=audio_present, timestamp_s=now_t)
                        spk = spk_cache

                        if speech_window_active:
                            multi_cache = estimate_speaker_count(window_audio, self.sample_rate)
                        else:
                            multi_cache = MultiSpeakerEstimate(speaker_count=1, confidence=0.0, reason="no_speech")

                        if speech_window_active:
                            feats = extract_voice_features(window_audio, self.sample_rate)
                            pitch_hz = float(feats.pitch_mean)
                            if min_reliable_pitch_hz <= pitch_hz <= max_reliable_pitch_hz:
                                pitch_match = soft_pitch_match(pitch_hz, pitch_min_hz, pitch_max_hz)
                                frequency_match = pitch_match
                            else:
                                # Unreliable/unvoiced pitch windows should stay neutral.
                                pitch_match = None
                                frequency_match = None
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

                        if speech_window_active:
                            decision_cache = fuse_window_decision(
                                similarity_score=similarity_score,
                                drift_score=drift_score,
                                lip_sync_score=lip_sync_score,
                                active_speaker_prob=active_prob,
                                single_face=(stable_faces == 1),
                                speaker_count=multi_cache.speaker_count,
                                hard_mismatch=bool(spk.is_mismatch),
                            )
                        else:
                            decision_cache = WindowDecision(
                                similarity_score=similarity_score,
                                drift_score=drift_score,
                                lip_sync_score=lip_sync_score,
                                active_speaker_prob=active_prob,
                                fused_score=0.5,
                                state="YELLOW",
                                anomaly=False,
                                reason="no_speech_window",
                            )

                        if speech_window_active and multi_cache.speaker_count > 1:
                            self._flag_event(
                                "MULTIPLE_FACES",
                                risk,
                                frame,
                                window_audio,
                                {"reason": "multiple_speakers_detected", "confidence": multi_cache.confidence},
                            )

                        if speech_window_active:
                            anomaly_streak = anomaly_streak + 1 if decision_cache.anomaly else max(0, anomaly_streak - 1)
                        else:
                            anomaly_streak = max(0, anomaly_streak - 1)
                        escalation_level = "ALERT" if anomaly_streak >= 3 else ("WARNING" if anomaly_streak >= 2 else "NORMAL")

                        # Temporal escalation policy.
                        if speech_window_active and anomaly_streak == 2:
                            self._flag_event(
                                "VOICE_POLICY_WARNING",
                                risk,
                                frame,
                                window_audio,
                                {"reason": decision_cache.reason, "streak": anomaly_streak},
                            )
                        elif speech_window_active and anomaly_streak >= 3:
                            self._flag_event(
                                "CHEATING_ALERT",
                                risk,
                                frame,
                                window_audio,
                                {"reason": decision_cache.reason, "streak": anomaly_streak},
                            )

                        suspicious = bool(
                            speech_window_active
                            and (
                                decision_cache.anomaly
                                or (sync_res.score < self.audio_sync_low_score_threshold)
                                or (len(sync_res.flags) >= 2)
                            )
                        )
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

                    # Run phone detection every ~1.5 seconds (45 frames)
                    if (frame_index - last_phone_check) >= 45:
                        detect_start = time.time()
                        has_phone = phone_detector.detect(frame)
                        last_phone_check = frame_index
                        if has_phone:
                            self._flag_event(
                                "PHONE_DETECTED",
                                risk,
                                frame,
                                mic.latest_seconds(1.0),
                                {"confidence": "high"},
                                cooldown_s=5.0
                            )

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
                        gaze_enabled=bool(gaze_ok),
                        gaze_status=str(gaze_cache.status),
                        gaze_confidence=float(gaze_cache.confidence),
                        gaze_calibrated=bool(gaze_cache.calibrated),
                        gaze_progress=float(gaze_cache.progress),
                        speaker_similarity_bar=float(decision_cache.similarity_score),
                        voice_stability=str(stability_label),
                        active_speaker_status=active_speaker_status,
                        face_count_status=face_count_status,
                        verification_state=str(decision_cache.state),
                        escalation_level=escalation_level,
                        anomaly_streak=int(anomaly_streak),
                        error=gaze_cache.error if gaze_cache.error else "",
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
            self._gaze_engine = None
            self._update_state(running=False)
