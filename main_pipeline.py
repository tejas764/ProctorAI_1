from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import time
from typing import Any, Optional

import cv2
import mediapipe as mp
import numpy as np
import yaml

from audio_sync_verification import AudioSyncVerifier
from av_correlation import AVCorrelationEngine
from hand_occlusion import HandMouthOcclusionDetector, mouth_bbox_from_facemesh
from lip_sync_verification import LipSyncVerifier
from risk_engine import RiskEngine
from speaker_verification import SpeakerVerifier

try:
    import sounddevice as sd
except ImportError:
    sd = None


def compute_mar(face_landmarks: Any) -> float:
    pts = face_landmarks.landmark
    upper_inner = pts[13]
    lower_inner = pts[14]
    left_corner = pts[78]
    right_corner = pts[308]
    vertical = abs(lower_inner.y - upper_inner.y)
    horizontal = max(abs(right_corner.x - left_corner.x), 1e-6)
    return float(vertical / horizontal)


def draw_overlay(frame: np.ndarray, lines: list[str]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = 16, 28
    line_h = 22
    font_scale = 0.62
    thickness = 2
    # Transparent background panel for readability.
    overlay = frame.copy()
    w = min(frame.shape[1] - 16, 980)
    h = min(frame.shape[0] - 16, 12 + line_h * len(lines))
    cv2.rectangle(overlay, (8, 8), (8 + w, 8 + h), (255, 255, 255), -1)
    cv2.addWeighted(overlay, 0.28, frame, 0.72, 0, frame)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x, y + i * line_h), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    an = a / (np.linalg.norm(a) + 1e-8)
    bn = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(an, bn))


def simple_face_embedding(frame: np.ndarray, face_landmarks: Any) -> np.ndarray:
    h, w = frame.shape[:2]
    xs = [int(p.x * w) for p in face_landmarks.landmark]
    ys = [int(p.y * h) for p in face_landmarks.landmark]
    x0, x1 = max(0, min(xs)), min(w - 1, max(xs))
    y0, y1 = max(0, min(ys)), min(h - 1, max(ys))
    if x1 <= x0 or y1 <= y0:
        return np.zeros(64, dtype=np.float32)
    roi_bgr = frame[y0:y1, x0:x1]
    if roi_bgr.size == 0:
         return np.zeros(64, dtype=np.float32)

    roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    from web_modules.frame_utils import safe_resize
    roi = safe_resize(roi_gray, (64, 64))
    if roi is None:
        return np.zeros(64, dtype=np.float32)

    hist = cv2.calcHist([roi], [0], None, [64], [0, 256]).flatten().astype(np.float32)
    return hist / (np.linalg.norm(hist) + 1e-8)


class AudioMonitor:
    def __init__(self, sample_rate: int, block_size: int, vad_threshold: float) -> None:
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.vad_threshold = vad_threshold
        self._stream = None
        self._latest = np.zeros((block_size,), dtype=np.float32)
        self._buffer = deque(maxlen=sample_rate * 4)  # 4s rolling

    def _callback(self, indata, frames, callback_time, status) -> None:
        del frames, callback_time, status
        mono = indata[:, 0].astype(np.float32)
        self._latest = mono
        self._buffer.extend(mono.tolist())

    def start(self) -> None:
        if sd is None:
            raise ImportError("sounddevice is required for real-time audio.")
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

    def audio_chunk(self) -> np.ndarray:
        return self._latest.copy()

    def latest_seconds(self, seconds: float) -> np.ndarray:
        n = int(max(1, seconds * self.sample_rate))
        arr = np.array(self._buffer, dtype=np.float32)
        return arr[-n:] if arr.size >= n else arr

    def rms(self) -> float:
        chunk = self.audio_chunk()
        return float(np.sqrt(np.mean(chunk * chunk))) if chunk.size else 0.0

    def vad(self) -> bool:
        return self.rms() >= self.vad_threshold


@dataclass
class PipelineConfig:
    camera_index: int
    sample_rate: int
    block_size: int
    vad_threshold: float
    hand_iou_threshold: float
    hand_consecutive_frames: int
    hand_pad_ratio: float
    speaker_threshold: float
    face_similarity_threshold: float
    lipsync_verify_threshold: float
    suspicious_streak_for_verify: int
    audio_sync_low_score_threshold: float
    user_id: str
    voice_db_path: str
    speaker_window_seconds: float
    speaker_drift_threshold: float
    terminate_on_cheating_alert: bool


def load_config(path: str = "config.yaml") -> PipelineConfig:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return PipelineConfig(
        camera_index=int(cfg.get("camera_index", 0)),
        sample_rate=int(cfg.get("mic_sample_rate", 16000)),
        block_size=int(cfg.get("mic_block_size", 1024)),
        vad_threshold=float(cfg.get("mic_threshold", 0.015)),
        hand_iou_threshold=float(cfg.get("hand_mouth_iou_threshold", 0.03)),
        hand_consecutive_frames=int(cfg.get("hand_occlusion_consecutive_frames", 3)),
        hand_pad_ratio=float(cfg.get("hand_box_padding_ratio", 0.20)),
        speaker_threshold=float(cfg.get("speaker_similarity_threshold", 0.72)),
        face_similarity_threshold=float(cfg.get("face_similarity_threshold", 0.65)),
        lipsync_verify_threshold=float(cfg.get("lipsync_verify_threshold", 0.45)),
        suspicious_streak_for_verify=int(cfg.get("suspicious_streak_for_verify", 8)),
        audio_sync_low_score_threshold=float(cfg.get("audio_sync_low_score_threshold", 0.40)),
        user_id=str(cfg.get("user_id", "default_user")),
        voice_db_path=str(cfg.get("voice_db_path", "proctorguard.db")),
        speaker_window_seconds=float(cfg.get("speaker_window_seconds", 2.5)),
        speaker_drift_threshold=float(cfg.get("speaker_drift_threshold", 0.08)),
        terminate_on_cheating_alert=bool(cfg.get("terminate_on_cheating_alert", True)),
    )


class ExamProctorPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.cfg = config
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self.audio = AudioMonitor(config.sample_rate, config.block_size, config.vad_threshold)
        self.occlusion = HandMouthOcclusionDetector(
            iou_threshold=config.hand_iou_threshold,
            consecutive_frames=config.hand_consecutive_frames,
            pad_ratio=config.hand_pad_ratio,
        )
        self.av = AVCorrelationEngine()
        self.speaker = SpeakerVerifier(
            sample_rate=config.sample_rate,
            similarity_threshold=config.speaker_threshold,
            drift_threshold=config.speaker_drift_threshold,
            window_seconds=config.speaker_window_seconds,
            user_id=config.user_id,
            db_path=config.voice_db_path,
        )
        self.verifier = LipSyncVerifier(threshold=config.lipsync_verify_threshold)
        self.audio_sync = AudioSyncVerifier(sample_rate=config.sample_rate)
        self.risk = RiskEngine(log_dir="proctor_logs")
        self.face_ref: Optional[np.ndarray] = None
        self.suspicious_streak = 0
        self.last_event_time: dict[str, float] = {}
        self.motion_series: deque[float] = deque(maxlen=90)  # ~3 sec @ 30fps
        self.audio_series: deque[float] = deque(maxlen=90)
        self.consecutive_voice_violations = 0

    def _log_once(self, reason: str, timestamp_s: float, frame: np.ndarray, audio: np.ndarray, details: dict) -> None:
        cooldown_s = 2.0
        if timestamp_s - self.last_event_time.get(reason, -999.0) < cooldown_s:
            return
        self.last_event_time[reason] = timestamp_s
        self.risk.add_event(reason, timestamp_s, frame, audio, self.cfg.sample_rate, details=details)

    def run(self) -> None:
        cap = cv2.VideoCapture(self.cfg.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.cfg.camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {self.cfg.camera_index}.")

        self.audio.start()
        start_time = time.time()

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                t = time.time() - start_time
                audio_chunk = self.audio.audio_chunk()
                audio_present = self.audio.vad()
                audio_energy = self.audio.rms()
                speaker_window_audio = self.audio.latest_seconds(self.cfg.speaker_window_seconds)
                spk = self.speaker.verify(speaker_window_audio, audio_present=audio_present, timestamp_s=t)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_result = self.face_mesh.process(rgb)
                hands_result = self.hands.process(rgb)
                faces = face_result.multi_face_landmarks if face_result.multi_face_landmarks else []
                num_faces = len(faces)

                mar = 0.0
                mouth_visible = False
                lip_sync_status = "NO_FACE"
                speaker_sim: Optional[float] = spk.similarity
                speaker_drift: Optional[float] = spk.drift
                face_consistent = True
                occl_iou = 0.0
                av_status = "IDLE"
                corr_score = None
                audio_sync_score = 1.0
                audio_sync_flags: list[str] = []
                av_offset_ms: Optional[float] = None
                speaking_face_visible = False
                security_reason = ""
                escalation_state = "NONE"

                if num_faces >= 1:
                    primary = faces[0]
                    mar = compute_mar(primary)
                    mouth_box = mouth_bbox_from_facemesh(primary, frame.shape)
                    occ = self.occlusion.update(
                        mouth_box=mouth_box,
                        hand_landmarks=hands_result.multi_hand_landmarks if hands_result else None,
                        frame_shape=frame.shape,
                    )
                    mouth_visible = occ.mouth_visible
                    occl_iou = occ.overlap_iou
                    speaking_face_visible = mouth_visible and mar >= 0.02

                    av_res = self.av.update(audio_present, audio_energy, mar)
                    av_status = av_res.status
                    corr_score = av_res.corr_score
                    self.motion_series.append(av_res.mar_delta)
                    self.audio_series.append(audio_energy)

                    # Advanced audio-sync verification.
                    sync_res = self.audio_sync.update(
                        timestamp_s=t,
                        audio_chunk=audio_chunk,
                        audio_present=audio_present,
                        mar_value=mar,
                        mouth_occluded=not mouth_visible,
                    )
                    audio_sync_score = sync_res.score
                    audio_sync_flags = sync_res.flags
                    av_offset_ms = sync_res.offset_ms

                    for flag in audio_sync_flags:
                        self._log_once(
                            flag,
                            t,
                            frame,
                            self.audio.latest_seconds(2.5),
                            {
                                "audio_sync_score": round(audio_sync_score, 4),
                                "energy_mar_corr": round(sync_res.energy_mar_corr, 4),
                                "offset_ms": round(av_offset_ms, 2) if av_offset_ms is not None else None,
                                "viseme_mismatch_count": sync_res.viseme_mismatch_count,
                            },
                        )
                    if audio_sync_score < self.cfg.audio_sync_low_score_threshold:
                        self._log_once(
                            "AUDIO_SYNC_LOW",
                            t,
                            frame,
                            self.audio.latest_seconds(2.5),
                            {
                                "audio_sync_score": round(audio_sync_score, 4),
                                "flags": audio_sync_flags,
                            },
                        )

                    if not mouth_visible:
                        lip_sync_status = "MOUTH_OCCLUDED"
                        self._log_once(
                            "HAND_OCCLUSION",
                            t,
                            frame,
                            self.audio.latest_seconds(2.5),
                            {"iou": round(occl_iou, 4), "streak": occ.consecutive_frames},
                        )
                    elif av_status in ("AUDIO_ONLY", "WEAK_SYNC"):
                        lip_sync_status = "SUSPICIOUS"
                        self._log_once(
                            "AUDIO_ONLY",
                            t,
                            frame,
                            self.audio.latest_seconds(2.5),
                            {"av_status": av_status, "corr_score": corr_score},
                        )
                    elif av_status == "SYNC_OK":
                        lip_sync_status = "SYNC_OK"
                    elif av_status == "SILENT_SPEECH":
                        lip_sync_status = "SILENT_SPEECH"
                    else:
                        lip_sync_status = "IDLE"

                    # Face identity consistency check.
                    curr_face_emb = simple_face_embedding(frame, primary)
                    if self.face_ref is None and np.linalg.norm(curr_face_emb) > 1e-6:
                        self.face_ref = curr_face_emb
                    elif self.face_ref is not None:
                        sim = cosine_similarity(self.face_ref, curr_face_emb)
                        face_consistent = sim >= self.cfg.face_similarity_threshold
                        if not face_consistent:
                            self._log_once(
                                "MULTIPLE_FACES",
                                t,
                                frame,
                                self.audio.latest_seconds(2.5),
                                {"face_similarity": round(sim, 4)},
                            )

                else:
                    self.motion_series.append(0.0)
                    self.audio_series.append(audio_energy)
                    if audio_present:
                        self._log_once(
                            "MULTIPLE_FACES",
                            t,
                            frame,
                            self.audio.latest_seconds(2.5),
                            {"reason": "audio_with_no_face"},
                        )

                # Multiple faces check.
                if num_faces > 1:
                    self._log_once(
                        "MULTIPLE_FACES",
                        t,
                        frame,
                        self.audio.latest_seconds(2.5),
                        {"num_faces": num_faces},
                    )

                if audio_present and not spk.has_reference:
                    self._log_once(
                        "ENROLLMENT_MISSING",
                        t,
                        frame,
                        self.audio.latest_seconds(2.5),
                        {"reason": spk.reason, "user_id": self.cfg.user_id},
                    )
                if spk.is_mismatch:
                    self._log_once(
                        "VOICE_MISMATCH",
                        t,
                        frame,
                        self.audio.latest_seconds(2.5),
                        {
                            "speaker_similarity": round(float(spk.similarity), 4) if spk.similarity is not None else None,
                            "threshold": round(float(spk.threshold), 4) if spk.threshold is not None else None,
                            "reason": spk.reason,
                        },
                    )
                elif spk.is_drift:
                    self._log_once(
                        "VOICE_DRIFT",
                        t,
                        frame,
                        self.audio.latest_seconds(2.5),
                        {
                            "speaker_similarity": round(float(spk.similarity), 4) if spk.similarity is not None else None,
                            "drift": round(float(spk.drift), 4) if spk.drift is not None else None,
                            "drift_threshold": round(float(spk.drift_threshold), 4) if spk.drift_threshold is not None else None,
                        },
                    )

                audio_without_visible_face = audio_present and (num_faces != 1 or not speaking_face_visible)
                segment_violation = bool(spk.is_mismatch or spk.is_drift or audio_without_visible_face)
                if segment_violation:
                    reasons = []
                    if spk.is_mismatch:
                        reasons.append("similarity_below_threshold")
                    if spk.is_drift:
                        reasons.append("drift_above_threshold")
                    if audio_without_visible_face:
                        reasons.append("audio_without_visible_speaking_face")
                    security_reason = ",".join(reasons)
                    self.consecutive_voice_violations += 1
                    if self.consecutive_voice_violations == 1:
                        escalation_state = "WARNING"
                        self._log_once(
                            "VOICE_POLICY_WARNING",
                            t,
                            frame,
                            self.audio.latest_seconds(2.5),
                            {"reason": security_reason, "consecutive": self.consecutive_voice_violations},
                        )
                    elif self.consecutive_voice_violations == 2:
                        escalation_state = "SOFT_FLAG"
                        self._log_once(
                            "VOICE_POLICY_SOFT_FLAG",
                            t,
                            frame,
                            self.audio.latest_seconds(2.5),
                            {"reason": security_reason, "consecutive": self.consecutive_voice_violations},
                        )
                    else:
                        escalation_state = "CHEATING_ALERT"
                        self._log_once(
                            "CHEATING_ALERT",
                            t,
                            frame,
                            self.audio.latest_seconds(2.5),
                            {"reason": security_reason, "consecutive": self.consecutive_voice_violations},
                        )
                        if self.cfg.terminate_on_cheating_alert:
                            lines = [
                                "Session terminated by security policy.",
                                f"Reason: {security_reason}",
                                f"Consecutive violations: {self.consecutive_voice_violations}",
                            ]
                            draw_overlay(frame, lines)
                            cv2.imshow("Exam Proctoring Lip-Sync Pipeline", frame)
                            cv2.waitKey(900)
                            break
                else:
                    self.consecutive_voice_violations = 0

                # Trigger heavy verification layer only when suspicious persists.
                suspicious = lip_sync_status in ("SUSPICIOUS", "MOUTH_OCCLUDED")
                suspicious = suspicious or (audio_sync_score < self.cfg.audio_sync_low_score_threshold) or (len(audio_sync_flags) >= 2)
                self.suspicious_streak = self.suspicious_streak + 1 if suspicious else 0
                verify_text = "SKIPPED"
                if self.suspicious_streak >= self.cfg.suspicious_streak_for_verify and len(self.motion_series) >= 30:
                    seg_motion = np.array(self.motion_series, dtype=np.float32)
                    seg_audio = np.array(self.audio_series, dtype=np.float32)
                    vr = self.verifier.verify_segment(seg_motion, seg_audio)
                    verify_text = f"{vr.score:.3f}"
                    if not vr.passed:
                        self._log_once(
                            "LIPSYNC_VERIFICATION_FAIL",
                            t,
                            frame,
                            self.audio.latest_seconds(2.8),
                            {"verification_model": vr.model_name, "sync_score": round(vr.score, 4)},
                        )
                    self.suspicious_streak = 0

                lines = [
                    f"Faces: {num_faces}",
                    f"Audio Present: {'YES' if audio_present else 'NO'}  RMS={audio_energy:.5f}",
                    f"MAR: {mar:.4f}",
                    f"LipSync Heuristic: {lip_sync_status}",
                    f"AV Status: {av_status}  Corr={corr_score if corr_score is not None else float('nan'):.3f}",
                    f"AudioSync Score: {audio_sync_score:.3f}  Offset(ms): {av_offset_ms if av_offset_ms is not None else float('nan'):.1f}",
                    f"AudioSync Flags: {', '.join(audio_sync_flags) if audio_sync_flags else 'NONE'}",
                    f"Mouth IoU(Hand): {occl_iou:.3f}  Mouth Visible: {'YES' if mouth_visible else 'NO'}",
                    f"Speaker Similarity: {speaker_sim if speaker_sim is not None else float('nan'):.3f} (thr={spk.threshold if spk.threshold is not None else float('nan'):.3f})",
                    f"Voice Drift: {speaker_drift if speaker_drift is not None else float('nan'):.3f} (thr={spk.drift_threshold if spk.drift_threshold is not None else float('nan'):.3f})",
                    f"Voice Base Match: {spk.status_color.upper()} ({spk.reason})",
                    f"Face Consistent: {'YES' if face_consistent else 'NO'}",
                    f"Visible Speaking Face: {'YES' if speaking_face_visible else 'NO'}",
                    f"Segment Security: {'ALERT' if segment_violation else 'OK'} {security_reason if security_reason else ''}",
                    f"Escalation: {escalation_state}  Consecutive={self.consecutive_voice_violations}",
                    f"Verification Score (on trigger): {verify_text}",
                    f"Risk Score: {self.risk.risk_score}  Level: {self.risk.level()}",
                    "Press Q to quit",
                ]
                draw_overlay(frame, lines)
                cv2.imshow("Exam Proctoring Lip-Sync Pipeline", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            self.audio.stop()
            cap.release()
            self.face_mesh.close()
            self.hands.close()
            cv2.destroyAllWindows()
            path = self.risk.export_json()
            print(f"Evidence log saved: {path}")


def main() -> None:
    cfg = load_config("config.yaml")
    pipeline = ExamProctorPipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
