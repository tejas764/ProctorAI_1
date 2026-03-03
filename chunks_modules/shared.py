from __future__ import annotations

from collections import deque
from threading import Lock
from typing import Any, Optional

import cv2
import mediapipe as mp
import numpy as np

from chunks_modules.config import (
    GATE_FLOW_EPSILON,
    GATE_FLOW_MIN_COUNT,
    GATE_MAR_MAX,
    GATE_MAR_MIN,
    GATE_STABILITY_FRAMES,
    HAND_BOX_PADDING_RATIO,
    MAHAL_THRESHOLD,
    MOUTH_HIDDEN_TEXTURE_THRESHOLD,
    MOUTH_OCCLUSION_COVERAGE_THRESHOLD,
    SILENCE_DELAY,
)

try:
    import sounddevice as sd
except ImportError:
    sd = None


def calculate_mahalanobis(feature_window: deque[np.ndarray], features: np.ndarray) -> float:
    if len(feature_window) < 2:
        return 0.0
    arr = np.array(feature_window)
    mean = arr.mean(axis=0)
    diff = features - mean
    cov = np.cov(arr.T) + np.eye(arr.shape[1]) * 1e-6
    inv_cov = np.linalg.inv(cov)
    return float(np.sqrt(diff.dot(inv_cov).dot(diff)))


def extract_lip_features(
    landmarks: Any,
    shape: tuple[int, int, int],
    height_window: deque[float],
    movement_window: deque[float],
    feature_window: deque[np.ndarray],
) -> np.ndarray:
    ih, iw = shape[:2]
    upper_ids = [40, 0, 270]
    lower_ids = [181, 17, 405]
    upper_pts = np.array([(int(landmarks.landmark[i].x * iw), int(landmarks.landmark[i].y * ih)) for i in upper_ids])
    lower_pts = np.array([(int(landmarks.landmark[i].x * iw), int(landmarks.landmark[i].y * ih)) for i in lower_ids])
    dists = [abs(upper_pts[i][1] - lower_pts[i][1]) for i in range(min(len(upper_pts), len(lower_pts)))]
    inner_h = np.mean(dists) / ih
    lip_w = (upper_pts[:, 0].max() - upper_pts[:, 0].min()) / iw
    lip_area = inner_h * lip_w

    height_window.append(inner_h)
    if len(height_window) >= 2:
        movement_window.append(abs(height_window[-1] - height_window[-2]))

    features = np.array([inner_h, movement_window[-1] if movement_window else 0.0, lip_area])
    feature_window.append(features)
    return features


def detect_speaking(
    features: np.ndarray,
    time_s: float,
    feature_window: deque[np.ndarray],
    last_time: Optional[float],
) -> tuple[bool, Optional[float]]:
    dist = calculate_mahalanobis(feature_window, features)
    if dist > MAHAL_THRESHOLD:
        return True, time_s
    if last_time is None:
        return False, last_time
    return (time_s - last_time) < SILENCE_DELAY, last_time


class MicrophoneVoiceMonitor:
    def __init__(self, sample_rate: int, block_size: int, threshold: float, device_index: Optional[int] = None) -> None:
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.threshold = threshold
        self.device_index = device_index
        self._rms: float = 0.0
        self._lock = Lock()
        self._stream = None

    def _audio_callback(self, indata, frames, callback_time, status) -> None:
        del frames, callback_time, status
        rms = float(np.sqrt(np.mean(np.square(indata)))) if indata.size else 0.0
        with self._lock:
            self._rms = rms

    def start(self) -> None:
        if sd is None:
            raise ImportError("sounddevice is required for live microphone voice status.")
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            channels=1,
            device=self.device_index,
            callback=self._audio_callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def current_rms(self) -> float:
        with self._lock:
            return self._rms

    def is_speaking(self) -> bool:
        return self.current_rms() >= self.threshold


class OverlayUIState:
    def __init__(self, persistence_s: float = 0.7) -> None:
        self.debug_mode = False
        self.visible = True
        self.persistence_s = persistence_s
        self._last_normal_update = 0.0
        self._normal_cache = {"audio": "SILENT", "lips": "STILL", "sync": "UNCERTAIN", "risk": "LOW"}

    def handle_key(self, key: int) -> bool:
        if key in (ord("d"), ord("D")):
            self.debug_mode = not self.debug_mode
            return False
        if key in (ord("h"), ord("H")):
            self.visible = not self.visible
            return False
        return key in (ord("q"), ord("Q"))

    def smoothed_normal(self, now_s: float, current: dict[str, str]) -> dict[str, str]:
        if (now_s - self._last_normal_update) >= self.persistence_s:
            self._normal_cache = current
            self._last_normal_update = now_s
        return self._normal_cache


def draw_rounded_rect(img: np.ndarray, x: int, y: int, w: int, h: int, radius: int, color: tuple[int, int, int]) -> None:
    radius = max(1, min(radius, w // 2, h // 2))
    cv2.rectangle(img, (x + radius, y), (x + w - radius, y + h), color, -1)
    cv2.rectangle(img, (x, y + radius), (x + w, y + h - radius), color, -1)
    cv2.circle(img, (x + radius, y + radius), radius, color, -1)
    cv2.circle(img, (x + w - radius, y + radius), radius, color, -1)
    cv2.circle(img, (x + radius, y + h - radius), radius, color, -1)
    cv2.circle(img, (x + w - radius, y + h - radius), radius, color, -1)


def draw_transparent_panel(
    frame: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    alpha: float = 0.42,
    radius: int = 12,
) -> None:
    overlay = frame.copy()
    draw_rounded_rect(overlay, x, y, w, h, radius, (0, 0, 0))
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)


def status_color(value: str) -> tuple[int, int, int]:
    green = (60, 220, 90)
    yellow = (0, 215, 255)
    red = (0, 80, 255)
    gray = (150, 150, 150)
    normalized = value.upper()
    if normalized in {"SYNCED", "LOW", "SPEAKING", "MOVING", "OK"}:
        return green
    if normalized in {"MEDIUM", "UNCERTAIN", "STILL", "HIDDEN"}:
        return yellow
    if normalized in {"HIGH", "NOT_SYNCED", "ERROR"}:
        return red
    return gray


def build_normal_overlay_status(
    audio_activity: bool,
    lips_detected: bool,
    lip_activity: bool,
    mouth_hidden: bool,
    mouth_occluded: bool,
    sync_status: str,
) -> dict[str, str]:
    audio_status = "SPEAKING" if audio_activity else "SILENT"
    if mouth_occluded or mouth_hidden or (not lips_detected):
        lips_status = "HIDDEN"
    else:
        lips_status = "MOVING" if lip_activity else "STILL"
    if sync_status in {"SYNCED", "QUIET"}:
        sync_view = "SYNCED"
    elif sync_status in {"NOT_SYNCED", "MOUTH_HIDDEN", "MOUTH_OCCLUDED", "GATED_OUT"}:
        sync_view = "NOT_SYNCED"
    else:
        sync_view = "UNCERTAIN"
    if mouth_occluded or (audio_activity and (not lips_detected) and (not mouth_hidden)):
        risk = "HIGH"
    elif sync_view == "NOT_SYNCED":
        risk = "MEDIUM"
    else:
        risk = "LOW"
    return {"audio": audio_status, "lips": lips_status, "sync": sync_view, "risk": risk}


def draw_normal_overlay(frame: np.ndarray, status: dict[str, str]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w = frame.shape[:2]
    panel_h = 48
    x = 12
    y = h - panel_h - 12
    panel_w = min(w - 24, 860)
    draw_transparent_panel(frame, x, y, panel_w, panel_h, alpha=0.45, radius=14)
    labels = [("Audio", status["audio"]), ("Lips", status["lips"]), ("Sync", status["sync"]), ("Integrity Risk", status["risk"])]
    cursor_x = x + 16
    baseline_y = y + 31
    for title, value in labels:
        left = f"{title}: "
        cv2.putText(frame, left, (cursor_x, baseline_y), font, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
        text_size, _ = cv2.getTextSize(left, font, 0.55, 1)
        cursor_x += text_size[0]
        cv2.putText(frame, value, (cursor_x, baseline_y), font, 0.58, status_color(value), 2, cv2.LINE_AA)
        value_size, _ = cv2.getTextSize(value, font, 0.58, 2)
        cursor_x += value_size[0] + 20


def draw_debug_overlay(frame: np.ndarray, lines: list[str]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    h, w = frame.shape[:2]
    line_h = 18
    box_w = min(760, max(420, w - 28))
    x = max(8, w - box_w - 10)
    y = 8
    box_h = min(h - 16, line_h * len(lines) + 16)
    draw_transparent_panel(frame, x, y, box_w, box_h, alpha=0.44, radius=12)
    text_y = y + 18
    max_lines = max(1, (box_h - 12) // line_h)
    for idx, text in enumerate(lines[:max_lines]):
        color = (200, 200, 200)
        upper = text.upper()
        if "SYNCED" in upper or "VALID" in upper or "MIC: OK" in upper:
            color = (60, 220, 90)
        elif "NOT_SYNCED" in upper or "ERROR" in upper or "NO_FACE" in upper:
            color = (0, 80, 255)
        elif "UNCERTAIN" in upper or "HIDDEN" in upper or "GATED_OUT" in upper:
            color = (0, 215, 255)
        cv2.putText(frame, text, (x + 12, text_y + idx * line_h), font, font_scale, color, thickness, cv2.LINE_AA)


def ema(previous: float, current: float, alpha: float) -> float:
    alpha = min(max(alpha, 0.01), 1.0)
    return (alpha * current) + ((1.0 - alpha) * previous)


def compute_mar_mesh(landmarks: Any) -> float:
    points = landmarks.landmark
    upper_inner = points[13]
    lower_inner = points[14]
    left_corner = points[78]
    right_corner = points[308]
    vertical = abs(lower_inner.y - upper_inner.y)
    horizontal = max(abs(right_corner.x - left_corner.x), 1e-6)
    return float(vertical / horizontal)


def classify_rule_based(mouth_open: bool, audio_active: bool) -> str:
    if mouth_open and audio_active:
        return "VALID"
    if mouth_open and (not audio_active):
        return "SUSPICIOUS_MOUTH_OPEN_NO_AUDIO"
    if audio_active and (not mouth_open):
        return "SUSPICIOUS_AUDIO_NO_MOUTH_MOVEMENT"
    return "QUIET"


def extract_mouth_roi_gray(frame: np.ndarray, lip_box: tuple[int, int, int, int], pad: int = 2) -> Optional[np.ndarray]:
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = lip_box
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(w, x1 + pad)
    y1 = min(h, y1 + pad)
    if x1 <= x0 or y1 <= y0:
        return None
    roi = frame[y0:y1, x0:x1]
    if roi.size == 0:
        return None
    return cv2.resize(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (96, 48))


def compute_optical_flow_intensity(prev_roi: Optional[np.ndarray], curr_roi: Optional[np.ndarray]) -> float:
    if prev_roi is None or curr_roi is None:
        return 0.0
    flow = cv2.calcOpticalFlowFarneback(
        prev_roi,
        curr_roi,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return float(np.mean(mag))


def classify_optical_flow(motion_high: bool, audio_high: bool) -> str:
    if motion_high and audio_high:
        return "SYNCED"
    if (not motion_high) and (not audio_high):
        return "QUIET"
    return "NOT_SYNCED"


def compute_cross_correlation_score(
    audio_series: deque[float],
    lip_series: deque[float],
    max_lag: int,
) -> tuple[Optional[float], int]:
    if len(audio_series) != len(lip_series) or len(audio_series) < 6:
        return None, 0
    audio = np.array(audio_series, dtype=np.float32)
    lips = np.array(lip_series, dtype=np.float32)
    if np.std(audio) < 1e-6 or np.std(lips) < 1e-6:
        return None, 0
    best_score = -1.0
    best_lag = 0
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            a, l = audio[-lag:], lips[: len(lips) + lag]
        elif lag > 0:
            a, l = audio[:-lag], lips[lag:]
        else:
            a, l = audio, lips
        if len(a) < 4 or len(l) < 4:
            continue
        corr = np.corrcoef(a, l)[0, 1]
        if np.isnan(corr):
            continue
        if corr > best_score:
            best_score = float(corr)
            best_lag = lag
    if best_score < -0.99:
        return None, 0
    return best_score, best_lag


def classify_av_correlation(score: Optional[float], threshold: float) -> str:
    if score is None:
        return "INSUFFICIENT_DATA"
    return "SYNCED" if score >= threshold else "NOT_SYNCED"


def is_mar_non_degenerate(mar_value: float, mar_min: float, mar_max: float) -> bool:
    return bool(np.isfinite(mar_value) and mar_min <= mar_value <= mar_max)


def evaluate_multi_signal_gate(
    face_detected: bool,
    texture_std: Optional[float],
    flow_value: float,
    mar_value: float,
    flow_window: deque[bool],
    stability_streak: int,
) -> tuple[bool, int, dict[str, bool]]:
    if not face_detected:
        flow_window.clear()
        return False, 0, {"face_ok": False, "texture_ok": False, "flow_ok": False, "mar_ok": False, "stability_ok": False}
    texture_ok = (texture_std is not None) and (texture_std > MOUTH_HIDDEN_TEXTURE_THRESHOLD)
    flow_window.append(flow_value > GATE_FLOW_EPSILON)
    flow_ok = sum(flow_window) >= GATE_FLOW_MIN_COUNT
    mar_ok = is_mar_non_degenerate(mar_value, GATE_MAR_MIN, GATE_MAR_MAX)
    candidate = texture_ok and flow_ok and mar_ok
    stability_streak = (stability_streak + 1) if candidate else 0
    stability_ok = stability_streak >= GATE_STABILITY_FRAMES
    gate_ok = candidate and stability_ok
    return gate_ok, stability_streak, {
        "face_ok": True,
        "texture_ok": texture_ok,
        "flow_ok": flow_ok,
        "mar_ok": mar_ok,
        "stability_ok": stability_ok,
    }


def safe_nanmean(values: list[float]) -> float:
    valid = [v for v in values if np.isfinite(v)]
    if not valid:
        return float("nan")
    return float(np.mean(valid))


def majority_sync_status(rule_status: str, optical_status: str, corr_status: str) -> str:
    positive = 0
    negative = 0
    if rule_status == "VALID":
        positive += 1
    elif rule_status.startswith("SUSPICIOUS"):
        negative += 1
    if optical_status == "SYNCED":
        positive += 1
    elif optical_status == "NOT_SYNCED":
        negative += 1
    if corr_status == "SYNCED":
        positive += 1
    elif corr_status == "NOT_SYNCED":
        negative += 1
    if positive == 0 and negative == 0:
        return "QUIET"
    if positive > negative:
        return "SYNCED"
    if negative > positive:
        return "NOT_SYNCED"
    return "UNCERTAIN"


def classify_expression_mesh(landmarks: Any) -> str:
    points = landmarks.landmark
    left_corner = points[61]
    right_corner = points[291]
    upper_lip = points[13]
    lower_lip = points[14]
    brow_left = points[70]
    brow_right = points[300]
    eye_left_top = points[159]
    eye_right_top = points[386]
    forehead = points[10]
    chin = points[152]
    face_height = max(chin.y - forehead.y, 1e-6)
    mouth_open = abs(lower_lip.y - upper_lip.y) / face_height
    corner_avg_y = (left_corner.y + right_corner.y) / 2.0
    smile_curve = (upper_lip.y - corner_avg_y) / face_height
    brow_eye = ((eye_left_top.y - brow_left.y) + (eye_right_top.y - brow_right.y)) / (2.0 * face_height)
    if smile_curve > 0.012 and mouth_open > 0.010:
        return "HAPPY"
    if smile_curve < -0.006 and mouth_open < 0.03:
        return "SAD"
    if brow_eye > 0.16 and mouth_open > 0.018:
        return "NERVOUS"
    return "NORMAL"


def get_lip_contour_mesh(frame: np.ndarray, landmarks: Any) -> tuple[Optional[np.ndarray], Optional[tuple[int, int, int, int]]]:
    h, w = frame.shape[:2]
    lip_ids = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78]
    points = []
    for idx in lip_ids:
        point = landmarks.landmark[idx]
        points.append([int(point.x * w), int(point.y * h)])
    if len(points) < 3:
        return None, None
    contour = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    x, y, bw, bh = cv2.boundingRect(contour)
    return contour, (x, y, x + bw, y + bh)


def bbox_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
        return 0.0
    inter_area = float((inter_x1 - inter_x0) * (inter_y1 - inter_y0))
    area_a = float(max(0, ax1 - ax0) * max(0, ay1 - ay0))
    area_b = float(max(0, bx1 - bx0) * max(0, by1 - by0))
    denom = area_a + area_b - inter_area
    if denom <= 1e-6:
        return 0.0
    return inter_area / denom


def bbox_intersection_area(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
        return 0.0
    return float((inter_x1 - inter_x0) * (inter_y1 - inter_y0))


def get_hand_boxes(frame_shape: tuple[int, int, int], hands_result: Any) -> list[tuple[int, int, int, int]]:
    h, w = frame_shape[:2]
    boxes: list[tuple[int, int, int, int]] = []
    if not hands_result or not getattr(hands_result, "multi_hand_landmarks", None):
        return boxes
    for hand_lm in hands_result.multi_hand_landmarks:
        xs = [int(pt.x * w) for pt in hand_lm.landmark]
        ys = [int(pt.y * h) for pt in hand_lm.landmark]
        if not xs or not ys:
            continue
        x0, x1 = max(0, min(xs)), min(w - 1, max(xs))
        y0, y1 = max(0, min(ys)), min(h - 1, max(ys))
        pad_x = int((x1 - x0) * HAND_BOX_PADDING_RATIO) + 2
        pad_y = int((y1 - y0) * HAND_BOX_PADDING_RATIO) + 2
        x0 = max(0, x0 - pad_x)
        y0 = max(0, y0 - pad_y)
        x1 = min(w - 1, x1 + pad_x)
        y1 = min(h - 1, y1 + pad_y)
        if x1 > x0 and y1 > y0:
            boxes.append((x0, y0, x1, y1))
    return boxes


def is_mouth_occluded_by_hand(
    mouth_box: Optional[tuple[int, int, int, int]],
    hand_boxes: list[tuple[int, int, int, int]],
    iou_threshold: float,
) -> bool:
    if mouth_box is None:
        return False
    mx0, my0, mx1, my1 = mouth_box
    mouth_area = float(max(1, (mx1 - mx0) * (my1 - my0)))
    for hand_box in hand_boxes:
        iou = bbox_iou(mouth_box, hand_box)
        coverage = bbox_intersection_area(mouth_box, hand_box) / mouth_area
        if iou > iou_threshold or coverage >= MOUTH_OCCLUSION_COVERAGE_THRESHOLD:
            return True
    return False


def create_face_mesh_backend():
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
        return (
            mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.8,
                min_tracking_confidence=0.8,
            ),
            "mediapipe_face_mesh",
        )
    return None, "opencv_fallback"


def create_hands_backend():
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
        return (
            mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                model_complexity=1,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6,
            ),
            "mediapipe_hands",
        )
    return None, "hands_unavailable"

