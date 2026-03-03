from __future__ import annotations

import asyncio
from collections import deque
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from chunks_modules.config import (
    CORR_MAX_LAG_FRAMES,
    CORR_THRESHOLD,
    CORR_WINDOW_FRAMES,
    GATE_FLOW_WINDOW_FRAMES,
    HAND_MOUTH_IOU_THRESHOLD,
    OPTICAL_FLOW_THRESHOLD,
    RULE_MAR_DELTA_THRESHOLD,
    RULE_MAR_THRESHOLD,
    WINDOW_SIZE,
)
from chunks_modules.media import is_vad_speaking
from chunks_modules.shared import (
    calculate_mahalanobis,
    classify_av_correlation,
    classify_optical_flow,
    classify_rule_based,
    compute_cross_correlation_score,
    compute_mar_mesh,
    compute_optical_flow_intensity,
    evaluate_multi_signal_gate,
    extract_lip_features,
    extract_mouth_roi_gray,
    get_hand_boxes,
    get_lip_contour_mesh,
    is_mouth_occluded_by_hand,
    majority_sync_status,
    safe_nanmean,
)


async def process_chunk_async(
    start_f: int,
    end_f: int,
    video_path: str,
    audio_path: str,
    fps: float,
    vad_segments: list[tuple[float, float]],
) -> list[dict]:
    return await asyncio.to_thread(process_chunk, (start_f, end_f, video_path, audio_path, fps, vad_segments))


def process_chunk(args: tuple[int, int, str, str, float, list[tuple[float, float]]]) -> list[dict]:
    start_f, end_f, video_path, audio_path, fps, vad_segments = args
    del audio_path
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8,
    )
    hands_detector = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    height_w = deque(maxlen=WINDOW_SIZE)
    movement_w = deque(maxlen=WINDOW_SIZE)
    feature_w = deque(maxlen=WINDOW_SIZE)
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    frame_idx = start_f
    curr_sec = int(start_f / fps)
    lip_seen: list[bool] = []
    rule_sync: list[bool] = []
    optical_sync: list[bool] = []
    corr_sync: list[bool] = []
    corr_scores: list[float] = []
    mar_values: list[float] = []
    flow_values: list[float] = []
    mahal_values: list[float] = []
    audio_energy_window: deque[float] = deque(maxlen=CORR_WINDOW_FRAMES)
    lip_energy_window: deque[float] = deque(maxlen=CORR_WINDOW_FRAMES)
    previous_mouth_roi: Optional[np.ndarray] = None
    previous_mar = 0.0
    gate_stability_streak = 0
    flow_gate_window: deque[bool] = deque(maxlen=GATE_FLOW_WINDOW_FRAMES)
    out = []

    while cap.isOpened() and frame_idx < end_f:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        t = frame_idx / fps
        sec = int(t)
        img = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        res = face_mesh.process(img)
        seen = False
        lip_box: Optional[tuple[int, int, int, int]] = None
        texture_std: Optional[float] = None
        mar_value = 0.0
        mar_delta = 0.0
        motion_value = 0.0
        hands_result = hands_detector.process(img)
        hand_boxes = get_hand_boxes(frame.shape, hands_result)
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0]
            p = lm.landmark[200]
            if 0 <= p.x <= 1 and 0 <= p.y <= 1:
                seen = True
                _, lip_box = get_lip_contour_mesh(frame, lm)
                mar_value = compute_mar_mesh(lm)
                mar_delta = abs(mar_value - previous_mar)
                previous_mar = mar_value
                if lip_box is not None:
                    x0, y0, x1, y1 = lip_box
                    roi = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
                    if roi.size > 0:
                        texture_std = float(np.std(roi))
                    curr_mouth_roi = extract_mouth_roi_gray(frame, lip_box)
                    motion_value = compute_optical_flow_intensity(previous_mouth_roi, curr_mouth_roi)
                    previous_mouth_roi = curr_mouth_roi
        gated_seen, gate_stability_streak, _ = evaluate_multi_signal_gate(
            seen, texture_std, motion_value, mar_value, flow_gate_window, gate_stability_streak
        )
        if is_mouth_occluded_by_hand(lip_box, hand_boxes, HAND_MOUTH_IOU_THRESHOLD):
            gated_seen = False
        if not seen:
            previous_mouth_roi = None
            previous_mar = 0.0
        lip_seen.append(gated_seen)

        if seen and gated_seen:
            features = extract_lip_features(lm, img.shape, height_w, movement_w, feature_w)
            mahal_val = calculate_mahalanobis(feature_w, features)
            vad_b = is_vad_speaking(t, vad_segments)
            mouth_open = (mar_value >= RULE_MAR_THRESHOLD) or (mar_delta >= RULE_MAR_DELTA_THRESHOLD)
            rb = classify_rule_based(mouth_open, vad_b)
            op = classify_optical_flow(motion_value >= OPTICAL_FLOW_THRESHOLD, vad_b)
            audio_energy_window.append(1.0 if vad_b else 0.0)
            lip_energy_window.append(motion_value)
            corr_score, _ = compute_cross_correlation_score(audio_energy_window, lip_energy_window, CORR_MAX_LAG_FRAMES)
            av = classify_av_correlation(corr_score, CORR_THRESHOLD)
            rule_sync.append(rb == "VALID")
            optical_sync.append(op == "SYNCED")
            corr_sync.append(av == "SYNCED")
            mahal_values.append(float(mahal_val))
            corr_scores.append(corr_score if corr_score is not None else np.nan)
            mar_values.append(mar_value)
            flow_values.append(motion_value)
        else:
            rule_sync.append(False)
            optical_sync.append(False)
            corr_sync.append(False)
            mahal_values.append(np.nan)
            corr_scores.append(np.nan)
            mar_values.append(np.nan)
            flow_values.append(np.nan)

        if sec > curr_sec:
            if not any(lip_seen):
                out.append(
                    {
                        "Time (s)": curr_sec,
                        "VAD Speaking": "No Lips Detected",
                        "Mahalanobis Status": "No Lips Detected",
                        "Rule-Based": "No Lips Detected",
                        "Optical Flow": "No Lips Detected",
                        "AV Correlation": "No Lips Detected",
                        "MAR Mean": np.nan,
                        "Flow Mean": np.nan,
                        "Corr Score Mean": np.nan,
                        "Ensemble": "NO_FACE",
                    }
                )
            else:
                rb_ratio = float(np.mean(rule_sync)) if rule_sync else 0.0
                op_ratio = float(np.mean(optical_sync)) if optical_sync else 0.0
                av_ratio = float(np.mean(corr_sync)) if corr_sync else 0.0
                rb_state = "SYNCED" if rb_ratio >= 0.5 else "NOT_SYNCED"
                op_state = "SYNCED" if op_ratio >= 0.5 else "NOT_SYNCED"
                av_state = "SYNCED" if av_ratio >= 0.5 else "NOT_SYNCED"
                ensemble = majority_sync_status(
                    "VALID" if rb_state == "SYNCED" else "SUSPICIOUS_AUDIO_NO_MOUTH_MOVEMENT",
                    op_state,
                    av_state,
                )
                out.append(
                    {
                        "Time (s)": curr_sec,
                        "VAD Speaking": any(rule_sync),
                        "Mahalanobis Status": safe_nanmean(mahal_values),
                        "Rule-Based": rb_state,
                        "Optical Flow": op_state,
                        "AV Correlation": av_state,
                        "MAR Mean": safe_nanmean(mar_values),
                        "Flow Mean": safe_nanmean(flow_values),
                        "Corr Score Mean": safe_nanmean(corr_scores),
                        "Ensemble": ensemble,
                    }
                )
            curr_sec = sec
            lip_seen, rule_sync, optical_sync, corr_sync = [], [], [], []
            corr_scores, mar_values, flow_values, mahal_values = [], [], [], []

    cap.release()
    face_mesh.close()
    hands_detector.close()
    return out

