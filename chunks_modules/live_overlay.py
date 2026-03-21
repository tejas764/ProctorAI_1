from __future__ import annotations

from collections import deque
import time
from typing import Optional

import cv2
import numpy as np

from chunks_modules.config import (
    CAMERA_INDEX,
    CORR_MAX_LAG_FRAMES,
    CORR_THRESHOLD,
    CORR_WINDOW_FRAMES,
    EMA_ALPHA,
    GATE_FLOW_WINDOW_FRAMES,
    HAND_MOUTH_IOU_THRESHOLD,
    LIP_MOTION_THRESHOLD,
    MAHAL_THRESHOLD,
    MIC_BLOCK_SIZE,
    MIC_DEVICE_INDEX,
    MIC_SAMPLE_RATE,
    MIC_THRESHOLD,
    MOUTH_HIDDEN_TEXTURE_THRESHOLD,
    MOUTH_OCCLUSION_COVERAGE_THRESHOLD,
    NERVOUS_LIP_MULTIPLIER,
    OPTICAL_FLOW_THRESHOLD,
    RULE_MAR_DELTA_THRESHOLD,
    RULE_MAR_THRESHOLD,
    WINDOW_SIZE,
)
from chunks_modules.shared import (
    MicrophoneVoiceMonitor,
    OverlayUIState,
    build_normal_overlay_status,
    calculate_mahalanobis,
    classify_av_correlation,
    classify_expression_mesh,
    classify_optical_flow,
    classify_rule_based,
    compute_cross_correlation_score,
    compute_mar_mesh,
    compute_optical_flow_intensity,
    create_face_mesh_backend,
    create_hands_backend,
    detect_speaking,
    draw_debug_overlay,
    draw_normal_overlay,
    ema,
    evaluate_multi_signal_gate,
    extract_lip_features,
    extract_mouth_roi_gray,
    get_hand_boxes,
    get_lip_contour_mesh,
    is_mouth_occluded_by_hand,
    majority_sync_status,
    safe_resize,
)


def run_live_voice_overlay() -> None:
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}.")

    mic = MicrophoneVoiceMonitor(
        sample_rate=MIC_SAMPLE_RATE,
        block_size=MIC_BLOCK_SIZE,
        threshold=MIC_THRESHOLD,
        device_index=MIC_DEVICE_INDEX,
    )
    mic_error: Optional[str] = None
    try:
        mic.start()
    except Exception as error:
        mic_error = str(error)

    face_mesh, vision_backend = create_face_mesh_backend()
    hands_detector, hands_backend = create_hands_backend()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    face_cascade_alt = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
    face_cascade_profile = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
    previous_mouth_roi: Optional[np.ndarray] = None
    smoothed_audio_rms = 0.0
    smoothed_lip_metric = 0.0
    smoothed_flow_metric = 0.0
    previous_mar = 0.0
    gate_stability_streak = 0
    flow_gate_window: deque[bool] = deque(maxlen=GATE_FLOW_WINDOW_FRAMES)
    audio_energy_window: deque[float] = deque(maxlen=CORR_WINDOW_FRAMES)
    lip_energy_window: deque[float] = deque(maxlen=CORR_WINDOW_FRAMES)
    last_face_bbox: Optional[tuple[int, int, int, int]] = None
    last_face_time = 0.0
    last_lip_box: Optional[tuple[int, int, int, int]] = None
    last_lip_time = 0.0
    height_w = deque(maxlen=WINDOW_SIZE)
    movement_w = deque(maxlen=WINDOW_SIZE)
    feature_w = deque(maxlen=WINDOW_SIZE)
    last_speak = None
    session_start = time.time()
    ui_state = OverlayUIState(persistence_s=0.7)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            t = time.time() - session_start
            lips_detected = False
            lip_activity = False
            mahal_value: Optional[float] = None
            mar_value = 0.0
            mar_delta = 0.0
            optical_flow_value = 0.0
            corr_score: Optional[float] = None
            corr_lag = 0
            texture_std: Optional[float] = None
            gate_components = {"face_ok": False, "texture_ok": False, "flow_ok": False, "mar_ok": False, "stability_ok": False}
            expression = "NORMAL"
            active_lip_threshold = MAHAL_THRESHOLD if face_mesh is not None else LIP_MOTION_THRESHOLD
            lip_box: Optional[tuple[int, int, int, int]] = None
            lip_contour: Optional[np.ndarray] = None
            hand_boxes: list[tuple[int, int, int, int]] = []
            mouth_hidden = False
            face_mesh_detected = False
            hands_detected = False

            if face_mesh is not None:
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if hands_detector is not None:
                    hands_result = hands_detector.process(image_rgb)
                    hand_boxes = get_hand_boxes(frame.shape, hands_result)
                    hands_detected = len(hand_boxes) > 0
                res = face_mesh.process(image_rgb)
                if res.multi_face_landmarks:
                    lm = res.multi_face_landmarks[0]
                    p = lm.landmark[200]
                    if 0 <= p.x <= 1 and 0 <= p.y <= 1:
                        face_mesh_detected = True
                    if face_mesh_detected:
                        features = extract_lip_features(lm, image_rgb.shape, height_w, movement_w, feature_w)
                        mahal_value = float(calculate_mahalanobis(feature_w, features))
                        lip_activity, last_speak = detect_speaking(features, t, feature_w, last_speak)
                        mar_value = compute_mar_mesh(lm)
                        mar_delta = abs(mar_value - previous_mar)
                        previous_mar = mar_value
                        expression = classify_expression_mesh(lm)
                        lip_contour, lip_box = get_lip_contour_mesh(frame, lm)
                        if lip_box is not None:
                            x0, y0, x1, y1 = lip_box
                            roi = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
                            if roi.size > 0:
                                texture_std = float(np.std(roi))
                            curr_mouth_roi = extract_mouth_roi_gray(frame, lip_box)
                            if (curr_mouth_roi is not None) and (previous_mouth_roi is not None):
                                # Ensure shapes match for optical flow.
                                if curr_mouth_roi.shape != previous_mouth_roi.shape:
                                    previous_mouth_roi = safe_resize(previous_mouth_roi, (curr_mouth_roi.shape[1], curr_mouth_roi.shape[0]))

                                if previous_mouth_roi is not None:
                                    flow = cv2.calcOpticalFlowFarneback(
                                        previous_mouth_roi,
                                        curr_mouth_roi,
                                        None,
                                        0.5,
                                        3,
                                        15,
                                        3,
                                        5,
                                        1.2,
                                        0,
                                    )
                                    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                                    optical_flow_value = np.mean(mag)
                            previous_mouth_roi = curr_mouth_roi
                lips_detected, gate_stability_streak, gate_components = evaluate_multi_signal_gate(
                    face_mesh_detected,
                    texture_std,
                    optical_flow_value,
                    mar_value,
                    flow_gate_window,
                    gate_stability_streak,
                )
                mouth_hidden = face_mesh_detected and (not gate_components["texture_ok"])
                if not face_mesh_detected:
                    previous_mouth_roi = None
                    previous_mar = 0.0
            else:
                if hands_detector is not None:
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    hands_result = hands_detector.process(image_rgb)
                    hand_boxes = get_hand_boxes(frame.shape, hands_result)
                    hands_detected = len(hand_boxes) > 0
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_eq = cv2.equalizeHist(gray)
                faces = face_cascade.detectMultiScale(gray_eq, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
                if len(faces) == 0:
                    faces = face_cascade_alt.detectMultiScale(gray_eq, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
                if len(faces) == 0:
                    faces = face_cascade_profile.detectMultiScale(gray_eq, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
                selected_face: Optional[tuple[int, int, int, int]] = None
                if len(faces) > 0:
                    selected_face = max(faces, key=lambda item: item[2] * item[3])
                    last_face_bbox = selected_face
                    last_face_time = t
                elif last_face_bbox is not None and (t - last_face_time) < 0.6:
                    selected_face = last_face_bbox
                if selected_face is not None:
                    lips_detected = True
                    x, y, w, h = selected_face
                    mouth_y0, mouth_y1 = y + int(h * 0.55), y + h
                    mouth_x0, mouth_x1 = x + int(w * 0.20), x + int(w * 0.80)
                    lip_box = (mouth_x0, mouth_y0, mouth_x1, mouth_y1)
                    mouth_roi = gray[mouth_y0:mouth_y1, mouth_x0:mouth_x1]
                    motion_value = 0.0
                    if mouth_roi.size > 0:
                        if float(np.std(mouth_roi)) < MOUTH_HIDDEN_TEXTURE_THRESHOLD:
                            mouth_hidden = True
                        mouth_roi = cv2.resize(mouth_roi, (64, 32))
                        if previous_mouth_roi is not None:
                            motion_value = float(np.mean(cv2.absdiff(mouth_roi, previous_mouth_roi)))
                        previous_mouth_roi = mouth_roi
                    mahal_value = motion_value
                    optical_flow_value = motion_value
                    lip_activity = motion_value >= LIP_MOTION_THRESHOLD
                    face_roi = gray[y : y + h, x : x + w]
                    smiles = smile_cascade.detectMultiScale(face_roi, scaleFactor=1.7, minNeighbors=22, minSize=(25, 25))
                    expression = "HAPPY" if len(smiles) > 0 else "NORMAL"

            mouth_occluded = is_mouth_occluded_by_hand(lip_box, hand_boxes, HAND_MOUTH_IOU_THRESHOLD)
            if mouth_occluded:
                lips_detected = False
                mouth_hidden = False
            if lips_detected and lip_box is not None:
                last_lip_box = lip_box
                last_lip_time = t
            elif (not lips_detected) and last_lip_box is not None and (t - last_lip_time) < 0.9:
                lip_box = last_lip_box
                mouth_hidden = True

            lip_color = (0, 0, 255) if (mouth_hidden or not lips_detected) else (0, 255, 0)
            if lip_contour is not None:
                cv2.polylines(frame, [lip_contour], True, lip_color, 1, cv2.LINE_AA)
            if lip_box is not None:
                x0, y0, x1, y1 = lip_box
                cv2.rectangle(frame, (x0, y0), (x1, y1), lip_color, 1)
            for hand_box in hand_boxes:
                hx0, hy0, hx1, hy1 = hand_box
                cv2.rectangle(frame, (hx0, hy0), (hx1, hy1), (255, 255, 0), 1)

            audio_rms = mic.current_rms() if mic_error is None else 0.0
            smoothed_audio_rms = ema(smoothed_audio_rms, audio_rms, EMA_ALPHA)
            audio_activity = smoothed_audio_rms >= MIC_THRESHOLD if mic_error is None else False
            smoothed_lip_metric = ema(smoothed_lip_metric, mahal_value if mahal_value is not None else 0.0, EMA_ALPHA)
            smoothed_flow_metric = ema(smoothed_flow_metric, optical_flow_value, EMA_ALPHA)
            lip_activity = smoothed_lip_metric >= active_lip_threshold if lips_detected else False

            rule_status = classify_rule_based((mar_value >= RULE_MAR_THRESHOLD) or (mar_delta >= RULE_MAR_DELTA_THRESHOLD), audio_activity)
            optical_status = classify_optical_flow(smoothed_flow_metric >= OPTICAL_FLOW_THRESHOLD, audio_activity)
            audio_energy_window.append(smoothed_audio_rms)
            lip_energy_window.append(smoothed_flow_metric)
            corr_score, corr_lag = compute_cross_correlation_score(audio_energy_window, lip_energy_window, CORR_MAX_LAG_FRAMES)
            corr_status = classify_av_correlation(corr_score, CORR_THRESHOLD)
            nervous_signal = lips_detected and lip_activity and (not audio_activity) and (
                smoothed_lip_metric >= (active_lip_threshold * NERVOUS_LIP_MULTIPLIER)
            )
            if nervous_signal:
                expression = "NERVOUS"

            if not lips_detected:
                if mouth_occluded:
                    sync_status, expression = "MOUTH_OCCLUDED", "MOUTH_OCCLUDED"
                elif face_mesh is not None and gate_components["face_ok"]:
                    sync_status, expression = "GATED_OUT", "GATED_OUT"
                else:
                    sync_status, expression = "NO_FACE", "NO_FACE"
            elif mouth_hidden:
                sync_status, expression = "MOUTH_HIDDEN", "MOUTH_HIDDEN"
            else:
                sync_status = majority_sync_status(rule_status, optical_status, corr_status)

            debug_lines = [
                f"Audio Voice: {'SPEAKING' if audio_activity else 'SILENT'}",
                f"Audio RMS: {smoothed_audio_rms:.5f} (thr={MIC_THRESHOLD:.5f})",
                "Mouth Occluded: YES" if mouth_occluded else f"Lips Detected: {'YES' if lips_detected else 'NO'}",
                f"Hands Detected: {'YES' if hands_detected else 'NO'}",
                f"Occlusion Thr: IoU>{HAND_MOUTH_IOU_THRESHOLD:.2f} or Coverage>{MOUTH_OCCLUSION_COVERAGE_THRESHOLD:.2f}",
                f"Gate Face/Tex/Flow/MAR/Stable: {int(gate_components['face_ok'])}/{int(gate_components['texture_ok'])}/{int(gate_components['flow_ok'])}/{int(gate_components['mar_ok'])}/{int(gate_components['stability_ok'])}",
                f"Lip Activity: {'SPEAKING' if lip_activity else 'SILENT'}",
                f"Lip Metric: {smoothed_lip_metric:.4f} (thr={active_lip_threshold:.4f})",
                f"Rule-Based: {rule_status} (MAR={mar_value:.3f}, d={mar_delta:.3f})",
                f"Optical Flow: {optical_status} (flow={smoothed_flow_metric:.3f}, thr={OPTICAL_FLOW_THRESHOLD:.3f})",
                f"AV Corr: {corr_status} (score={corr_score if corr_score is not None else float('nan'):.3f}, lag={corr_lag})",
                f"LipSync Status: {sync_status} (ensemble)",
                f"Expression: {expression}",
                f"Vision Backend: {vision_backend}",
                f"Hands Backend: {hands_backend}",
                "Mic: ERROR (check device/index)" if mic_error else "Mic: OK",
                "Controls: D=Normal/Debug, H=Hide/Show, Q=Quit",
            ]
            normal_status = build_normal_overlay_status(
                audio_activity=audio_activity,
                lips_detected=lips_detected,
                lip_activity=lip_activity,
                mouth_hidden=mouth_hidden,
                mouth_occluded=mouth_occluded,
                sync_status=sync_status,
            )
            normal_status = ui_state.smoothed_normal(t, normal_status)
            if ui_state.visible:
                if ui_state.debug_mode:
                    draw_debug_overlay(frame, debug_lines)
                else:
                    draw_normal_overlay(frame, normal_status)
            cv2.imshow("Live Voice + LipSync Overlay", frame)
            if ui_state.handle_key(cv2.waitKey(1) & 0xFF):
                break
    finally:
        mic.stop()
        cap.release()
        if face_mesh is not None:
            face_mesh.close()
        if hands_detector is not None:
            hands_detector.close()
        cv2.destroyAllWindows()
