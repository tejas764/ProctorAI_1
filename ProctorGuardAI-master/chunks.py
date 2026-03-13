"""
ProctorGuard AI - Hybrid 3.3 (4D Mahalanobis + Recalibrated Confidence Range)
Modified:
- 4D Mahalanobis (gaze + head pose)
- Soft geometric gating (raised penalty)
- Hysteresis-based INSIDE/OUTSIDE switching (recalibrated to actual confidence range)
- Temporal confidence smoothing
- Interactive calibration (Start button per step)
- Head-pose tracking during calibration
- Ground-truth labeling (press 'i' for INSIDE, 'o' for OUTSIDE)
- Saves ground truth to CSV for evaluation
- Persistent calibration save/load (press 'r' to recalibrate)
"""

import cv2
import time
import csv
import os
import numpy as np
from openvino import Core

# ==========================
# CONFIGURATION
# ==========================

DEVICE = "CPU"
FRAME_WIDTH = 640
EYE_SIZE = 40

SMOOTHING = 0.6

CALIBRATION_STEPS = ["CENTER", "LEFT", "RIGHT", "TOP", "BOTTOM"]
FRAMES_PER_STEP = 90
WAIT_BEFORE_CAPTURE = 2.0
ADAPT_RATE = 0.005

MAHALANOBIS_THRESHOLD = 3.5
OUTSIDE_CONFIRM_FRAMES = 6

HEAD_WEIGHT = 0.005
GEOMETRIC_MARGIN = 1.10
GEOMETRIC_SOFT_PENALTY = 0.05
DOWN_RELAX_FACTOR = 0.5

INSIDE_THRESHOLD = 0.08
OUTSIDE_THRESHOLD = 0.04

CONF_WINDOW = 6

LOG_FILE = "gaze_log.csv"
CALIBRATION_FILE = "calibration_data.npz"

PROMPT_COLOR = (255, 0, 0)
PROGRESS_COLOR = (255, 0, 0)
BUTTON_COLOR = (50, 200, 50)
NO_FACE_COLOR = (0, 0, 255)
INSIDE_COLOR = (0, 255, 0)

# ==========================
# MODEL LOADING
# ==========================

def load_models():
    ie = Core()
    return {
        "face": ie.compile_model(
            ie.read_model("intel/face-detection-retail-0004/FP32/face-detection-retail-0004.xml"),
            DEVICE),
        "landmarks": ie.compile_model(
            ie.read_model("intel/facial-landmarks-35-adas-0002/FP32/facial-landmarks-35-adas-0002.xml"),
            DEVICE),
        "head_pose": ie.compile_model(
            ie.read_model("intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml"),
            DEVICE),
        "gaze": ie.compile_model(
            ie.read_model("intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002.xml"),
            DEVICE),
    }

# ==========================
# UTILITIES
# ==========================

def preprocess(img, shape):
    _, _, H, W = shape
    img = cv2.resize(img, (W, H))
    img = img.transpose(2, 0, 1)[None].astype(np.float32)
    return img

def crop_square(img, center, size):
    x, y = center
    s = size // 2
    x1, x2 = max(0, x - s), min(img.shape[1], x + s)
    y1, y2 = max(0, y - s), min(img.shape[0], y + s)
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)
    return cv2.resize(roi, (size, size))

def largest_face(dets, shape, conf=0.6):
    H, W = shape[:2]
    best = None
    for det in dets[0][0]:
        if det[2] < conf:
            continue
        x1, y1, x2, y2 = (det[3:] * [W, H, W, H]).astype(int)
        area = (x2 - x1) * (y2 - y1)
        if best is None or area > best[0]:
            best = (area, (x1, y1, x2, y2))
    return None if best is None else best[1]

# ==========================
# CALIBRATION SAVE / LOAD
# ==========================

def save_calibration(mean_gaze, inv_cov, H_THRESHOLD, V_THRESHOLD):
    np.savez(
        CALIBRATION_FILE,
        mean_gaze=mean_gaze,
        inv_cov=inv_cov,
        H_THRESHOLD=np.array([H_THRESHOLD]),
        V_THRESHOLD=np.array([V_THRESHOLD]),
    )
    print("Calibration saved to disk.")

def load_calibration():
    if not os.path.exists(CALIBRATION_FILE):
        return None
    data = np.load(CALIBRATION_FILE)
    return {
        "mean_gaze": data["mean_gaze"],
        "inv_cov": data["inv_cov"],
        "H_THRESHOLD": float(data["H_THRESHOLD"][0]),
        "V_THRESHOLD": float(data["V_THRESHOLD"][0]),
    }

# ==========================
# FEATURE EXTRACTION
# ==========================

def get_features(frame, models):

    scale = FRAME_WIDTH / frame.shape[1]
    frame = cv2.resize(frame, (FRAME_WIDTH, int(frame.shape[0] * scale)))

    face_blob = preprocess(frame, models["face"].input(0).shape)
    dets = list(models["face"](face_blob).values())[0]

    bbox = largest_face(dets, frame.shape)
    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox
    face = frame[y1:y2, x1:x2]

    lm_blob = preprocess(face, models["landmarks"].input(0).shape)
    lm = list(models["landmarks"](lm_blob).values())[0].reshape(-1)

    fx, fy = face.shape[1], face.shape[0]
    left_eye = (x1 + int(lm[0] * fx), y1 + int(lm[1] * fy))
    right_eye = (x1 + int(lm[2] * fx), y1 + int(lm[3] * fy))

    le = crop_square(frame, left_eye, EYE_SIZE)
    re = crop_square(frame, right_eye, EYE_SIZE)

    hp_blob = preprocess(face, models["head_pose"].input(0).shape)
    hp_out = models["head_pose"](hp_blob)
    yaw, pitch, roll = [v.flatten()[0] for v in hp_out.values()]

    gz_inputs = {
        models["gaze"].inputs[0].any_name:
            preprocess(le, models["gaze"].inputs[0].shape),
        models["gaze"].inputs[1].any_name:
            preprocess(re, models["gaze"].inputs[1].shape),
        models["gaze"].inputs[2].any_name:
            np.array([[yaw, pitch, roll]], dtype=np.float32)
    }

    gv = list(models["gaze"](gz_inputs).values())[0][0]

    dx = gv[0] + yaw * 0.002
    dy = gv[1] + pitch * 0.002

    return float(dx), float(dy), float(yaw), float(pitch)

# ==========================
# MAIN ENGINE
# ==========================

def run():

    models = load_models()
    cap = cv2.VideoCapture(0)

    prev_dx, prev_dy = 0, 0
    outside_counter = 0

    learning_samples = []
    learned = False

    mean_gaze = None
    inv_cov = None
    H_THRESHOLD = 0.0
    V_THRESHOLD = 0.0

    ground_truth = 0  # 0 = INSIDE, 1 = OUTSIDE

    calib_index = 0
    start_pressed = False
    start_time = None
    capturing = False
    frames_captured = 0

    confidence_history = []
    current_status = "INSIDE"

    BTN_W, BTN_H = 180, 50
    BTN_X, BTN_Y = 30, 80
    button_rect = (BTN_X, BTN_Y, BTN_W, BTN_H)
    button_clicked = False

    # --- Attempt to load saved calibration ---
    saved = load_calibration()
    if saved is not None:
        mean_gaze = saved["mean_gaze"]
        inv_cov = saved["inv_cov"]
        H_THRESHOLD = saved["H_THRESHOLD"]
        V_THRESHOLD = saved["V_THRESHOLD"]
        learned = True
        calib_index = len(CALIBRATION_STEPS)
        print("Loaded saved calibration.")
    else:
        print("No calibration found. Starting calibration...")

    def on_mouse(event, x, y, flags, param):
        nonlocal button_clicked
        bx, by, bw, bh = button_rect
        if event == cv2.EVENT_LBUTTONDOWN:
            if bx <= x <= bx + bw and by <= y <= by + bh:
                button_clicked = True

    cv2.namedWindow("ProctorGuard Hybrid 3.3")
    cv2.setMouseCallback("ProctorGuard Hybrid 3.3", on_mouse)

    log = open(LOG_FILE, "w", newline="")
    writer = csv.writer(log)
    writer.writerow(["timestamp", "status", "confidence", "ground_truth"])

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        feats = get_features(frame, models)

        status = "NO_FACE"
        confidence = 0.0

        # ==========================
        # CALIBRATION
        # ==========================
        if not learned:

            current_step = CALIBRATION_STEPS[calib_index] if calib_index < len(CALIBRATION_STEPS) else None

            if current_step is not None:

                cv2.putText(frame, f"Calibration: Look at {current_step}", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, PROMPT_COLOR, 2)

                bx, by, bw, bh = button_rect
                cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), BUTTON_COLOR, -1)
                cv2.putText(frame, "START", (bx + 20, by + 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

                if button_clicked and not start_pressed and not capturing:
                    start_pressed = True
                    start_time = time.time()
                    button_clicked = False

                if start_pressed and not capturing:
                    elapsed = time.time() - start_time
                    remaining = max(0.0, WAIT_BEFORE_CAPTURE - elapsed)
                    cv2.putText(frame, f"Starting in {remaining:.1f}s", (30, 140),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, PROMPT_COLOR, 2)
                    if elapsed >= WAIT_BEFORE_CAPTURE:
                        capturing = True
                        frames_captured = 0

                if capturing and feats is not None:
                    dx_raw, dy_raw, yaw, pitch = feats

                    dx = SMOOTHING * prev_dx + (1 - SMOOTHING) * dx_raw
                    dy = SMOOTHING * prev_dy + (1 - SMOOTHING) * dy_raw
                    prev_dx, prev_dy = dx, dy

                    learning_samples.append([dx, dy, yaw, pitch, calib_index])
                    frames_captured += 1

                    cv2.putText(frame, f"Capturing {frames_captured}/{FRAMES_PER_STEP}",
                                (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, INSIDE_COLOR, 2)

                    if frames_captured >= FRAMES_PER_STEP:
                        capturing = False
                        start_pressed = False
                        calib_index += 1

            # 4D Mahalanobis calibration completion — use only CENTER for baseline
            if calib_index >= len(CALIBRATION_STEPS):
                samples = np.array(learning_samples)

                # Use only CENTER samples for mean (calibration step 0)
                center_mask = samples[:, 4] == 0
                center_samples = samples[center_mask][:, 0:4]
                mean_gaze = np.mean(center_samples, axis=0)

                # Use ALL samples for covariance (captures full variance)
                features_4d = samples[:, 0:4]
                cov_gaze = np.cov(features_4d.T)
                inv_cov = np.linalg.inv(cov_gaze + 1e-6 * np.eye(4))

                horizontal_scores = np.abs(samples[:, 0]) + HEAD_WEIGHT * np.abs(samples[:, 2])
                vertical_scores = np.abs(samples[:, 1]) + HEAD_WEIGHT * np.abs(samples[:, 3])

                H_THRESHOLD = np.percentile(horizontal_scores, 95) * GEOMETRIC_MARGIN
                V_THRESHOLD = np.percentile(vertical_scores, 95) * GEOMETRIC_MARGIN

                learned = True
                save_calibration(mean_gaze, inv_cov, H_THRESHOLD, V_THRESHOLD)

            total_expected = FRAMES_PER_STEP * len(CALIBRATION_STEPS)
            cv2.putText(frame, f"Progress: {len(learning_samples)}/{total_expected}",
                        (30, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, PROGRESS_COLOR, 2)

            status = "CALIBRATING"

        # ==========================
        # OPERATIONAL
        # ==========================
        else:

            if feats is None:
                status = "NO_FACE"
            else:
                dx_raw, dy_raw, yaw, pitch = feats

                dx = SMOOTHING * prev_dx + (1 - SMOOTHING) * dx_raw
                dy = SMOOTHING * prev_dy + (1 - SMOOTHING) * dy_raw
                prev_dx, prev_dy = dx, dy

                # 4D Mahalanobis distance
                diff = np.array([dx, dy, yaw, pitch]) - mean_gaze
                maha = np.sqrt(diff.T @ inv_cov @ diff)
                maha_score = max(0, 1 - maha / MAHALANOBIS_THRESHOLD)

                horizontal_score = abs(dx) + HEAD_WEIGHT * abs(yaw)
                vertical_score = abs(dy) + HEAD_WEIGHT * abs(pitch)

                # Relax downward gaze/head-pose to allow keyboard glances
                if (dy - mean_gaze[1]) > 0:
                    vertical_score *= DOWN_RELAX_FACTOR

                geometric_inside = (horizontal_score <= H_THRESHOLD and
                                    vertical_score <= V_THRESHOLD)

                # Soft geometric gating
                if geometric_inside:
                    final_score = maha_score
                else:
                    final_score = maha_score * GEOMETRIC_SOFT_PENALTY

                confidence = max(0, min(1, final_score))

                # Temporal confidence smoothing
                confidence_history.append(confidence)
                if len(confidence_history) > CONF_WINDOW:
                    confidence_history.pop(0)
                smoothed_confidence = np.mean(confidence_history)

                # Hysteresis-based decision
                if current_status == "INSIDE":
                    if smoothed_confidence >= OUTSIDE_THRESHOLD:
                        status = "INSIDE"
                        outside_counter = 0
                        mean_gaze = (1 - ADAPT_RATE) * mean_gaze + ADAPT_RATE * np.array([dx, dy, yaw, pitch])
                    else:
                        outside_counter += 1
                        if outside_counter >= OUTSIDE_CONFIRM_FRAMES:
                            status = "OUTSIDE"
                            current_status = "OUTSIDE"
                        else:
                            status = "INSIDE"
                else:  # current_status == "OUTSIDE"
                    if smoothed_confidence >= INSIDE_THRESHOLD:
                        status = "INSIDE"
                        current_status = "INSIDE"
                        outside_counter = 0
                        mean_gaze = (1 - ADAPT_RATE) * mean_gaze + ADAPT_RATE * np.array([dx, dy, yaw, pitch])
                    else:
                        status = "OUTSIDE"
                        outside_counter += 1

                confidence = smoothed_confidence

        # ==========================
        # LOGGING
        # ==========================
        writer.writerow([
            time.time(),
            status,
            round(float(confidence), 3),
            ground_truth
        ])
        log.flush()

        # ==========================
        # OVERLAY (only after calibration)
        # ==========================
        if learned:
            color = INSIDE_COLOR if status == "INSIDE" else NO_FACE_COLOR
            cv2.putText(frame, f"{status} | Conf:{confidence:.2f}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            gt_text = "GT: INSIDE" if ground_truth == 0 else "GT: OUTSIDE"
            cv2.putText(frame, gt_text, (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.putText(frame, f"H_th:{H_THRESHOLD:.3f} V_th:{V_THRESHOLD:.3f}",
                        (30, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.putText(frame, "Press 'r' to recalibrate", (30, 360),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow("ProctorGuard Hybrid 3.3", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        if not learned and key == ord('s'):
            button_clicked = True
        if key == ord('i'):
            ground_truth = 0
        if key == ord('o'):
            ground_truth = 1
        if key == ord('r'):
            if os.path.exists(CALIBRATION_FILE):
                os.remove(CALIBRATION_FILE)
                print("Calibration deleted. Restarting calibration...")
            # Reset all calibration state
            learned = False
            learning_samples = []
            calib_index = 0
            start_pressed = False
            start_time = None
            capturing = False
            frames_captured = 0
            confidence_history = []
            current_status = "INSIDE"
            outside_counter = 0
            prev_dx, prev_dy = 0, 0
            mean_gaze = None
            inv_cov = None
            H_THRESHOLD = 0.0
            V_THRESHOLD = 0.0

    cap.release()
    log.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
