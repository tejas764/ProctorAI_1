from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path
import time
from typing import Any

import numpy as np


@dataclass
class GazeReading:
    status: str
    confidence: float
    calibrated: bool
    progress: float
    error: str = ""
    step: str = ""
    step_index: int = 0
    total_steps: int = 0
    step_progress: float = 0.0
    awaiting_start: bool = False
    prompt: str = ""


# Keys from calibration_state() that are valid GazeReading fields
_READING_KEYS = frozenset(GazeReading.__dataclass_fields__.keys())


def _reading_kwargs(state: dict[str, Any]) -> dict[str, Any]:
    """Filter a calibration_state() dict to only keys accepted by GazeReading."""
    return {k: v for k, v in state.items() if k in _READING_KEYS}


class GazeEngine:
    def __init__(
        self,
        learning_frames: int = 140,
        smoothing: float = 0.7,
        mahalanobis_threshold: float = 3.0,
        outside_confirm_frames: int = 3,
        head_weight: float = 0.015,
        geometric_margin: float = 1.2,
        adapt_rate: float = 0.01,
    ) -> None:
        self.learning_frames = learning_frames
        self.smoothing = smoothing
        self.mahalanobis_threshold = mahalanobis_threshold
        self.outside_confirm_frames = outside_confirm_frames
        self.head_weight = head_weight
        self.geometric_margin = geometric_margin
        self.adapt_rate = adapt_rate

        self._module = None
        self._module_dir: Path | None = None
        self._models = None
        self._mean_gaze: np.ndarray | None = None
        self._inv_cov: np.ndarray | None = None
        self._h_threshold = 0.0
        self._v_threshold = 0.0
        self._learning_samples: list[list[float]] = []
        self._outside_counter = 0
        self._prev_dx = 0.0
        self._prev_dy = 0.0
        self._confidence_history: list[float] = []
        self._ready = False
        self._error = ""
        self._current_status = "INSIDE"
        self._calibration_steps: list[str] = ["CENTER", "LEFT", "RIGHT", "TOP", "BOTTOM"]
        self._frames_per_step = learning_frames
        self._wait_before_capture_s = 2.0
        self._geometric_soft_penalty = 0.5
        self._inside_threshold = 0.12
        self._outside_threshold = 0.08
        self._confidence_window = 6
        self._calibration_file = "calibration_data.npz"
        self._calib_index = 0
        self._start_pressed = False
        self._start_time: float | None = None
        self._capturing = False
        self._frames_captured = 0

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def error(self) -> str:
        return self._error

    @property
    def calibrated(self) -> bool:
        return self._mean_gaze is not None and self._inv_cov is not None

    def _candidate_module_paths(self) -> list[Path]:
        this_file = Path(__file__).resolve()
        outer_root = this_file.parents[2]
        inner_root = this_file.parents[1]
        candidates = [
            inner_root / "ProctorGuardAI-master" / "chunks.py",
            outer_root / "ProctorGuardAI-master" / "chunks.py",
        ]
        for root in (inner_root, outer_root):
            for d in root.iterdir():
                if not d.is_dir():
                    continue
                name = d.name.lower()
                if "proctor" in name and "guard" in name and "master" in name:
                    p = d / "chunks.py"
                    if p.exists():
                        candidates.append(p)
        # Keep order while removing duplicates.
        deduped: list[Path] = []
        seen: set[str] = set()
        for c in candidates:
            key = str(c.resolve())
            if key not in seen:
                seen.add(key)
                deduped.append(c)
        return deduped

    def _import_external_module(self) -> tuple[bool, str]:
        import_errors: list[str] = []
        for path in self._candidate_module_paths():
            if not path.exists():
                continue
            spec = importlib.util.spec_from_file_location("proctorguard_chunks_ext", str(path))
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
            except Exception as exc:  # noqa: BLE001
                import_errors.append(f"{path}: {exc}")
                continue
            self._module = module
            self._module_dir = path.parent
            return True, f"Loaded gaze module from {path}"
        if import_errors:
            return False, "Failed to import gaze module dependencies: " + " | ".join(import_errors)
        return False, "Gaze module file not found. Expected ProctorGuardAI-master/chunks.py"

    def _sync_module_config(self) -> None:
        if self._module is None:
            return
        self.smoothing = float(getattr(self._module, "SMOOTHING", self.smoothing))
        self.mahalanobis_threshold = float(getattr(self._module, "MAHALANOBIS_THRESHOLD", self.mahalanobis_threshold))
        self.outside_confirm_frames = int(getattr(self._module, "OUTSIDE_CONFIRM_FRAMES", self.outside_confirm_frames))
        self.head_weight = float(getattr(self._module, "HEAD_WEIGHT", self.head_weight))
        self.geometric_margin = float(getattr(self._module, "GEOMETRIC_MARGIN", self.geometric_margin))
        self.adapt_rate = float(getattr(self._module, "ADAPT_RATE", self.adapt_rate))
        self._calibration_steps = list(getattr(self._module, "CALIBRATION_STEPS", self._calibration_steps))
        self._frames_per_step = int(getattr(self._module, "FRAMES_PER_STEP", self.learning_frames))
        self._wait_before_capture_s = float(getattr(self._module, "WAIT_BEFORE_CAPTURE", self._wait_before_capture_s))
        self._geometric_soft_penalty = float(getattr(self._module, "GEOMETRIC_SOFT_PENALTY", self._geometric_soft_penalty))
        self._inside_threshold = float(getattr(self._module, "INSIDE_THRESHOLD", self._inside_threshold))
        self._outside_threshold = float(getattr(self._module, "OUTSIDE_THRESHOLD", self._outside_threshold))
        self._confidence_window = int(getattr(self._module, "CONF_WINDOW", self._confidence_window))
        self._calibration_file = str(getattr(self._module, "CALIBRATION_FILE", self._calibration_file))

    def _load_saved_calibration(self) -> None:
        if self._module is None:
            return
        loader = getattr(self._module, "load_calibration", None)
        if not callable(loader):
            return
        saved = loader()
        if not saved:
            return
        data = dict(saved)
        self._mean_gaze = np.array(data["mean_gaze"], dtype=np.float32)
        self._inv_cov = np.array(data["inv_cov"], dtype=np.float32)
        self._h_threshold = float(data["H_THRESHOLD"])
        self._v_threshold = float(data["V_THRESHOLD"])
        self._calib_index = len(self._calibration_steps)

    def start(self) -> tuple[bool, str]:
        ok, msg = self._import_external_module()
        if not ok:
            self._error = msg
            return False, msg

        self._sync_module_config()
        try:
            old_cwd = Path.cwd()
            try:
                if self._module_dir is not None:
                    os.chdir(self._module_dir)
                self._models = self._module.load_models()
                self._load_saved_calibration()
            finally:
                os.chdir(old_cwd)
        except Exception as exc:  # noqa: BLE001
            self._error = f"Gaze model load failed: {exc}"
            self._ready = False
            return False, self._error

        self._ready = True
        self._error = ""
        return True, msg

    def calibration_state(self) -> dict[str, Any]:
        step = self._calibration_steps[self._calib_index] if self._calib_index < len(self._calibration_steps) else ""
        if self.calibrated:
            prompt = "Calibration complete."
        elif self._capturing:
            prompt = f"Capturing {step}: keep eyes fixed and head steady."
        elif self._start_pressed:
            remaining = max(0.0, self._wait_before_capture_s - (time.time() - float(self._start_time or time.time())))
            prompt = f"Starting {step} capture in {remaining:.1f}s."
        else:
            prompt = f"Look at {step} and press Start." if step else "Calibration unavailable."
        total_expected = max(1, self._frames_per_step * max(1, len(self._calibration_steps)))
        return {
            "step": step,
            "step_index": int(min(self._calib_index, len(self._calibration_steps))),
            "total_steps": len(self._calibration_steps),
            "step_progress": 1.0 if self.calibrated else (self._frames_captured / float(max(1, self._frames_per_step))),
            "progress": 1.0 if self.calibrated else (len(self._learning_samples) / float(total_expected)),
            "awaiting_start": bool((not self.calibrated) and (not self._capturing) and (not self._start_pressed)),
            "capturing": self._capturing,
            "countdown_active": self._start_pressed and not self._capturing,
            "countdown_remaining": max(0.0, self._wait_before_capture_s - (time.time() - float(self._start_time or time.time()))) if self._start_pressed and not self._capturing else 0.0,
            "prompt": prompt,
        }

    def begin_calibration_step(self) -> tuple[bool, str, dict[str, Any]]:
        if not self._ready:
            return False, self._error or "not_ready", self.calibration_state()
        if self.calibrated:
            return True, "Calibration already complete.", self.calibration_state()
        if self._calib_index >= len(self._calibration_steps):
            return False, "No calibration steps remaining.", self.calibration_state()
        if self._start_pressed or self._capturing:
            return False, "Calibration step already in progress.", self.calibration_state()
        self._start_pressed = True
        self._start_time = time.time()
        self._frames_captured = 0
        return True, f"Starting {self._calibration_steps[self._calib_index]} calibration.", self.calibration_state()

    def reset_calibration(self, delete_saved: bool = True) -> tuple[bool, str, dict[str, Any]]:
        try:
            if delete_saved and self._module_dir is not None:
                calib_path = self._module_dir / self._calibration_file
                if calib_path.exists():
                    calib_path.unlink()
        except Exception as exc:  # noqa: BLE001
            self._error = f"Unable to reset saved calibration: {exc}"
            return False, self._error, self.calibration_state()

        self._mean_gaze = None
        self._inv_cov = None
        self._h_threshold = 0.0
        self._v_threshold = 0.0
        self._learning_samples = []
        self._outside_counter = 0
        self._prev_dx = 0.0
        self._prev_dy = 0.0
        self._confidence_history = []
        self._current_status = "INSIDE"
        self._calib_index = 0
        self._start_pressed = False
        self._start_time = None
        self._capturing = False
        self._frames_captured = 0
        self._error = ""
        return True, "Calibration reset.", self.calibration_state()

    def _finalize_calibration(self) -> None:
        samples = np.array(self._learning_samples, dtype=np.float32)
        if samples.size == 0:
            self._error = "No calibration samples collected."
            return
        center_mask = samples[:, 4] == 0
        center_samples = samples[center_mask][:, 0:4]
        if center_samples.size == 0:
            center_samples = samples[:, 0:4]
        self._mean_gaze = np.mean(center_samples, axis=0).astype(np.float32)
        features_4d = samples[:, 0:4]
        cov_gaze = np.cov(features_4d.T)
        self._inv_cov = np.linalg.inv(cov_gaze + 1e-6 * np.eye(4, dtype=np.float32)).astype(np.float32)
        horizontal_scores = np.abs(samples[:, 0]) + self.head_weight * np.abs(samples[:, 2])
        vertical_scores = np.abs(samples[:, 1]) + self.head_weight * np.abs(samples[:, 3])
        self._h_threshold = float(np.percentile(horizontal_scores, 95) * self.geometric_margin)
        self._v_threshold = float(np.percentile(vertical_scores, 95) * self.geometric_margin)
        saver = getattr(self._module, "save_calibration", None) if self._module is not None else None
        if callable(saver):
            old_cwd = Path.cwd()
            try:
                if self._module_dir is not None:
                    os.chdir(self._module_dir)
                saver(self._mean_gaze, self._inv_cov, self._h_threshold, self._v_threshold)
            finally:
                os.chdir(old_cwd)
        self._calib_index = len(self._calibration_steps)
        self._start_pressed = False
        self._start_time = None
        self._capturing = False
        self._frames_captured = 0
        self._confidence_history = []
        self._current_status = "INSIDE"
        self._outside_counter = 0

    def process(self, frame: np.ndarray) -> GazeReading:
        if not self._ready or self._module is None or self._models is None:
            state = self.calibration_state()
            return GazeReading(status="DISABLED", confidence=0.0, calibrated=False, error=self._error or "not_ready", **_reading_kwargs(state))

        try:
            feats = self._module.get_features(frame, self._models)
        except Exception as exc:  # noqa: BLE001
            self._error = str(exc)
            state = self.calibration_state()
            return GazeReading(status="ERROR", confidence=0.0, calibrated=self.calibrated, error=self._error, **_reading_kwargs(state))

        state = self.calibration_state()

        if not self.calibrated:
            if self._start_pressed and not self._capturing:
                if (time.time() - float(self._start_time or time.time())) >= self._wait_before_capture_s:
                    self._capturing = True
                    self._frames_captured = 0
                state = self.calibration_state()

            if feats is None:
                return GazeReading(status="NO_FACE", confidence=0.0, calibrated=False, error=self._error, **_reading_kwargs(state))

            if not self._capturing:
                return GazeReading(status="CALIBRATING", confidence=1.0, calibrated=False, error=self._error, **_reading_kwargs(state))

            dx_raw, dy_raw, yaw, pitch = feats
            dx = self.smoothing * self._prev_dx + (1.0 - self.smoothing) * float(dx_raw)
            dy = self.smoothing * self._prev_dy + (1.0 - self.smoothing) * float(dy_raw)
            self._prev_dx, self._prev_dy = dx, dy
            self._learning_samples.append([dx, dy, float(yaw), float(pitch), float(self._calib_index)])
            self._frames_captured += 1

            if self._frames_captured >= self._frames_per_step:
                self._capturing = False
                self._start_pressed = False
                self._start_time = None
                self._frames_captured = 0
                self._calib_index += 1
                if self._calib_index >= len(self._calibration_steps):
                    self._finalize_calibration()
                    state = self.calibration_state()
                    return GazeReading(status="CALIBRATED", confidence=1.0, calibrated=True, error=self._error, **_reading_kwargs(state))

            state = self.calibration_state()
            return GazeReading(status="CALIBRATING", confidence=1.0, calibrated=False, error=self._error, **_reading_kwargs(state))

        if feats is None:
            return GazeReading(status="NO_FACE", confidence=0.0, calibrated=True, error=self._error, **_reading_kwargs(state))

        dx_raw, dy_raw, yaw, pitch = feats
        dx = self.smoothing * self._prev_dx + (1.0 - self.smoothing) * float(dx_raw)
        dy = self.smoothing * self._prev_dy + (1.0 - self.smoothing) * float(dy_raw)
        self._prev_dx, self._prev_dy = dx, dy

        diff = np.array([dx, dy, float(yaw), float(pitch)], dtype=np.float32) - self._mean_gaze
        maha = float(np.sqrt(diff.T @ self._inv_cov @ diff))
        maha_score = max(0.0, 1.0 - (maha / self.mahalanobis_threshold))
        horizontal_score = abs(dx) + self.head_weight * abs(float(yaw))
        vertical_score = abs(dy) + self.head_weight * abs(float(pitch))
        geometric_inside = horizontal_score <= self._h_threshold and vertical_score <= self._v_threshold
        final_score = maha_score if geometric_inside else maha_score * self._geometric_soft_penalty
        confidence = float(max(0.0, min(1.0, final_score)))
        self._confidence_history.append(confidence)
        if len(self._confidence_history) > self._confidence_window:
            self._confidence_history.pop(0)
        smoothed_confidence = float(np.mean(self._confidence_history))

        if self._current_status == "INSIDE":
            if smoothed_confidence >= self._outside_threshold:
                status = "INSIDE"
                self._outside_counter = 0
                self._mean_gaze = (1.0 - self.adapt_rate) * self._mean_gaze + self.adapt_rate * np.array([dx, dy, float(yaw), float(pitch)], dtype=np.float32)
            else:
                self._outside_counter += 1
                if self._outside_counter >= self.outside_confirm_frames:
                    status = "OUTSIDE"
                    self._current_status = "OUTSIDE"
                else:
                    status = "INSIDE"
        else:
            if smoothed_confidence >= self._inside_threshold:
                status = "INSIDE"
                self._current_status = "INSIDE"
                self._outside_counter = 0
                self._mean_gaze = (1.0 - self.adapt_rate) * self._mean_gaze + self.adapt_rate * np.array([dx, dy, float(yaw), float(pitch)], dtype=np.float32)
            else:
                status = "OUTSIDE"
                self._outside_counter += 1

        state = self.calibration_state()
        return GazeReading(status=status, confidence=smoothed_confidence, calibrated=True, error=self._error, **_reading_kwargs(state))

    def progress(self) -> float:
        return float(self.calibration_state()["progress"])
