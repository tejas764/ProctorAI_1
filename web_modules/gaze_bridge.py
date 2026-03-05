from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path

import numpy as np


@dataclass
class GazeReading:
    status: str
    confidence: float
    calibrated: bool
    progress: float
    error: str = ""


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
        self._horizontal_scores: list[float] = []
        self._vertical_scores: list[float] = []
        self._outside_counter = 0
        self._prev_dx = 0.0
        self._prev_dy = 0.0
        self._ready = False
        self._error = ""

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
        return [
            outer_root / "ProctorGuardAI-master" / "proctorguard_mahalanobis.py",
            inner_root / "ProctorGuardAI-master" / "proctorguard_mahalanobis.py",
        ]

    def _import_external_module(self) -> tuple[bool, str]:
        import_errors: list[str] = []
        for path in self._candidate_module_paths():
            if not path.exists():
                continue
            spec = importlib.util.spec_from_file_location("proctorguard_mahalanobis_ext", str(path))
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
        return False, "Gaze module file not found. Expected ProctorGuardAI-master/proctorguard_mahalanobis.py"

    def start(self) -> tuple[bool, str]:
        ok, msg = self._import_external_module()
        if not ok:
            self._error = msg
            return False, msg

        try:
            old_cwd = Path.cwd()
            try:
                if self._module_dir is not None:
                    os.chdir(self._module_dir)
                self._models = self._module.load_models()
            finally:
                os.chdir(old_cwd)
        except Exception as exc:  # noqa: BLE001
            self._error = f"Gaze model load failed: {exc}"
            self._ready = False
            return False, self._error

        self._ready = True
        self._error = ""
        return True, msg

    def process(self, frame: np.ndarray) -> GazeReading:
        if not self._ready or self._module is None or self._models is None:
            return GazeReading(status="DISABLED", confidence=0.0, calibrated=False, progress=0.0, error=self._error or "not_ready")

        try:
            feats = self._module.get_features(frame, self._models)
        except Exception as exc:  # noqa: BLE001
            self._error = str(exc)
            return GazeReading(status="ERROR", confidence=0.0, calibrated=self.calibrated, progress=self.progress(), error=self._error)

        if feats is None:
            return GazeReading(status="NO_FACE", confidence=0.0, calibrated=self.calibrated, progress=self.progress())

        dx_raw, dy_raw, yaw, pitch = feats
        dx = self.smoothing * self._prev_dx + (1.0 - self.smoothing) * float(dx_raw)
        dy = self.smoothing * self._prev_dy + (1.0 - self.smoothing) * float(dy_raw)
        self._prev_dx, self._prev_dy = dx, dy

        horizontal_score = abs(dx) + self.head_weight * abs(float(yaw))
        vertical_score = abs(dy) + self.head_weight * abs(float(pitch))

        if not self.calibrated:
            self._learning_samples.append([dx, dy])
            self._horizontal_scores.append(float(horizontal_score))
            self._vertical_scores.append(float(vertical_score))

            if len(self._learning_samples) >= self.learning_frames:
                samples = np.array(self._learning_samples, dtype=np.float32)
                self._mean_gaze = np.mean(samples, axis=0)
                cov_gaze = np.cov(samples.T)
                self._inv_cov = np.linalg.inv(cov_gaze + 1e-6 * np.eye(2, dtype=np.float32))
                self._h_threshold = float(np.percentile(np.array(self._horizontal_scores, dtype=np.float32), 95) * self.geometric_margin)
                self._v_threshold = float(np.percentile(np.array(self._vertical_scores, dtype=np.float32), 95) * self.geometric_margin)
                return GazeReading(status="CALIBRATED", confidence=1.0, calibrated=True, progress=1.0)
            return GazeReading(status="CALIBRATING", confidence=1.0, calibrated=False, progress=self.progress())

        diff = np.array([dx, dy], dtype=np.float32) - self._mean_gaze
        maha = float(np.sqrt(diff.T @ self._inv_cov @ diff))
        maha_score = max(0.0, 1.0 - (maha / self.mahalanobis_threshold))
        geometric_inside = horizontal_score <= self._h_threshold and vertical_score <= self._v_threshold

        final_score = float(maha_score if geometric_inside else 0.0)
        if final_score > 0.3:
            status = "INSIDE"
            self._outside_counter = 0
            self._mean_gaze = (1.0 - self.adapt_rate) * self._mean_gaze + self.adapt_rate * np.array([dx, dy], dtype=np.float32)
        else:
            self._outside_counter += 1
            status = "OUTSIDE" if self._outside_counter >= self.outside_confirm_frames else "INSIDE"

        conf = float(max(0.0, min(1.0, final_score)))
        return GazeReading(status=status, confidence=conf, calibrated=True, progress=1.0)

    def progress(self) -> float:
        return float(max(0.0, min(1.0, len(self._learning_samples) / float(max(1, self.learning_frames)))))
