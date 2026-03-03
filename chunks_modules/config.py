from __future__ import annotations

from typing import Literal, Optional

import yaml


with open("config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

CONFIG_PATH: str = cfg["config_path"]
VIDEO_PATH: str = cfg["video_path"]
WINDOW_SIZE: int = int(cfg["window_size"])
SILENCE_DELAY: float = float(cfg["silence_delay"])
MAHAL_THRESHOLD: float = float(cfg["mahal_threshold"])
NUM_CORES: int = int(cfg["num_cores"])
ONLY_VAD: Literal[True, False] = bool(cfg["only_vad"])
LIVE_MODE: Literal[True, False] = bool(cfg.get("live_mode", True))
CAMERA_INDEX: int = int(cfg.get("camera_index", 0))
MIC_DEVICE_INDEX: Optional[int] = cfg.get("mic_device_index")
MIC_SAMPLE_RATE: int = int(cfg.get("mic_sample_rate", 16000))
MIC_BLOCK_SIZE: int = int(cfg.get("mic_block_size", 1024))
MIC_THRESHOLD: float = float(cfg.get("mic_threshold", 0.015))
LIP_MOTION_THRESHOLD: float = float(cfg.get("lip_motion_threshold", 6.0))
EMA_ALPHA: float = float(cfg.get("ema_alpha", 0.35))
NERVOUS_LIP_MULTIPLIER: float = float(cfg.get("nervous_lip_multiplier", 1.4))
MOUTH_HIDDEN_TEXTURE_THRESHOLD: float = float(cfg.get("mouth_hidden_texture_threshold", 10.0))
RULE_MAR_THRESHOLD: float = float(cfg.get("rule_mar_threshold", 0.20))
RULE_MAR_DELTA_THRESHOLD: float = float(cfg.get("rule_mar_delta_threshold", 0.015))
OPTICAL_FLOW_THRESHOLD: float = float(cfg.get("optical_flow_threshold", 0.90))
CORR_WINDOW_FRAMES: int = int(cfg.get("corr_window_frames", 24))
CORR_THRESHOLD: float = float(cfg.get("corr_threshold", 0.30))
CORR_MAX_LAG_FRAMES: int = int(cfg.get("corr_max_lag_frames", 4))
GATE_FLOW_EPSILON: float = float(cfg.get("gate_flow_epsilon", 0.35))
GATE_FLOW_WINDOW_FRAMES: int = int(cfg.get("gate_flow_window_frames", 12))
GATE_FLOW_MIN_COUNT: int = int(cfg.get("gate_flow_min_count", 2))
GATE_MAR_MIN: float = float(cfg.get("gate_mar_min", 0.02))
GATE_MAR_MAX: float = float(cfg.get("gate_mar_max", 1.20))
GATE_STABILITY_FRAMES: int = int(cfg.get("gate_stability_frames", 4))
HAND_MOUTH_IOU_THRESHOLD: float = float(cfg.get("hand_mouth_iou_threshold", 0.03))
HAND_BOX_PADDING_RATIO: float = float(cfg.get("hand_box_padding_ratio", 0.20))
MOUTH_OCCLUSION_COVERAGE_THRESHOLD: float = float(cfg.get("mouth_occlusion_coverage_threshold", 0.20))

