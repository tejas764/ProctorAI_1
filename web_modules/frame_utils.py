import logging
import cv2
import numpy as np
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

def is_valid_frame(frame: Optional[np.ndarray]) -> bool:
    if frame is None:
        return False
    if not hasattr(frame, "size") or frame.size == 0:
        return False
    if len(frame.shape) < 2:
        return False
    # Check if dimensions are > 0
    if frame.shape[0] <= 0 or frame.shape[1] <= 0:
        return False
    return True

def safe_resize(frame: Optional[np.ndarray], target_size: Tuple[int, int], interpolation: int = cv2.INTER_AREA, fx: float = 0.0, fy: float = 0.0) -> Optional[np.ndarray]:
    """
    Safely resizes a frame.
    If target_size is (0,0), uses fx and fy.
    Returns None if the frame is invalid or resize fails.
    """
    if not is_valid_frame(frame):
        # Only log debug or warning if this happens unexpectedly often
        # logger.debug("Skipping invalid frame before resize")
        return None

    try:
        return cv2.resize(frame, target_size, fx=fx, fy=fy, interpolation=interpolation)
    except cv2.error as e:
        logger.warning(f"Skipping frame due to resize failure: {e}")
        return None

