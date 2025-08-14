from typing import Tuple

import cv2
import numpy as np


def center_crop(image_bgr: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
	h, w = image_bgr.shape[:2]
	th, tw = target_size

	scale = max(th / h, tw / w)
	resized = cv2.resize(image_bgr, (int(w * scale), int(h * scale)))
	rh, rw = resized.shape[:2]
	y1 = max(0, (rh - th) // 2)
	x1 = max(0, (rw - tw) // 2)
	crop = resized[y1:y1 + th, x1:x1 + tw]
	return crop
