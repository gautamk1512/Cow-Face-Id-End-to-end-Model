from pathlib import Path
from typing import List, Tuple, Optional

import torch
import cv2
import numpy as np


class YOLOv5Detector:
	def __init__(self, weights: Optional[str] = None, device: Optional[str] = None, conf_thres: float = 0.25, iou_thres: float = 0.45):
		self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
		self.conf_thres = conf_thres
		self.iou_thres = iou_thres

		if weights and Path(weights).suffix == ".pt":
			# Try to load a custom yolov5 model via torch.hub (weights path)
			self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights, force_reload=False)
		else:
			# Default to yolov5s from hub
			self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, force_reload=False)
		self.model = self.model.to(self.device)

	def detect(self, image_bgr: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
		# Run inference (expects RGB)
		image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
		results = self.model(image_rgb, size=640)
		# results.xyxy[0]: [x1,y1,x2,y2,conf,cls]
		pred = results.xyxy[0].detach().cpu().numpy()
		boxes = []
		for x1, y1, x2, y2, conf, cls in pred:
			if conf < self.conf_thres:
				continue
			x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
			boxes.append((x1, y1, x2, y2, float(conf)))
		return boxes

	@staticmethod
	def crop_with_padding(image_bgr: np.ndarray, box: Tuple[int, int, int, int], pad_ratio: float = 0.1) -> np.ndarray:
		h, w = image_bgr.shape[:2]
		x1, y1, x2, y2 = box
		bw = x2 - x1
		bh = y2 - y1
		pad_w = int(bw * pad_ratio)
		pad_h = int(bh * pad_ratio)
		x1 = max(0, x1 - pad_w)
		y1 = max(0, y1 - pad_h)
		x2 = min(w, x2 + pad_w)
		y2 = min(h, y2 + pad_h)
		return image_bgr[y1:y2, x1:x2]
