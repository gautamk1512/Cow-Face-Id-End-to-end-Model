import argparse
from pathlib import Path

import cv2
from tqdm import tqdm

from src.detection.detect_faces import YOLOv5Detector


def process_dir(input_dir: Path, output_dir: Path, yolo_weights: str = "", min_size: int = 64):
	output_dir.mkdir(parents=True, exist_ok=True)
	detector = YOLOv5Detector(weights=yolo_weights if yolo_weights else None)

	class_dirs = [p for p in input_dir.iterdir() if p.is_dir()]
	for cls_dir in class_dirs:
		out_cls = output_dir / cls_dir.name
		out_cls.mkdir(parents=True, exist_ok=True)

		images = sorted(list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png")) + list(cls_dir.glob("*.jpeg")))
		for img_path in tqdm(images, desc=f"{cls_dir.name}"):
			img = cv2.imread(str(img_path))
			if img is None:
				continue
			boxes = detector.detect(img)
			if len(boxes) == 0:
				continue
			# take highest confidence
			boxes.sort(key=lambda b: b[-1], reverse=True)
			x1, y1, x2, y2, conf = boxes[0]
			crop = YOLOv5Detector.crop_with_padding(img, (x1, y1, x2, y2), pad_ratio=0.15)
			h, w = crop.shape[:2]
			if min(h, w) < min_size:
				continue
			out_path = out_cls / img_path.name
			cv2.imwrite(str(out_path), crop)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--input_dir", type=str, required=True, help="Raw images organized by class folders")
	parser.add_argument("--output_dir", type=str, required=True, help="Where to save cropped faces")
	parser.add_argument("--yolo_weights", type=str, default="", help="Path to custom YOLOv5 .pt for cow faces (optional)")
	parser.add_argument("--min_size", type=int, default=64)
	args = parser.parse_args()

	process_dir(Path(args.input_dir), Path(args.output_dir), args.yolo_weights, args.min_size)


if __name__ == "__main__":
	main()
