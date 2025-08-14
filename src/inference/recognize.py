import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import yaml
import torchvision.transforms as T

from src.models.vit_arcface import ViTArcFaceModel, ViTArcFaceConfig


def load_checkpoint(ckpt_path: str) -> Tuple[ViTArcFaceModel, List[str]]:
	ckpt = torch.load(ckpt_path, map_location="cpu")
	cfg_dict = ckpt["config"]
	classes = ckpt["classes"]
	cfg = ViTArcFaceConfig(**cfg_dict)
	# Inference: no classifier head needed, but safe to load
	model = ViTArcFaceModel(cfg)
	model.load_state_dict(ckpt["model_state"], strict=False)
	model.eval()
	return model, classes


def build_transform(img_size: int, mean, std):
	return T.Compose([
		T.Resize((img_size, img_size)),
		T.ToTensor(),
		T.Normalize(mean, std),
	])


def embed_images(model: ViTArcFaceModel, image_paths: List[Path], transform, device: str) -> torch.Tensor:
	embs = []
	with torch.no_grad():
		for p in image_paths:
			img = Image.open(p).convert("RGB")
			img = transform(img).unsqueeze(0).to(device)
			e, _ = model(img, None)
			embs.append(e.cpu())
	return torch.cat(embs, dim=0)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--checkpoint", type=str, required=True)
	parser.add_argument("--gallery_dir", type=str, required=True)
	parser.add_argument("--query", type=str, required=True)
	parser.add_argument("--top_k", type=int, default=5)
	parser.add_argument("--threshold", type=float, default=0.35, help="Cosine distance threshold for unknown")
	parser.add_argument("--config", type=str, default="configs/default.yaml")
	args = parser.parse_args()

	with open(args.config, "r") as f:
		config = yaml.safe_load(f)

	img_size = config["input"]["img_size"]
	mean = tuple(config["input"]["mean"])
	std = tuple(config["input"]["std"])
	device = "cuda" if torch.cuda.is_available() else "cpu"

	model, classes = load_checkpoint(args.checkpoint)
	model = model.to(device)
	transform = build_transform(img_size, mean, std)

	# Build gallery
	gallery_paths = []
	gallery_labels = []
	for cls_idx, cls in enumerate(classes):
		cls_dir = Path(args.gallery_dir) / cls
		if not cls_dir.is_dir():
			continue
		for p in sorted(list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png")) + list(cls_dir.glob("*.jpeg"))):
			gallery_paths.append(p)
			gallery_labels.append(cls_idx)

	if len(gallery_paths) == 0:
		raise RuntimeError("No gallery images found.")

	gallery_embs = embed_images(model, gallery_paths, transform, device)
	gallery_norm = F.normalize(gallery_embs, dim=1)

	# Query
	query_emb = embed_images(model, [Path(args.query)], transform, device)
	query_norm = F.normalize(query_emb, dim=1)

	# Cosine similarity and distances
	sim = torch.mm(query_norm, gallery_norm.t()).squeeze(0)
	dist = 1.0 - sim

	# Top-K
	vals, idxs = torch.topk(sim, k=min(args.top_k, sim.numel()), largest=True)
	results = []
	for v, i in zip(vals.tolist(), idxs.tolist()):
		results.append((gallery_paths[i].name, classes[gallery_labels[i]], float(v), float(1.0 - v)))

	best_name, best_class, best_sim, best_dist = results[0]
	label = best_class if best_dist <= args.threshold else "unknown"

	print({
		"predicted_label": label,
		"best_match_file": best_name,
		"best_similarity": round(best_sim, 4),
		"best_distance": round(best_dist, 4),
		"top_k": results,
	})


if __name__ == "__main__":
	main()
