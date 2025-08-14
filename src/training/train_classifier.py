import argparse
import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm

from src.datasets.cowface_dataset import CowFaceDataset
from src.models.vit_arcface import ViTArcFaceModel, ViTArcFaceConfig


def set_seed(seed: int):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


def load_config(path: str):
	with open(path, "r") as f:
		return yaml.safe_load(f)


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
	preds = logits.argmax(dim=1)
	return (preds == labels).float().mean().item()


def train_epoch(model, loader, optimizer, scaler, device):
	model.train()
	loss_fn = nn.CrossEntropyLoss()
	pbar = tqdm(loader, desc="train", leave=False)
	running_loss, running_acc = 0.0, 0.0
	for images, labels, _ in pbar:
		images = images.to(device)
		labels = labels.to(device)

		optimizer.zero_grad()
		with torch.cuda.amp.autocast(enabled=scaler is not None):
			embeddings, logits = model(images, labels)
			loss = loss_fn(logits, labels)
		acc = accuracy_from_logits(logits.detach(), labels)
		if scaler is not None:
			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()
		else:
			loss.backward()
			optimizer.step()
		running_loss += loss.item() * images.size(0)
		running_acc += acc * images.size(0)
		pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{acc:.3f}"})
	return running_loss / len(loader.dataset), running_acc / len(loader.dataset)


def eval_epoch(model, loader, device):
	model.eval()
	loss_fn = nn.CrossEntropyLoss()
	running_loss, running_acc = 0.0, 0.0
	with torch.no_grad():
		for images, labels, _ in tqdm(loader, desc="val", leave=False):
			images = images.to(device)
			labels = labels.to(device)
			_, logits = model(images, labels)
			loss = loss_fn(logits, labels)
			acc = accuracy_from_logits(logits, labels)
			running_loss += loss.item() * images.size(0)
			running_acc += acc * images.size(0)
	return running_loss / len(loader.dataset), running_acc / len(loader.dataset)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_root", type=str, required=True, help="Root folder containing train/ and val/ directories")
	parser.add_argument("--config", type=str, default="configs/default.yaml")
	parser.add_argument("--epochs", type=int, default=None)
	parser.add_argument("--batch_size", type=int, default=None)
	parser.add_argument("--lr", type=float, default=None)
	parser.add_argument("--weight_decay", type=float, default=None)
	parser.add_argument("--num_workers", type=int, default=None)
	args = parser.parse_args()

	cfg = load_config(args.config)
	seed = cfg["train"]["seed"]
	set_seed(seed)

	device = "cuda" if torch.cuda.is_available() else "cpu"

	img_size = cfg["input"]["img_size"]
	mean = tuple(cfg["input"]["mean"])
	std = tuple(cfg["input"]["std"])
	hflip_prob = cfg["augment"]["hflip_prob"]
	color_jitter = tuple(cfg["augment"]["color_jitter"])

	train_dir = str(Path(args.data_root) / "train")
	val_dir = str(Path(args.data_root) / "val")

	train_ds = CowFaceDataset(train_dir, img_size, mean, std, is_train=True, hflip_prob=hflip_prob, color_jitter=color_jitter)
	val_ds = CowFaceDataset(val_dir, img_size, mean, std, is_train=False)

	batch_size = args.batch_size or cfg["train"]["batch_size"]
	num_workers = args.num_workers or cfg["train"]["num_workers"]
	train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
	val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

	num_classes = len(train_ds.idx_to_class)
	model_cfg = ViTArcFaceConfig(
		vit_name=cfg["model"]["vit_name"],
		pretrained=cfg["model"]["pretrained"],
		embed_dim=cfg["model"]["embed_dim"],
		arcface_scale=cfg["model"]["arcface"]["scale"],
		arcface_margin=cfg["model"]["arcface"]["margin"],
		num_classes=num_classes,
	)
	model = ViTArcFaceModel(model_cfg).to(device)

	epochs = args.epochs or cfg["train"]["epochs"]
	lr = args.lr or cfg["train"]["lr"]
	weight_decay = args.weight_decay or cfg["train"]["weight_decay"]

	optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
	scaler = torch.cuda.amp.GradScaler(enabled=cfg["train"]["amp"] and device == "cuda")

	runs_dir = Path("runs/checkpoints")
	runs_dir.mkdir(parents=True, exist_ok=True)
	best_val_acc = 0.0

	for epoch in range(1, epochs + 1):
		print(f"Epoch {epoch}/{epochs}")
		train_loss, train_acc = train_epoch(model, train_loader, optimizer, scaler, device)
		print(f"train: loss={train_loss:.4f}, acc={train_acc:.4f}")

		if (epoch % cfg["train"]["val_interval"]) == 0:
			val_loss, val_acc = eval_epoch(model, val_loader, device)
			print(f"val  : loss={val_loss:.4f}, acc={val_acc:.4f}")
			if val_acc > best_val_acc:
				best_val_acc = val_acc
				ckpt_path = runs_dir / "best.pt"
				torch.save({
					"model_state": model.state_dict(),
					"config": model_cfg.__dict__,
					"classes": train_ds.idx_to_class,
				}, ckpt_path)
				print(f"Saved best checkpoint to {ckpt_path}")

	# Final checkpoint
	final_ckpt = runs_dir / "last.pt"
	torch.save({
		"model_state": model.state_dict(),
		"config": model_cfg.__dict__,
		"classes": train_ds.idx_to_class,
	}, final_ckpt)
	print(f"Saved final checkpoint to {final_ckpt}")


if __name__ == "__main__":
	main()
