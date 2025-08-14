from pathlib import Path
from typing import Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class CowFaceDataset(Dataset):
	def __init__(self, root_dir: str, img_size: int = 224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), is_train: bool = True, hflip_prob: float = 0.5, color_jitter=(0.1, 0.1, 0.1, 0.05)):
		self.root_dir = Path(root_dir)
		self.samples = []
		self.class_to_idx = {}
		self.idx_to_class = []

		# Scan folders
		for idx, cls_dir in enumerate(sorted([p for p in self.root_dir.iterdir() if p.is_dir()])):
			self.class_to_idx[cls_dir.name] = idx
			self.idx_to_class.append(cls_dir.name)
			for img_path in sorted(cls_dir.rglob("*.jpg")) + sorted(cls_dir.rglob("*.png")) + sorted(cls_dir.rglob("*.jpeg")):
				self.samples.append((img_path, idx))

		# Transforms
		train_tfms = [
			T.Resize((img_size, img_size)),
			T.RandomHorizontalFlip(p=hflip_prob),
			T.ColorJitter(*color_jitter),
			T.ToTensor(),
			T.Normalize(mean, std),
		]
		val_tfms = [
			T.Resize((img_size, img_size)),
			T.ToTensor(),
			T.Normalize(mean, std),
		]
		self.transform = T.Compose(train_tfms if is_train else val_tfms)

	def __len__(self) -> int:
		return len(self.samples)

	def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str]:
		img_path, label = self.samples[index]
		img = Image.open(img_path).convert("RGB")
		img = self.transform(img)
		return img, label, str(img_path)
