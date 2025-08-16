from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import timm

from .arcface import ArcMarginProduct


@dataclass
class ViTArcFaceConfig:
	vit_name: str = "vit_base_patch16_224"
	pretrained: bool = True
	embed_dim: int = 512
	arcface_scale: float = 64.0
	arcface_margin: float = 0.5
	num_classes: Optional[int] = None


class ViTArcFaceModel(nn.Module):
	def __init__(self, config: ViTArcFaceConfig):
		super().__init__()
		self.config = config

		# Backbone outputs feature embeddings (num_classes=0 -> feature extractor)
		self.backbone = timm.create_model(config.vit_name, pretrained=config.pretrained, num_classes=0)
		backbone_dim = self.backbone.num_features

		# Projection to desired embedding size
		self.embedding_head = nn.Sequential(
			nn.Linear(backbone_dim, config.embed_dim),
			nn.BatchNorm1d(config.embed_dim),
		)

		if config.num_classes is not None:
			self.arcface = ArcMarginProduct(config.embed_dim, config.num_classes, s=config.arcface_scale, m=config.arcface_margin)
		else:
			self.arcface = None

	def forward_features(self, x: torch.Tensor) -> torch.Tensor:
		features = self.backbone(x)
		embeddings = self.embedding_head(features)
		return embeddings

	def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
		embeddings = self.forward_features(x)
		logits = None
		if self.arcface is not None:
			logits = self.arcface(embeddings, labels)
		return embeddings, logits
	
	def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
		"""Get embedding for inference"""
		return self.forward_features(x)


class ViTArcFace(nn.Module):
	"""Simplified interface for ViTArcFace model"""
	def __init__(self, vit_name="vit_base_patch16_224", num_classes=None, embed_dim=512, pretrained=True):
		super().__init__()
		config = ViTArcFaceConfig(
			vit_name=vit_name,
			pretrained=pretrained,
			embed_dim=embed_dim,
			num_classes=num_classes
		)
		self.model = ViTArcFaceModel(config)
	
	def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None):
		return self.model(x, labels)
	
	def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
		return self.model.get_embedding(x)
	
	def forward_features(self, x: torch.Tensor) -> torch.Tensor:
		return self.model.forward_features(x)
