import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcMarginProduct(nn.Module):
	"""
	Implement of ArcFace: Additive Angular Margin Loss
	Parameters:
		in_features: size of each input sample (embedding dim)
		out_features: number of classes
		s: norm scaling factor
		m: angular margin in radians
	"""

	def __init__(self, in_features: int, out_features: int, s: float = 64.0, m: float = 0.5):
		super().__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.s = s
		self.m = m

		self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
		nn.init.xavier_uniform_(self.weight)

		self.cos_m = math.cos(m)
		self.sin_m = math.sin(m)
		self.th = math.cos(math.pi - m)
		self.mm = math.sin(math.pi - m) * m

	def forward(self, input_features: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
		# Normalize features and weights
		normalized_input = F.normalize(input_features)
		normalized_weight = F.normalize(self.weight)

		# Cosine similarity between features and class centers
		cosine = F.linear(normalized_input, normalized_weight).clamp(-1.0, 1.0)

		if labels is None:
			# Inference mode: return scaled cosine logits
			return cosine * self.s

		# Compute sine and phi for target classes
		sine = torch.sqrt((1.0 - cosine * cosine).clamp(0, 1))
		phi = cosine * self.cos_m - sine * self.sin_m

		# Decide whether to use phi or cosine - mm to ensure monotonicity
		phi_adjusted = torch.where(cosine > self.th, phi, cosine - self.mm)

		# One-hot labels for margin application
		one_hot = torch.zeros_like(cosine)
		one_hot.scatter_(1, labels.view(-1, 1), 1.0)

		# Only add margin to the target class
		logits = (one_hot * phi_adjusted) + ((1.0 - one_hot) * cosine)
		logits = logits * self.s
		return logits
