import torch
import torch.nn.functional as F


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
	na = F.normalize(a, dim=1)
	nb = F.normalize(b, dim=1)
	return torch.mm(na, nb.t())
