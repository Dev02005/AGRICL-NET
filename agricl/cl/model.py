"""Expandable classifier and losses for class-incremental learning."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.classifier import SWIN_T


class IncrementalNet(nn.Module):
    """timm backbone (no head) + a bias-free linear head that grows each round.
    No bias keeps weight aligning (rescaling new-class rows) well defined."""

    def __init__(self, n_classes: int, name: str = SWIN_T, pretrained: bool = True, drop_path: float = 0.1) -> None:
        super().__init__()
        import timm

        self.backbone = timm.create_model(name, pretrained=pretrained, num_classes=0, drop_path_rate=drop_path)
        self.dim = self.backbone.num_features
        self.head = nn.Linear(self.dim, n_classes, bias=False)

    @property
    def n_classes(self) -> int:
        return self.head.out_features

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    def expand(self, n_new: int) -> None:
        old = self.head
        new = nn.Linear(self.dim, old.out_features + n_new, bias=False).to(old.weight.device, old.weight.dtype)
        with torch.no_grad():
            new.weight[: old.out_features] = old.weight
        self.head = new

    @torch.no_grad()
    def weight_align(self, n_old: int) -> float:
        """Weight aligning (Zhao et al., CVPR 2020): scale new-class weight rows so
        their mean norm matches the old classes', removing the bias toward new
        classes that comes from having far more new-class images. Returns the scale."""
        w = self.head.weight
        gamma = (w[:n_old].norm(dim=1).mean() / w[n_old:].norm(dim=1).mean()).item()
        w[n_old:] *= gamma
        return gamma


def kd_loss(new_logits: torch.Tensor, old_logits: torch.Tensor, temperature: float = 2.0) -> torch.Tensor:
    """Distillation on the old classes: KL(old || new) on softened outputs, x T^2
    (Hinton et al., 2015) so its gradient scale does not depend on T."""
    n_old = old_logits.shape[1]
    log_p = F.log_softmax(new_logits[:, :n_old] / temperature, dim=1)
    q = F.softmax(old_logits / temperature, dim=1)
    return F.kl_div(log_p, q, reduction="batchmean") * temperature**2
