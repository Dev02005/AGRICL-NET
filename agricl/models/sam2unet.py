"""SAM2-UNet lesion segmenter (binary: lesion vs. background).

Built on github.com/WZH0120/SAM2-UNet (cloned at run time). Differences from
the repo's own SAM2UNet class:
- any Hiera size (t, s, b+, l), channel widths read from the SAM2 config
  instead of being hard-coded for Large;
- built on CPU first, so it does not require CUDA to construct;
- structure_loss uses reduction="none". The repo passes the deprecated
  reduce="none", which PyTorch reads as "reduce = True" (mean), so its
  edge-weighted BCE silently becomes a plain mean BCE.
Checkpoints are the original SAM2 weights (not SAM2.1), as the repo requires.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

CONFIGS = {"t": "sam2_hiera_t.yaml", "s": "sam2_hiera_s.yaml", "b+": "sam2_hiera_b+.yaml", "l": "sam2_hiera_l.yaml"}
CHECKPOINTS = {"t": "sam2_hiera_tiny.pt", "s": "sam2_hiera_small.pt", "b+": "sam2_hiera_base_plus.pt", "l": "sam2_hiera_large.pt"}
CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/"


def _import_repo(repo_dir: str | Path):
    repo = str(Path(repo_dir).resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from sam2.build_sam import build_sam2  # noqa: E402  (also initialises hydra configs)
    from SAM2UNet import RFB_modified, Adapter, Up  # noqa: E402

    return build_sam2, Adapter, RFB_modified, Up


class LesionSAM2UNet(nn.Module):
    def __init__(self, repo_dir: str | Path, size: str = "l", checkpoint: str | None = None) -> None:
        super().__init__()
        build_sam2, Adapter, RFB_modified, Up = _import_repo(repo_dir)
        sam = build_sam2(CONFIGS[size], checkpoint, device="cpu")
        # neck lists channels deepest first, e.g. [1152, 576, 288, 144] for Large
        c1, c2, c3, c4 = list(sam.image_encoder.neck.backbone_channel_list)[::-1]
        self.encoder = sam.image_encoder.trunk
        del sam  # drop mask decoder, prompt encoder, memory modules, neck

        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.blocks = nn.Sequential(*[Adapter(b) for b in self.encoder.blocks])

        self.rfb1, self.rfb2 = RFB_modified(c1, 64), RFB_modified(c2, 64)
        self.rfb3, self.rfb4 = RFB_modified(c3, 64), RFB_modified(c4, 64)
        self.up1, self.up2, self.up3 = Up(128, 64), Up(128, 64), Up(128, 64)
        self.side1 = nn.Conv2d(64, 1, kernel_size=1)
        self.side2 = nn.Conv2d(64, 1, kernel_size=1)
        self.head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)
        x1, x2, x3, x4 = self.rfb1(x1), self.rfb2(x2), self.rfb3(x3), self.rfb4(x4)
        x = self.up1(x4, x3)
        out1 = F.interpolate(self.side1(x), scale_factor=16, mode="bilinear")
        x = self.up2(x, x2)
        out2 = F.interpolate(self.side2(x), scale_factor=8, mode="bilinear")
        x = self.up3(x, x1)
        out = F.interpolate(self.head(x), scale_factor=4, mode="bilinear")
        return out, out1, out2


def structure_loss(pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Edge-weighted BCE + weighted IoU loss (as in SAM2-UNet / PraNet), fixed."""
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction="none")
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    prob = torch.sigmoid(pred)
    inter = ((prob * mask) * weit).sum(dim=(2, 3))
    union = ((prob + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()
