"""LinkNet with a ResNet-18 encoder (Chaurasia and Culurciello, 2017).

The lightweight lesion student for edge devices. It is the baseline network
of base paper 1 (PDSNets, about 11.5 M parameters). Output: one lesion logit
map at the input resolution (input side must be a multiple of 32).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def _cbr(cin: int, cout: int, k: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, k, padding=k // 2, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class DecoderBlock(nn.Module):
    """1x1 reduce -> 3x3 transposed conv (x2 upsampling) -> 1x1 expand."""

    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        mid = cin // 4
        self.reduce = _cbr(cin, mid, 1)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(mid, mid, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True))
        self.expand = _cbr(mid, cout, 1)

    def forward(self, x):
        return self.expand(self.up(self.reduce(x)))


class LinkNet18(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        r = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)  # /4
        self.enc1, self.enc2, self.enc3, self.enc4 = r.layer1, r.layer2, r.layer3, r.layer4  # 64,128,256,512
        self.dec4, self.dec3 = DecoderBlock(512, 256), DecoderBlock(256, 128)
        self.dec2, self.dec1 = DecoderBlock(128, 64), DecoderBlock(64, 64)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            _cbr(32, 32, 3),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, x):
        e1 = self.enc1(self.stem(x))  # /4
        e2 = self.enc2(e1)  # /8
        e3 = self.enc3(e2)  # /16
        e4 = self.enc4(e3)  # /32
        d = self.dec4(e4) + e3
        d = self.dec3(d) + e2
        d = self.dec2(d) + e1
        d = self.dec1(d)  # /2
        return self.final(d)  # /1


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    m = LinkNet18(pretrained=False)
    print(f"{count_parameters(m) / 1e6:.2f} M parameters", tuple(m(torch.zeros(1, 3, 352, 352)).shape))
