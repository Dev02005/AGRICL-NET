"""Disease classifiers from timm.

Main model: Swin-T pretrained on ImageNet-22k then ImageNet-1k
(timm/swin_tiny_patch4_window7_224.ms_in22k_ft_in1k, fixed 224 px input).
Edge fallback: MobileNetV3-Large (timm/mobilenetv3_large_100.ra_in1k).
"""

from __future__ import annotations

SWIN_T = "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k"
MOBILENET_V3 = "mobilenetv3_large_100.ra_in1k"


def build_classifier(num_classes: int, name: str = SWIN_T, pretrained: bool = True, drop_path: float = 0.1):
    import timm

    return timm.create_model(name, pretrained=pretrained, num_classes=num_classes, drop_path_rate=drop_path)
