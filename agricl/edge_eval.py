"""INT8 quantization, speed benchmark and accuracy checks for the ONNX students.

Needs only ONNX Runtime + NumPy (no PyTorch), so the same code can run on a
Raspberry Pi. Models are any callable (1,3,S,S) float32 -> array, so tests can
pass fakes.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PIL import Image

from .calibration import topk_accuracy
from .data.plantseg import load_image, load_lesion_mask
from .edge import lesion_from_logits, to_input
from .metrics import SegMeter
from .roi import roi_box


def lesion_inputs(records, size: int) -> list[np.ndarray]:
    return [to_input(load_image(r), size) for r in records]


def classifier_inputs(records, size: int) -> list[np.ndarray]:
    """Lesion-centred crops from the true masks, as during training."""
    return [to_input(load_image(r).crop(roi_box(load_lesion_mask(r))), size, Image.BICUBIC) for r in records]


def quantize_int8(fp32_path: str | Path, int8_path: str | Path, calibration: list[np.ndarray]) -> Path:
    """Static INT8 quantization (QDQ format, per-channel weights) calibrated on real inputs."""
    from onnxruntime.quantization import (
        CalibrationDataReader, QuantFormat, QuantType, quant_pre_process, quantize_static,
    )

    class Reader(CalibrationDataReader):
        def __init__(self, arrays):
            self.it = iter(arrays)

        def get_next(self):
            x = next(self.it, None)
            return None if x is None else {"input": x}

    pre = Path(fp32_path).with_suffix(".pre.onnx")
    quant_pre_process(str(fp32_path), str(pre))
    quantize_static(str(pre), str(int8_path), Reader(calibration), quant_format=QuantFormat.QDQ,
                    per_channel=True, activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
    pre.unlink(missing_ok=True)
    return Path(int8_path)


def latency_ms(model, x: np.ndarray, runs: int = 30, warmup: int = 5) -> float:
    """Median milliseconds per call for one input."""
    for _ in range(warmup):
        model(x)
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        model(x)
        times.append((time.perf_counter() - t) * 1000)
    return float(np.median(times))


def classifier_top1(model, inputs: list[np.ndarray], labels: np.ndarray) -> float:
    logits = np.concatenate([np.asarray(model(x), np.float32).reshape(1, -1) for x in inputs])
    return topk_accuracy(logits, np.asarray(labels), 1)


def lesion_scores(model, records, inputs: list[np.ndarray]) -> dict:
    meter = SegMeter()
    for r, x in zip(records, inputs):
        gt = load_lesion_mask(r)
        meter.update(lesion_from_logits(np.asarray(model(x)), gt.shape), gt)
    return meter.summary()


def file_mb(path: str | Path) -> float:
    return Path(path).stat().st_size / 2**20
