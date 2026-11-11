"""Distil the edge students from the accurate teachers, and export them to ONNX.

Classifier: MobileNetV3-Large learns from the labels and from the teacher's
softened outputs (Swin-T from notebook 03, or the continual model from 04).
Lesions: LinkNet-ResNet18 learns from the true masks and from SAM2-UNet's
probability maps, which are computed once and cached as 8-bit PNGs.
"""

from __future__ import annotations

import inspect
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from .calibration import auroc, fit_temperature, macro_f1, softmax, threshold_at_tpr, topk_accuracy
from .cl.model import IncrementalNet, kd_loss
from .data.cls_dataset import CropDataset
from .data.lesion_dataset import LesionDataset, preprocess
from .data.plantseg import load_image, load_records, resized_copy, split_seen_unseen
from .models.classifier import MOBILENET_V3, SWIN_T, build_classifier
from .models.linknet import LinkNet18
from .models.sam2unet import structure_loss
from .train_classifier import _sampler, collect_logits
from .train_lesion import evaluate as evaluate_lesion


@dataclass
class EdgeConfig:
    data_root: str
    out_dir: str
    cache_dir: str | None = None  # resized image copies on local disk
    # teachers
    sam2unet_repo: str = ""
    sam2_size: str = "l"
    sam2_checkpoint: str = ""
    lesion_teacher: str = ""  # notebook 01 best.pt
    lesion_size: int = 352
    classifier_teacher_kind: str = "timm"  # "timm" (notebook 03) or "incremental" (notebook 04)
    classifier_teacher_weights: str = ""
    classifier_teacher_classes: str = ""  # JSON list of class names in output order
    teacher_model_name: str = SWIN_T
    # students
    student_classifier: str = MOBILENET_V3
    cls_size: int = 224
    cls_epochs: int = 30
    cls_lr: float = 5e-4
    kd_temperature: float = 4.0
    kd_alpha: float = 0.7  # weight of the teacher's soft targets
    seg_epochs: int = 40
    seg_lr: float = 1e-3
    seg_kd_weight: float = 1.0
    seg_batch: int = 16
    batch_size: int = 64
    weight_decay: float = 0.01
    workers: int = 2
    amp: bool = True
    limit: int = 0  # >0: tiny subset per split, for a smoke test
    seed: int = 0
    unknown_tpr: float = 0.95


def _seed(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def splits(cfg: EdgeConfig) -> dict:
    seen, unseen = split_seen_unseen(load_records(cfg.data_root))
    out = {"train": [r for r in seen if r.split == "train"], "val": [r for r in seen if r.split == "val"],
           "test_seen": [r for r in seen if r.split == "test"], "unseen": unseen}
    if cfg.limit:
        rng = random.Random(cfg.seed)
        out = {k: rng.sample(v, min(len(v), cfg.limit)) for k, v in out.items()}
    if cfg.cache_dir:
        out = {k: resized_copy(v, cfg.cache_dir) for k, v in out.items()}
    return out


# ------------------------------------------------------------ classifier
def load_teacher_classifier(cfg: EdgeConfig):
    classes = json.loads(Path(cfg.classifier_teacher_classes).read_text())
    if cfg.classifier_teacher_kind == "incremental":
        model = IncrementalNet(len(classes), cfg.teacher_model_name, pretrained=False)
    else:
        model = build_classifier(len(classes), cfg.teacher_model_name, pretrained=False)
    model.load_state_dict(torch.load(cfg.classifier_teacher_weights, map_location="cpu", weights_only=True))
    return model.eval(), classes


def distill_classifier(cfg: EdgeConfig) -> dict:
    _seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(cfg.out_dir) / "classifier"
    out.mkdir(parents=True, exist_ok=True)
    data = splits(cfg)
    teacher, classes = load_teacher_classifier(cfg)
    teacher.to(device)
    for p in teacher.parameters():
        p.requires_grad = False
    c2i = {d: i for i, d in enumerate(classes)}
    (out / "classes.json").write_text(json.dumps(classes, indent=1))

    student = build_classifier(len(classes), cfg.student_classifier).to(device)
    ds = CropDataset(data["train"], c2i, cfg.cls_size, train=True)
    dl = DataLoader(ds, batch_size=cfg.batch_size, sampler=_sampler(ds.labels(), 0.5, cfg.seed),
                    num_workers=cfg.workers, pin_memory=device == "cuda", drop_last=len(ds) > cfg.batch_size)
    opt = torch.optim.AdamW(student.parameters(), lr=cfg.cls_lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(cfg.cls_epochs * len(dl), 1))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device == "cuda")

    start, best, history = 0, -1.0, []
    last = out / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=True)
        student.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start, best, history = ck["epoch"], ck["best"], ck["history"]
        print(f"resumed after epoch {start}")

    for epoch in range(start, cfg.cls_epochs):
        student.train()
        t0, total, n = time.time(), 0.0, 0
        for b in dl:
            x, y = b["image"].to(device, non_blocking=True), b["label"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=scaler.is_enabled()):
                with torch.no_grad():
                    t_logits = teacher(x).float()
                s_logits = student(x).float()
                loss = ((1 - cfg.kd_alpha) * F.cross_entropy(s_logits, y, label_smoothing=0.1)
                        + cfg.kd_alpha * kd_loss(s_logits, t_logits, cfg.kd_temperature))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
            n += 1
        lv, yv = collect_logits(student, data["val"], c2i, cfg_view(cfg), device)
        rec = {"epoch": epoch + 1, "loss": total / max(n, 1), "val_top1": topk_accuracy(lv, yv, 1),
               "minutes": (time.time() - t0) / 60}
        history.append(rec)
        print(json.dumps(rec))
        if rec["val_top1"] > best:
            best = rec["val_top1"]
            torch.save(student.state_dict(), out / "best.pt")
        torch.save({"model": student.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": epoch + 1, "best": best, "history": history}, last)

    student.load_state_dict(torch.load(out / "best.pt", map_location="cpu", weights_only=True))
    results = {"student": classifier_report(student, data, c2i, cfg, device, out)}
    results["teacher"] = classifier_report(teacher, data, c2i, cfg, device, None)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    return results


def cfg_view(cfg: EdgeConfig) -> SimpleNamespace:
    """collect_logits needs img_size, batch_size and workers."""
    return SimpleNamespace(img_size=cfg.cls_size, batch_size=cfg.batch_size, workers=cfg.workers)


def classifier_report(model, data, c2i, cfg, device, out: Path | None) -> dict:
    v = cfg_view(cfg)
    lv, yv = collect_logits(model, data["val"], c2i, v, device)
    lt, yt = collect_logits(model, data["test_seen"], c2i, v, device)
    lu, _ = collect_logits(model, data["unseen"], c2i, v, device)
    t = fit_temperature(lv, yv)
    thr = threshold_at_tpr(softmax(lv, t).max(1), cfg.unknown_tpr)
    ct, cu = softmax(lt, t).max(1), softmax(lu, t).max(1)
    if out is not None:
        (out / "calibration.json").write_text(json.dumps({"temperature": t, "unknown_threshold": thr}, indent=2))
    return {"top1": topk_accuracy(lt, yt, 1), "top3": topk_accuracy(lt, yt, 3),
            "macro_f1": macro_f1(lt.argmax(1), yt, len(c2i)), "temperature": t, "unknown_threshold": thr,
            "unseen_flagged_unknown": float((cu < thr).mean()), "auroc_known_vs_unseen": auroc(ct, cu),
            "parameters_m": sum(p.numel() for p in model.parameters()) / 1e6}


# --------------------------------------------------------------- lesions
def cache_teacher_masks(cfg: EdgeConfig, records) -> str:
    """SAM2-UNet probability maps at lesion_size, saved as 8-bit PNGs. Skips existing files."""
    from .pipeline import load_lesion_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    soft = Path(cfg.out_dir) / "teacher_masks"
    todo = [r for r in records if not (soft / r.split / r.mask.name).exists()]
    if todo:
        teacher = load_lesion_model(cfg.sam2unet_repo, cfg.sam2_size, cfg.sam2_checkpoint, cfg.lesion_teacher, device)
        for i, r in enumerate(todo):
            x = preprocess(load_image(r), cfg.lesion_size).to(device)
            with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
                logits = teacher(x)[0].float()
            prob = (torch.sigmoid(logits)[0, 0].cpu().numpy() * 255).round().astype(np.uint8)
            (soft / r.split).mkdir(parents=True, exist_ok=True)
            Image.fromarray(prob).save(soft / r.split / r.mask.name)
            if i % 500 == 0:
                print(f"  teacher masks {i}/{len(todo)}")
        del teacher
        torch.cuda.empty_cache()
    return str(soft)


def distill_lesion(cfg: EdgeConfig) -> dict:
    _seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(cfg.out_dir) / "lesion"
    out.mkdir(parents=True, exist_ok=True)
    data = splits(cfg)
    soft = cache_teacher_masks(cfg, data["train"])

    student = LinkNet18(pretrained=True).to(device)
    dl = DataLoader(LesionDataset(data["train"], cfg.lesion_size, train=True, soft_dir=soft),
                    batch_size=cfg.seg_batch, shuffle=True, num_workers=cfg.workers,
                    pin_memory=device == "cuda", drop_last=len(data["train"]) > cfg.seg_batch)
    opt = torch.optim.AdamW(student.parameters(), lr=cfg.seg_lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(cfg.seg_epochs, 1), eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device == "cuda")

    start, best, history = 0, -1.0, []
    last = out / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=True)
        student.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start, best, history = ck["epoch"], ck["best"], ck["history"]
        print(f"resumed after epoch {start}")

    for epoch in range(start, cfg.seg_epochs):
        student.train()
        t0, total, n = time.time(), 0.0, 0
        for b in dl:
            x, y, s = (b[k].to(device, non_blocking=True) for k in ("image", "mask", "soft"))
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=scaler.is_enabled()):
                logits = student(x).float()
            loss = structure_loss(logits, y) + cfg.seg_kd_weight * F.binary_cross_entropy_with_logits(logits, s)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item()
            n += 1
        sched.step()
        val, _ = evaluate_lesion(student, data["val"], cfg.lesion_size, device, full_res=False)
        rec = {"epoch": epoch + 1, "loss": total / max(n, 1), "val_mean_dice": val["mean_dice"],
               "minutes": (time.time() - t0) / 60}
        history.append(rec)
        print(json.dumps(rec))
        if val["mean_dice"] > best:
            best = val["mean_dice"]
            torch.save(student.state_dict(), out / "best.pt")
        torch.save({"model": student.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": epoch + 1, "best": best, "history": history}, last)

    student.load_state_dict(torch.load(out / "best.pt", map_location="cpu", weights_only=True))
    results = {name: evaluate_lesion(student, data[name], cfg.lesion_size, device, full_res=True)[0]
               for name in ("test_seen", "unseen")}
    results["parameters_m"] = sum(p.numel() for p in student.parameters()) / 1e6
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    return results


# ---------------------------------------------------------------- export
def export_onnx(model: torch.nn.Module, path: str | Path, size: int, opset: int = 17) -> Path:
    """FP32 ONNX with a dynamic batch axis. Uses the TorchScript exporter
    (dynamo=False) where torch offers the choice, for dynamic_axes support."""
    model = model.eval().cpu()
    kwargs = {"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}
    torch.onnx.export(model, torch.zeros(1, 3, size, size), str(path), input_names=["input"],
                      output_names=["output"], dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
                      opset_version=opset, **kwargs)
    return Path(path)
