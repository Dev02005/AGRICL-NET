"""Train and evaluate the binary lesion segmenter on PlantSeg.

Unseen diseases (UNSEEN_DISEASES) are excluded from training and validation
and evaluated separately: the project's key experiment.
Resumable: last.pt is written every epoch, so a Colab disconnect only loses
the current epoch. Checkpoints hold only trainable weights plus buffers; the
frozen Hiera encoder comes from the SAM2 checkpoint.
"""

from __future__ import annotations

import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from .data.lesion_dataset import LesionDataset, preprocess
from .data.plantseg import load_image, load_lesion_mask, load_records, split_seen_unseen
from .metrics import SegMeter
from .models.sam2unet import LesionSAM2UNet, structure_loss


@dataclass
class TrainConfig:
    data_root: str  # folder containing Metadata.csv
    out_dir: str  # checkpoints + results (put it on Google Drive)
    repo_dir: str = ""  # SAM2-UNet clone
    size: str = "l"  # Hiera size: t, s, b+, l
    checkpoint: str | None = None  # sam2_hiera_*.pt
    img_size: int = 352  # multiple of 32
    batch_size: int = 4
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 5e-4
    workers: int = 2
    amp: bool = True
    limit: int = 0  # >0: tiny random subset, for a smoke test
    seed: int = 0


def build_splits(cfg: TrainConfig) -> dict:
    seen, unseen = split_seen_unseen(load_records(cfg.data_root))
    splits = {
        "train": [r for r in seen if r.split == "train"],
        "val": [r for r in seen if r.split == "val"],
        "test_seen": [r for r in seen if r.split == "test"],
        "unseen": unseen,  # every split: these diseases are never trained on
    }
    if cfg.limit:
        rng = random.Random(cfg.seed)
        splits = {k: rng.sample(v, min(len(v), cfg.limit)) for k, v in splits.items()}
    return splits


def _as_tuple(out):
    return (out,) if isinstance(out, torch.Tensor) else tuple(out)


def _trainable_state(model: torch.nn.Module) -> dict:
    frozen = {n for n, p in model.named_parameters() if not p.requires_grad}
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if k not in frozen}


def _load_trainable(model: torch.nn.Module, state: dict) -> None:
    missing, unexpected = model.load_state_dict(state, strict=False)
    frozen = {n for n, p in model.named_parameters() if not p.requires_grad}
    bad = [k for k in missing if k not in frozen]
    if bad or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing {bad[:5]}, unexpected {unexpected[:5]}")


@torch.no_grad()
def predict_mask(model, img: Image.Image, img_size: int, device: str, out_hw=None, thresh: float = 0.5) -> np.ndarray:
    """Binary lesion mask (uint8) at out_hw (default: the image's own size)."""
    h, w = out_hw or (img.height, img.width)
    x = preprocess(img, img_size).to(device)
    with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
        logits = _as_tuple(model(x))[0]
    logits = F.interpolate(logits.float(), size=(h, w), mode="bilinear", align_corners=False)
    return (torch.sigmoid(logits)[0, 0] > thresh).cpu().numpy().astype(np.uint8)


@torch.no_grad()
def evaluate(model, records, img_size: int, device: str, full_res: bool = True):
    """Returns (summary dict, per-image rows). full_res compares at the original
    mask size (final results); otherwise at img_size (faster, for validation)."""
    model.eval()
    meter = SegMeter()
    rows = []
    for r in records:
        img = load_image(r)
        gt = load_lesion_mask(r)
        if not full_res:
            gt = np.array(Image.fromarray(gt).resize((img_size, img_size), Image.NEAREST))
        pred = predict_mask(model, img, img_size, device, out_hw=gt.shape)
        meter.update(pred, gt)
        rows.append({
            "image": r.image.name, "disease": r.disease, "plant": r.plant, "split": r.split,
            "iou": meter.ious[-1], "dice": meter.dices[-1],
            "true_share": meter.true_share[-1], "pred_share": meter.pred_share[-1],
        })
    return meter.summary(), rows


def train(cfg: TrainConfig, model: torch.nn.Module | None = None) -> torch.nn.Module:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    splits = build_splits(cfg)
    print({k: len(v) for k, v in splits.items()})

    model = model or LesionSAM2UNet(cfg.repo_dir, cfg.size, cfg.checkpoint)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable parameters: {sum(p.numel() for p in params) / 1e6:.2f} M")
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.epochs, eta_min=1e-7)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device == "cuda")

    start, best, history = 0, -1.0, []
    last = out / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=True)
        _load_trainable(model, ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start, best, history = ck["epoch"], ck["best"], ck["history"]
        print(f"resumed after epoch {start}, best val dice {best:.4f}")

    loader = DataLoader(
        LesionDataset(splits["train"], cfg.img_size, train=True),
        batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.workers,
        pin_memory=device == "cuda", drop_last=True,
    )

    for epoch in range(start, cfg.epochs):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for i, batch in enumerate(loader):
            x = batch["image"].to(device, non_blocking=True)
            y = batch["mask"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=scaler.is_enabled()):
                outs = _as_tuple(model(x))
            loss = sum(structure_loss(o.float(), y) for o in outs)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item()
            n += 1
            if i % 100 == 0:
                print(f"  epoch {epoch + 1} step {i}/{len(loader)} loss {loss.item():.4f}")
        sched.step()

        val, _ = evaluate(model, splits["val"], cfg.img_size, device, full_res=False)
        rec = {"epoch": epoch + 1, "train_loss": total / max(n, 1), "minutes": (time.time() - t0) / 60,
               **{f"val_{k}": v for k, v in val.items()}}
        history.append(rec)
        print(json.dumps(rec))

        if val["mean_dice"] > best:
            best = val["mean_dice"]
            torch.save(_trainable_state(model), out / "best.pt")
            print(f"  new best val dice {best:.4f}")
        torch.save({"model": _trainable_state(model), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch + 1, "best": best, "history": history}, last)
        (out / "history.json").write_text(json.dumps(history, indent=2))
    return model


def final_report(cfg: TrainConfig, model: torch.nn.Module | None = None) -> dict:
    """Load best.pt, evaluate at full resolution on seen-test and unseen diseases."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(cfg.out_dir)
    model = model or LesionSAM2UNet(cfg.repo_dir, cfg.size, cfg.checkpoint)
    _load_trainable(model, torch.load(out / "best.pt", map_location="cpu", weights_only=True))
    model.to(device)
    splits = build_splits(cfg)
    results = {}
    for name in ("test_seen", "unseen"):
        summary, rows = evaluate(model, splits[name], cfg.img_size, device, full_res=True)
        results[name] = summary
        with open(out / f"{name}_per_image.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(name, json.dumps(summary))
    (out / "results.json").write_text(json.dumps(results, indent=2))
    return results
