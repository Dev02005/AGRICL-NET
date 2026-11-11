"""Train and evaluate the disease classifier on seen PlantSeg diseases.

Trained on all seen diseases at once: this is the main classifier and also
the "joint training" upper bound for the continual-learning experiments.
Unseen diseases are never trained on; they test unknown-disease detection.
After training: temperature scaling and an unknown threshold fitted on the
validation set, then metrics on the seen test split and the unseen diseases.
"""

from __future__ import annotations

import csv
import json
import math
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .calibration import (
    auroc, ece, fit_temperature, macro_f1, softmax, threshold_at_tpr, topk_accuracy,
)
from .data.cls_dataset import CropDataset, build_class_index
from .data.plantseg import load_records, split_seen_unseen
from .models.classifier import SWIN_T, build_classifier


@dataclass
class ClsConfig:
    data_root: str
    out_dir: str
    model_name: str = SWIN_T
    img_size: int = 224
    batch_size: int = 32
    epochs: int = 30
    lr: float = 1e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 1
    label_smoothing: float = 0.1
    balance_power: float = 0.5  # sample weight = count^-power (0: none, 1: fully balanced)
    workers: int = 2
    amp: bool = True
    limit: int = 0  # >0: tiny random subset, for a smoke test
    seed: int = 0
    unknown_tpr: float = 0.95  # keep 95% of known-disease images as known


def build_splits(cfg: ClsConfig) -> dict:
    seen, unseen = split_seen_unseen(load_records(cfg.data_root))
    splits = {
        "train": [r for r in seen if r.split == "train"],
        "val": [r for r in seen if r.split == "val"],
        "test_seen": [r for r in seen if r.split == "test"],
        "unseen": unseen,
    }
    if cfg.limit:
        rng = random.Random(cfg.seed)
        splits = {k: rng.sample(v, min(len(v), cfg.limit)) for k, v in splits.items()}
    return splits


def _sampler(labels: np.ndarray, power: float, seed: int) -> WeightedRandomSampler:
    counts = Counter(labels.tolist())
    w = torch.tensor([counts[l] ** -power for l in labels], dtype=torch.double)
    return WeightedRandomSampler(w, num_samples=len(labels), replacement=True,
                                 generator=torch.Generator().manual_seed(seed))


@torch.no_grad()
def collect_logits(model, records, class_to_idx, cfg: ClsConfig, device: str):
    model.eval()
    ds = CropDataset(records, class_to_idx, cfg.img_size, train=False)
    dl = DataLoader(ds, batch_size=cfg.batch_size * 2, shuffle=False, num_workers=cfg.workers)
    out = []
    for b in dl:
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
            out.append(model(b["image"].to(device)).float().cpu())
    return torch.cat(out).numpy(), ds.labels()


def train(cfg: ClsConfig, model=None):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    splits = build_splits(cfg)
    classes = build_class_index(splits["train"] + splits["val"] + splits["test_seen"])
    (out / "classes.json").write_text(json.dumps(sorted(classes, key=classes.get), indent=1))
    print({k: len(v) for k, v in splits.items()}, "classes:", len(classes))

    model = model or build_classifier(len(classes), cfg.model_name)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    train_ds = CropDataset(splits["train"], classes, cfg.img_size, train=True)
    loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=_sampler(train_ds.labels(), cfg.balance_power, cfg.seed),
                        num_workers=cfg.workers, pin_memory=device == "cuda", drop_last=True)
    steps = cfg.epochs * len(loader)
    warm = cfg.warmup_epochs * len(loader)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(warm, 1) if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(steps - warm, 1))))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device == "cuda")
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    start, best, history = 0, -1.0, []
    last = out / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=True)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start, best, history = ck["epoch"], ck["best"], ck["history"]
        print(f"resumed after epoch {start}, best val top-1 {best:.4f}")

    for epoch in range(start, cfg.epochs):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for i, b in enumerate(loader):
            x, y = b["image"].to(device, non_blocking=True), b["label"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=scaler.is_enabled()):
                loss = loss_fn(model(x).float(), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
            n += 1
            if i % 100 == 0:
                print(f"  epoch {epoch + 1} step {i}/{len(loader)} loss {loss.item():.4f}")

        logits, labels = collect_logits(model, splits["val"], classes, cfg, device)
        rec = {"epoch": epoch + 1, "train_loss": total / max(n, 1), "minutes": (time.time() - t0) / 60,
               "val_top1": topk_accuracy(logits, labels, 1), "val_top3": topk_accuracy(logits, labels, 3),
               "val_macro_f1": macro_f1(logits.argmax(1), labels, len(classes))}
        history.append(rec)
        print(json.dumps(rec))
        if rec["val_top1"] > best:
            best = rec["val_top1"]
            torch.save(model.state_dict(), out / "best.pt")
            print(f"  new best val top-1 {best:.4f}")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch + 1, "best": best, "history": history}, last)
        (out / "history.json").write_text(json.dumps(history, indent=2))
    return model


def final_report(cfg: ClsConfig, model=None) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(cfg.out_dir)
    names = json.loads((out / "classes.json").read_text())
    classes = {d: i for i, d in enumerate(names)}
    model = model or build_classifier(len(classes), cfg.model_name, pretrained=False)
    model.load_state_dict(torch.load(out / "best.pt", map_location="cpu", weights_only=True))
    model.to(device)
    splits = build_splits(cfg)

    lv, yv = collect_logits(model, splits["val"], classes, cfg, device)
    lt, yt = collect_logits(model, splits["test_seen"], classes, cfg, device)
    lu, _ = collect_logits(model, splits["unseen"], classes, cfg, device)
    np.savez_compressed(out / "logits.npz", val=lv, val_y=yv, test=lt, test_y=yt, unseen=lu)

    t = fit_temperature(lv, yv)
    conf_val = softmax(lv, t).max(1)
    thr = threshold_at_tpr(conf_val, cfg.unknown_tpr)
    pt, pu = softmax(lt, t), softmax(lu, t)
    results = {
        "classes": len(classes),
        "temperature": t,
        "unknown_threshold": thr,
        "test_seen": {
            "images": len(yt),
            "top1": topk_accuracy(lt, yt, 1),
            "top3": topk_accuracy(lt, yt, 3),
            "macro_f1": macro_f1(lt.argmax(1), yt, len(classes)),
            "ece_before": ece(softmax(lt), yt),
            "ece_after": ece(pt, yt),
            "flagged_unknown": float((pt.max(1) < thr).mean()),
        },
        "unseen": {
            "images": len(lu),
            "flagged_unknown": float((pu.max(1) < thr).mean()),
            "auroc_known_vs_unseen": auroc(pt.max(1), pu.max(1)),
        },
    }
    (out / "calibration.json").write_text(json.dumps({"temperature": t, "unknown_threshold": thr}, indent=2))
    (out / "results.json").write_text(json.dumps(results, indent=2))

    pred = lt.argmax(1)
    with open(out / "per_class.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["disease", "test_images", "accuracy"])
        for c, name in enumerate(names):
            m = yt == c
            if m.any():
                w.writerow([name, int(m.sum()), float((pred[m] == c).mean())])
    print(json.dumps(results, indent=2))
    return results
