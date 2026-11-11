"""Class-incremental learning experiment on the seen PlantSeg diseases.

Rounds are groups of crops (make_tasks). All methods share one base model
trained on round 0, then add rounds 1..T-1:
  finetune   new round's images only                       (lower bound)
  replay     + exemplar buffer, m images per old disease   (herding)
  replay_kd  + knowledge distillation from the previous model
             + weight aligning of the head after each round (AgriCL-Net)
The joint-training upper bound is notebook 03's classifier.
Evaluation is class-incremental: no round id is given, the model picks among
all diseases seen so far. Results are saved after every round, so a Colab
disconnect loses at most one round.
"""

from __future__ import annotations

import copy
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..calibration import auroc, fit_temperature, softmax, threshold_at_tpr
from ..data.cls_dataset import CropDataset
from ..data.plantseg import load_records, resized_copy, split_seen_unseen
from ..models.classifier import SWIN_T
from ..train_classifier import _sampler, collect_logits
from .core import average_incremental_accuracy, forgetting, herding_select, make_tasks
from .model import IncrementalNet, kd_loss

METHODS = ("finetune", "replay", "replay_kd")


@dataclass
class CLConfig:
    data_root: str
    out_dir: str
    cache_dir: str | None = None  # resized image copies (local disk), None = use originals
    cache_max_side: int = 640
    model_name: str = SWIN_T
    img_size: int = 224
    batch_size: int = 32
    n_tasks: int = 5
    base_frac: float = 0.5
    base_epochs: int = 15
    inc_epochs: int = 10
    lr: float = 1e-4
    inc_lr: float = 5e-5
    weight_decay: float = 0.05
    label_smoothing: float = 0.1
    balance_power: float = 0.5
    exemplars_per_class: int = 20
    kd_temperature: float = 2.0
    methods: tuple = field(default_factory=lambda: METHODS)
    workers: int = 2
    amp: bool = True
    limit_per_class: int = 0  # >0: at most this many images per disease and split (smoke test)
    seed: int = 0
    unknown_tpr: float = 0.95


def _seed(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def prepare(cfg: CLConfig) -> dict:
    """Records by split, rounds, and the global class order (round by round)."""
    seen, unseen = split_seen_unseen(load_records(cfg.data_root))
    if cfg.limit_per_class:
        rng, kept, count = random.Random(cfg.seed), [], {}
        for r in rng.sample(seen, len(seen)):
            key = (r.disease, r.split)
            if count.get(key, 0) < cfg.limit_per_class:
                kept.append(r)
                count[key] = count.get(key, 0) + 1
        seen, unseen = kept, rng.sample(unseen, min(len(unseen), 30))
    if cfg.cache_dir:
        seen = resized_copy(seen, cfg.cache_dir, cfg.cache_max_side)
        unseen = resized_copy(unseen, cfg.cache_dir, cfg.cache_max_side)
    tasks, plants = make_tasks(seen, cfg.n_tasks, cfg.base_frac, cfg.seed)
    order = [d for t in tasks for d in t]
    return {
        "train": [r for r in seen if r.split == "train"],
        "val": [r for r in seen if r.split == "val"],
        "test": [r for r in seen if r.split == "test"],
        "unseen": unseen,
        "tasks": tasks, "plants": plants,
        "class_to_idx": {d: i for i, d in enumerate(order)},
    }


def _train_round(model, records, class_to_idx, epochs, lr, cfg, device, old_model=None):
    model.train()
    ds = CropDataset(records, class_to_idx, cfg.img_size, train=True)
    dl = DataLoader(ds, batch_size=cfg.batch_size, sampler=_sampler(ds.labels(), cfg.balance_power, cfg.seed),
                    num_workers=cfg.workers, pin_memory=device == "cuda", drop_last=len(ds) > cfg.batch_size)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)
    steps = max(epochs * len(dl), 1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / steps)))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device == "cuda")
    n_old = old_model.n_classes if old_model is not None else 0
    alpha = n_old / model.n_classes  # weight of distillation grows with the share of old classes
    for epoch in range(epochs):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for b in dl:
            x, y = b["image"].to(device, non_blocking=True), b["label"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=scaler.is_enabled()):
                logits = model(x).float()
                loss = F.cross_entropy(logits, y, label_smoothing=cfg.label_smoothing)
                if old_model is not None:
                    with torch.no_grad():
                        old = old_model(x).float()
                    loss = (1 - alpha) * loss + alpha * kd_loss(logits, old, cfg.kd_temperature)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
            n += 1
        print(f"    epoch {epoch + 1}/{epochs} loss {total / max(n, 1):.4f} ({(time.time() - t0) / 60:.1f} min)")


@torch.no_grad()
def _features(model, records, class_to_idx, cfg, device) -> np.ndarray:
    model.eval()
    dl = DataLoader(CropDataset(records, class_to_idx, cfg.img_size, train=False),
                    batch_size=cfg.batch_size * 2, shuffle=False, num_workers=cfg.workers)
    out = []
    for b in dl:
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
            out.append(model.features(b["image"].to(device)).float().cpu())
    return torch.cat(out).numpy()


def _exemplars(model, train_records, diseases, class_to_idx, cfg, device) -> list:
    picked = []
    for d in diseases:
        recs = [r for r in train_records if r.disease == d]
        if recs:
            idx = herding_select(_features(model, recs, class_to_idx, cfg, device), cfg.exemplars_per_class)
            picked += [recs[i] for i in idx]
    return picked


def _evaluate_round(model, data, t, cfg, device) -> dict:
    """Accuracy on each round 0..t, choosing among all diseases seen so far."""
    seen = {d for task in data["tasks"][: t + 1] for d in task}
    test = [r for r in data["test"] if r.disease in seen]
    logits, y = collect_logits(model, test, data["class_to_idx"], cfg, device)
    pred = logits.argmax(1)
    per_task = []
    for task in data["tasks"][: t + 1]:
        ids = [data["class_to_idx"][d] for d in task]
        m = np.isin(y, ids)
        per_task.append(float((pred[m] == y[m]).mean()) if m.any() else float("nan"))
    return {"all": float((pred == y).mean()), "per_task": per_task}


def _calibration_report(model, data, cfg, device) -> dict:
    lv, yv = collect_logits(model, data["val"], data["class_to_idx"], cfg, device)
    lt, yt = collect_logits(model, data["test"], data["class_to_idx"], cfg, device)
    lu, _ = collect_logits(model, data["unseen"], data["class_to_idx"], cfg, device)
    t = fit_temperature(lv, yv)
    thr = threshold_at_tpr(softmax(lv, t).max(1), cfg.unknown_tpr)
    ct, cu = softmax(lt, t).max(1), softmax(lu, t).max(1)
    return {"temperature": t, "unknown_threshold": thr, "test_flagged_unknown": float((ct < thr).mean()),
            "unseen_flagged_unknown": float((cu < thr).mean()), "auroc_known_vs_unseen": auroc(ct, cu)}


def run(cfg: CLConfig) -> dict:
    _seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    data = prepare(cfg)
    c2i, tasks = data["class_to_idx"], data["tasks"]
    (out / "rounds.json").write_text(json.dumps(
        [{"round": i, "plants": p, "diseases": d} for i, (p, d) in enumerate(zip(data["plants"], tasks))], indent=1))
    print("diseases per round:", [len(t) for t in tasks])
    by_task = lambda recs, t: [r for r in recs if r.disease in set(tasks[t])]  # noqa: E731

    # ---- round 0, shared by every method
    base_path = out / "base.pt"
    model = IncrementalNet(len(tasks[0]), cfg.model_name).to(device)
    if base_path.exists():
        model.load_state_dict(torch.load(base_path, map_location="cpu", weights_only=True))
        print("loaded base model")
    else:
        _seed(cfg.seed)
        print(f"round 0: {len(tasks[0])} diseases, {len(by_task(data['train'], 0))} images")
        _train_round(model, by_task(data["train"], 0), c2i, cfg.base_epochs, cfg.lr, cfg, device)
        torch.save(model.state_dict(), base_path)
    base_eval = _evaluate_round(model, data, 0, cfg, device)
    print("round 0 accuracy:", round(base_eval["all"], 4))

    results = {}
    for method in cfg.methods:
        state_path = out / f"{method}_state.pt"
        if state_path.exists():
            st = torch.load(state_path, map_location="cpu", weights_only=True)
            model = IncrementalNet(st["n_classes"], cfg.model_name, pretrained=False).to(device)
            model.load_state_dict(st["model"])
            done, evals = st["done"], st["evals"]
            names = set(st["buffer"])
            buffer = [r for r in data["train"] if r.image.name in names]
            print(f"{method}: resumed after round {done}")
        else:
            model = IncrementalNet(len(tasks[0]), cfg.model_name, pretrained=False).to(device)
            model.load_state_dict(torch.load(base_path, map_location="cpu", weights_only=True))
            done, evals = 0, [base_eval]
            buffer = (_exemplars(model, data["train"], tasks[0], c2i, cfg, device)
                      if method != "finetune" else [])

        for t in range(done + 1, len(tasks)):
            _seed(cfg.seed + t)
            n_old = model.n_classes
            old_model = copy.deepcopy(model).eval() if method == "replay_kd" else None
            if old_model is not None:
                for p in old_model.parameters():
                    p.requires_grad = False
            model.expand(len(tasks[t]))
            train_recs = by_task(data["train"], t) + buffer
            print(f"{method} round {t}: {len(tasks[t])} new diseases, {len(train_recs)} images "
                  f"({len(buffer)} replayed)")
            _train_round(model, train_recs, c2i, cfg.inc_epochs, cfg.inc_lr, cfg, device, old_model)
            if method == "replay_kd":
                print(f"    weight aligning scale {model.weight_align(n_old):.3f}")
            evals.append(_evaluate_round(model, data, t, cfg, device))
            print(f"    accuracy on all {model.n_classes} diseases so far: {evals[-1]['all']:.4f}")
            if method != "finetune":
                buffer += _exemplars(model, data["train"], tasks[t], c2i, cfg, device)
            torch.save({"model": model.state_dict(), "n_classes": model.n_classes, "done": t, "evals": evals,
                        "buffer": [r.image.name for r in buffer]}, state_path)
            del old_model

        acc = np.full((len(tasks), len(tasks)), np.nan)
        for i, e in enumerate(evals):
            acc[i, : len(e["per_task"])] = e["per_task"]
        results[method] = {
            "step_accuracy": [e["all"] for e in evals],
            "average_incremental_accuracy": average_incremental_accuracy([e["all"] for e in evals]),
            "final_accuracy": evals[-1]["all"],
            "forgetting": forgetting(acc),
            "accuracy_matrix": acc.tolist(),
            "calibration": _calibration_report(model, data, cfg, device),
        }
        torch.save(model.state_dict(), out / f"{method}_final.pt")
        (out / "results.json").write_text(json.dumps(results, indent=2))
        print(method, json.dumps({k: v for k, v in results[method].items() if k != "accuracy_matrix"}, indent=1))
    return results
