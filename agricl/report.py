"""Collect every run's results into one Markdown report for the documentation.

Reads the run folders the notebooks write (MyDrive/AGRI-NET/runs/...), skips
smoke tests, and says plainly which steps have not been run yet. Pure Python,
so it is tested locally with fake runs.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _runs(runs_dir: Path, prefix: str) -> list[Path]:
    return sorted(p for p in runs_dir.glob(f"{prefix}*") if p.is_dir() and not p.name.endswith("_smoke"))


def pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def num(x, digits: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


def md_table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def collect(runs_dir: str | Path) -> dict:
    """{section: {run name: loaded results}} for every finished (non-smoke) run."""
    d = Path(runs_dir)
    out = {"lesion": {}, "severity": {}, "classifier": {}, "continual": {}, "edge": {}}
    for p in _runs(d, "lesion_"):
        if (r := _load(p / "results.json")) is not None:
            out["lesion"][p.name] = r
    for p in _runs(d, "severity_"):
        if (r := _load(p / "severity_summary.json")) is not None:
            out["severity"][p.name] = r
    for p in _runs(d, "classifier_"):
        if (r := _load(p / "results.json")) is not None:
            out["classifier"][p.name] = r
    for p in _runs(d, "continual_"):
        if (r := _load(p / "results.json")) is not None:
            out["continual"][p.name] = r
    for p in _runs(d, "edge_"):
        r = {"classifier": _load(p / "classifier" / "results.json"), "lesion": _load(p / "lesion" / "results.json")}
        bench = p / "edge_benchmark.csv"
        r["benchmark"] = list(csv.DictReader(open(bench, encoding="utf-8"))) if bench.exists() else None
        if any(v is not None for v in r.values()):
            out["edge"][p.name] = r
    return out


NOT_RUN = "_Not run yet._"


def section_lesion(res: dict) -> str:
    if not res:
        return NOT_RUN
    rows = []
    for name, r in res.items():
        for group in ("test_seen", "unseen"):
            g = r.get(group, {})
            rows.append([name, "seen diseases (test)" if group == "test_seen" else "unseen diseases",
                         g.get("images", "n/a"), pct(g.get("mean_dice")), pct(g.get("mean_iou")),
                         num(g.get("share_r2"))])
    return md_table(["Run", "Images", "Count", "Mean Dice", "Mean IoU", "Lesion-share R²"], rows)


def section_severity(res: dict) -> str:
    if not res:
        return NOT_RUN
    rows = []
    for name, r in res.items():
        for group in ("seen", "unseen"):
            g = r.get(group, {})
            rows.append([name, group, g.get("images", "n/a"), num(g.get("severity_r2")),
                         num(g.get("severity_rmse"), 2), pct(g.get("level_accuracy"))])
    return md_table(["Run", "Diseases", "Images", "Severity R²", "RMSE (points)", "Level agreement"], rows)


def section_classifier(res: dict) -> str:
    if not res:
        return NOT_RUN
    rows = []
    for name, r in res.items():
        t, u = r.get("test_seen", {}), r.get("unseen", {})
        rows.append([name, r.get("classes", "n/a"), pct(t.get("top1")), pct(t.get("top3")), pct(t.get("macro_f1")),
                     num(t.get("ece_before")), num(t.get("ece_after")), pct(u.get("flagged_unknown")),
                     num(u.get("auroc_known_vs_unseen"))])
    return md_table(["Run", "Classes", "Top-1", "Top-3", "Macro-F1", "ECE before", "ECE after",
                     "Unseen flagged unknown", "AUROC known vs unseen"], rows)


def section_continual(res: dict) -> str:
    if not res:
        return NOT_RUN
    rows = []
    for name, methods in res.items():
        for m, r in methods.items():
            rows.append([name, m, pct(r.get("average_incremental_accuracy")), pct(r.get("final_accuracy")),
                         pct(r.get("forgetting")), pct(r.get("calibration", {}).get("unseen_flagged_unknown"))])
    return md_table(["Run", "Method", "Avg. incremental accuracy", "Final accuracy", "Forgetting",
                     "Unseen flagged unknown"], rows)


def section_edge(res: dict) -> str:
    if not res:
        return NOT_RUN
    parts = []
    for name, r in res.items():
        parts.append(f"**{name}**")
        c = r.get("classifier") or {}
        if c:
            parts.append(md_table(
                ["Classifier", "Top-1", "Top-3", "Parameters (M)", "Unseen flagged unknown"],
                [[k, pct(v.get("top1")), pct(v.get("top3")), num(v.get("parameters_m"), 1),
                  pct(v.get("unseen_flagged_unknown"))] for k, v in c.items()]))
        les = r.get("lesion") or {}
        if les:
            parts.append(md_table(
                ["Lesion student", "Mean Dice", "Mean IoU"],
                [[g, pct(les.get(g, {}).get("mean_dice")), pct(les.get(g, {}).get("mean_iou"))]
                 for g in ("test_seen", "unseen")]))
        if r.get("benchmark"):
            parts.append(md_table(
                ["Model", "Precision", "Size (MB)", "ms, 1 thread", "ms, all threads", "Accuracy", "Metric"],
                [[b["model"], b["precision"], b["size_mb"], b["ms_1_thread"], b["ms_all_threads"],
                  b["accuracy"], b["metric"]] for b in r["benchmark"]]))
    return "\n\n".join(parts)


def key_claim(res: dict) -> str:
    """One paragraph on the project's claim: severity works on unseen diseases."""
    lines = []
    for name, r in res["lesion"].items():
        s, u = r.get("test_seen", {}).get("mean_dice"), r.get("unseen", {}).get("mean_dice")
        if s is not None and u is not None:
            lines.append(f"- Lesion model ({name}): mean Dice {pct(s)} on seen diseases vs {pct(u)} on unseen "
                         f"diseases (gap {100 * (s - u):+.1f} points).")
    for name, r in res["severity"].items():
        s, u = r.get("seen", {}).get("severity_r2"), r.get("unseen", {}).get("severity_r2")
        if s is not None and u is not None:
            lines.append(f"- Severity ({name}): R² {num(s)} on seen vs {num(u)} on unseen diseases.")
    return "\n".join(lines) if lines else NOT_RUN


def build_report(runs_dir: str | Path) -> str:
    res = collect(runs_dir)
    return "\n\n".join([
        "# AgriCL-Net results",
        "Generated from the run folders by `agricl.report`. Smoke-test runs are excluded.",
        "## Key claim: severity on diseases never seen in training", key_claim(res),
        "## Lesion segmentation (SAM2-UNet)", section_lesion(res["lesion"]),
        "## Severity: true vs. predicted lesions (same leaf mask)", section_severity(res["severity"]),
        "## Disease classifier (Swin-T, joint training)", section_classifier(res["classifier"]),
        "## Continual learning (5 rounds by crop)", section_continual(res["continual"]),
        "## Edge models (distilled, ONNX, INT8)", section_edge(res["edge"]),
    ]) + "\n"
