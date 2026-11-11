import json

from agricl.report import build_report, collect, md_table, pct


def write(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj))


def fake_runs(root):
    write(root / "lesion_sam2unet_l_352" / "results.json",
          {"test_seen": {"images": 1400, "mean_dice": 0.62, "mean_iou": 0.5, "share_r2": 0.8},
           "unseen": {"images": 387, "mean_dice": 0.58, "mean_iou": 0.46, "share_r2": 0.75}})
    write(root / "lesion_sam2unet_l_352_smoke" / "results.json", {"test_seen": {"mean_dice": 0.01}})
    write(root / "severity_lesion_sam2unet_l_352_sam2_leaf" / "severity_summary.json",
          {"seen": {"images": 150, "severity_r2": 0.85, "severity_rmse": 6.2, "level_accuracy": 0.8},
           "unseen": {"images": 150, "severity_r2": 0.81, "severity_rmse": 7.0, "level_accuracy": 0.77}})
    write(root / "continual_swin" / "results.json",
          {"finetune": {"average_incremental_accuracy": 0.5, "final_accuracy": 0.3, "forgetting": 0.6,
                        "calibration": {"unseen_flagged_unknown": 0.4}},
           "replay_kd": {"average_incremental_accuracy": 0.7, "final_accuracy": 0.6, "forgetting": 0.15,
                         "calibration": {"unseen_flagged_unknown": 0.55}}})
    (root / "edge_continual").mkdir(parents=True)
    (root / "edge_continual" / "edge_benchmark.csv").write_text(
        "model,precision,size_mb,ms_1_thread,ms_all_threads,accuracy,metric\n"
        "classifier,int8,4.2,12.0,6.1,0.71,top-1\n", encoding="utf-8")


def test_collect_skips_smoke_and_missing(tmp_path):
    fake_runs(tmp_path)
    res = collect(tmp_path)
    assert list(res["lesion"]) == ["lesion_sam2unet_l_352"]
    assert res["classifier"] == {}
    assert res["edge"]["edge_continual"]["benchmark"][0]["precision"] == "int8"


def test_report_content(tmp_path):
    fake_runs(tmp_path)
    md = build_report(tmp_path)
    assert "62.0% on seen diseases vs 58.0% on unseen diseases (gap +4.0 points)" in md
    assert "R² 0.850 on seen vs 0.810 on unseen" in md
    assert "| continual_swin | replay_kd | 70.0% | 60.0% | 15.0% | 55.0% |" in md
    assert "## Disease classifier (Swin-T, joint training)\n\n_Not run yet._" in md
    assert "smoke" not in md


def test_empty_runs_folder(tmp_path):
    md = build_report(tmp_path)
    assert md.count("_Not run yet._") == 6


def test_helpers():
    assert pct(None) == "n/a" and pct(0.1234) == "12.3%"
    assert md_table(["a", "b"], [[1, 2]]) == "| a | b |\n| --- | --- |\n| 1 | 2 |"
