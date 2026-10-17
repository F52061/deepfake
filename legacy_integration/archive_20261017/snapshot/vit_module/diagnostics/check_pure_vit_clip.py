"""Synthetic G27 checks; not empirical evidence about deepfake detection."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from pure_vit_clip import RegionCorrection, build_reader, calibrate, macro_comparison
from conditional_clip import parameter_count


def main():
    rng = np.random.default_rng(43)
    count, source_rows, regions, visual_dim, clip_dim = 120, 80, 4, 8, 12
    labels = np.arange(count) % 2
    domains = np.array(["source"] * 80 + ["target_a"] * 20 + ["target_b"] * 20)
    splits = np.array(["train"] * 80 + ["test"] * 40)
    videos = np.array([f"video_{index // 4}" for index in range(count)])
    visual = rng.normal(size=(count, visual_dim)).astype(np.float32)
    visual[:, 0] += 2 * (2 * labels - 1)
    pooled = rng.normal(size=(count, clip_dim)).astype(np.float32)
    tokens = rng.normal(size=(count, regions, clip_dim)).astype(np.float32)
    logits = rng.normal(size=(count, 2)).astype(np.float32)
    args = SimpleNamespace(dim=4, width=8)
    mean_reader = build_reader("MEAN", visual_dim, clip_dim, args, 7)
    adaptive = build_reader("ADAPTIVE", visual_dim, clip_dim, args, 7)
    assert parameter_count(mean_reader) == parameter_count(adaptive)
    assert parameter_count(build_reader("V_ONLY", visual_dim, clip_dim, args, 7)) >= parameter_count(adaptive)
    for name in adaptive.state_dict():
        torch.testing.assert_close(adaptive.state_dict()[name], mean_reader.state_dict()[name])
    with torch.no_grad():
        adaptive.head[-1].weight.fill_(0.1)
        batch = tuple(torch.as_tensor(value) for value in
                      (visual[:6], pooled[:6], tokens[:6], np.zeros(6, dtype=np.float32)))
        first, weights = adaptive(*batch)
        permuted = list(batch)
        permuted[2] = batch[2][:, [2, 0, 3, 1]]
        second, weights = adaptive(*permuted)
        torch.testing.assert_close(first, second)
    baseline = visual[:, 0].astype(np.float64)
    scale, offset = calibrate(baseline[:source_rows], labels[:source_rows])
    data = {"domain": domains, "split": splits, "video_id": videos, "y": labels}
    comparison = macro_comparison(data, baseline, scale * baseline + offset,
                                  ["target_a", "target_b"], 20, 7)
    assert comparison["delta_macro_auc"] == 0
    assert comparison["ci95"] == [0, 0]
    macro_data = {"domain": np.array(["first"] * 4 + ["second"] * 4),
                  "split": np.array(["test"] * 8), "y": np.array([0, 0, 1, 1] * 2),
                  "video_id": np.array((["video_first"] * 2 + ["video_second"] * 2) * 2)}
    macro_baseline = np.array([-2, -1, 1, 2] * 2, dtype=float)
    shifted = macro_baseline + np.array([0] * 4 + [100] * 4)
    macro_result = macro_comparison(macro_data, macro_baseline, shifted, ["first", "second"], 0, 7)
    assert macro_result["delta_macro_auc"] == 0
    assert roc_auc_score(macro_data["y"], shifted) != roc_auc_score(macro_data["y"], macro_baseline)
    with tempfile.TemporaryDirectory(prefix="deepfake-g27-check-") as temporary:
        directory = Path(temporary)
        root = directory / "input"
        root.mkdir()
        shard = root / "clean_00000000.npz"
        arrays = {"row_id": np.arange(count), "y": labels,
                  "path": np.array([f"synthetic/{index}.png" for index in range(count)]),
                  "domain": domains, "split": splits, "video_id": videos,
                  "V": visual, "C": pooled, "clip_regions": tokens, "logits": logits}
        np.savez_compressed(shard, **arrays)
        (root / "config.json").write_text(json.dumps({"arguments": {"save_regions": True},
                                                    "label": "1=fake, 0=real"}), encoding="utf-8")
        (root / "COMPLETE").write_text("ok\n", encoding="utf-8")
        command = [sys.executable, str(Path(__file__).with_name("pure_vit_clip.py")),
                   "--input", str(root), "--source", "source", "--primary-domains", "target_a", "target_b",
                   "--seeds", "7", "8", "--folds", "2", "--epochs", "3", "--dim", "4", "--width", "8",
                   "--lambdas", "0.01", "--weight-decays", "0.001", "--repeats", "2", "--bootstrap", "12"]
        first_output = directory / "first"
        subprocess.run(command + ["--output", str(first_output)], check=True)
        changed_output = directory / "changed_target"
        arrays["y"] = labels.copy()
        arrays["y"][source_rows:] = 1 - arrays["y"][source_rows:]
        arrays["V"] = visual.copy()
        arrays["V"][source_rows:] += 10
        arrays["C"] = pooled.copy()
        arrays["C"][source_rows:] -= 10
        arrays["logits"] = logits * -100
        np.savez_compressed(shard, **arrays)
        subprocess.run(command + ["--output", str(changed_output)], check=True)
        first_config = json.loads((first_output / "config.json").read_text())
        changed_config = json.loads((changed_output / "config.json").read_text())
        assert first_config["selection"] == changed_config["selection"]
        assert first_config["calibration"] == changed_config["calibration"]
        with np.load(first_output / "scalers.npz") as first, np.load(changed_output / "scalers.npz") as changed:
            for key in first.files:
                np.testing.assert_array_equal(first[key], changed[key])
        with np.load(first_output / "scores.npz") as first, np.load(changed_output / "scores.npz") as changed:
            for key in ("V_BASE", "CALIBRATED", "V_ONLY|7", "MEAN|7", "ADAPTIVE|7"):
                np.testing.assert_array_equal(first[key][:source_rows], changed[key][:source_rows])
            np.testing.assert_array_equal(first["DETECTOR_REFERENCE"], logits[:, 0] - logits[:, 1])
        with np.load(first_output / "controls.npz") as controls:
            for seed in (7, 8):
                for repeat in range(2):
                    donors = controls[f"donors|{seed}|{repeat}"]
                    usable = donors >= 0
                    assert np.all(videos[donors[usable]] != videos[usable])
                    assert np.all(domains[donors[usable]] == domains[usable])
                    assert np.all(splits[donors[usable]] == splits[usable])
        report = json.loads((first_output / "summary.json").read_text())
        assert "ADAPTIVE|7_vs_MEAN|7" in report["primary_macro"]
        assert "ADAPTIVE|ensemble_vs_MEAN|ensemble" in report["primary_macro"]
        assert report["primary_macro"]["CALIBRATED_vs_V_BASE"]["delta_macro_auc"] == 0
        assert (first_output / "COMPLETE").exists()
    print("PASS: pure V baseline, fit-fold isolation, matched pooling, calibration, macro bootstrap, donors")


if __name__ == "__main__":
    main()
