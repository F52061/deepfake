"""Synthetic G29 checks, not empirical evidence about CLIP or deepfakes."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from vector_fusion import VectorFusion, clip_controls, file_digest, predict, standardize


def main():
    torch.set_num_threads(1)
    rng = np.random.default_rng(71)
    count, source_count, visual_dim, clip_dim = 120, 80, 8, 12
    labels = np.arange(count) % 2
    arrays = {"row_id": np.arange(count), "y": labels.copy(),
              "path": np.array([f"synthetic/{index}.png" for index in range(count)]),
              "domain": np.array(["source"] * 80 + ["target_a"] * 20 + ["target_b"] * 20),
              "split": np.array(["train"] * 80 + ["test"] * 40),
              "video_id": np.array([f"video_{index // 4}" for index in range(count)]),
              "V": rng.normal(size=(count, visual_dim)).astype(np.float32),
              "C": rng.normal(size=(count, clip_dim)).astype(np.float32)}
    arrays["V"][:, 0] += 2 * (2 * labels - 1)
    arrays["C"][:, 0] += (2 * labels - 1)
    rows = np.arange(source_count)
    features, scalers = standardize(arrays, rows)
    fold_fit = np.arange(40)
    fold_features, fold_scalers = standardize(arrays, fold_fit)
    modified_validation = {name: value.copy() for name, value in arrays.items()}
    modified_validation["V"][40:80] += 100
    modified_validation["C"][40:80] -= 100
    modified_features, modified_scalers = standardize(modified_validation, fold_fit)
    for name in fold_scalers:
        np.testing.assert_array_equal(fold_scalers[name], modified_scalers[name])
    for name in ("V", "C"):
        np.testing.assert_array_equal(fold_features[name][fold_fit], modified_features[name][fold_fit])
    fixed, learned, visual = (VectorFusion(visual_dim, clip_dim, name, 7)
                              for name in ("FIXED", "LEARNED", "V_MATCHED"))
    for key, value in fixed.state_dict().items():
        torch.testing.assert_close(value, learned.state_dict()[key], rtol=0, atol=0)
    for key in visual.head.state_dict():
        torch.testing.assert_close(visual.head.state_dict()[key], fixed.head.state_dict()[key], rtol=0, atol=0)
    assert sum(parameter.numel() for parameter in learned.parameters()) == sum(
        parameter.numel() for parameter in fixed.parameters()) + 1
    batch = tuple(torch.as_tensor(features[name][:9]) for name in ("V", "C"))
    score, combined, visual_value, clip_value = learned(*batch)
    torch.testing.assert_close(combined, 0.5 * clip_value + 0.5 * visual_value)
    torch.testing.assert_close(score, fixed(*batch)[0], rtol=0, atol=0)
    assert combined.shape == (9, visual_dim)
    assert float(learned.alpha()) == 0.5
    score.sum().backward()
    assert learned.alpha_logit.grad is not None and torch.isfinite(learned.alpha_logit.grad)
    assert learned.clip_projection.weight.grad is not None
    assert not list(learned.visual_norm.parameters()) and not list(learned.clip_norm.parameters())
    clip_off = predict(learned, features, alpha_override=0)[0]
    np.testing.assert_array_equal(clip_off, predict(visual, features)[0])
    normal, diagnostic = predict(learned, features)
    np.testing.assert_allclose(normal, diagnostic["clip_head_component"] + diagnostic["visual_head_component"]
                               + float(learned.head.bias.detach()[0]), atol=1e-6)
    donor, noise, parameters = clip_controls(features, arrays, rows, 123)
    usable = parameters["usable"]
    positions = parameters["donors"][usable]
    assert np.all(arrays["video_id"][positions] != arrays["video_id"][usable])
    assert np.all(arrays["domain"][positions] == arrays["domain"][usable])
    assert np.all(arrays["split"][positions] == arrays["split"][usable])
    np.testing.assert_array_equal(donor["V"], features["V"])
    np.testing.assert_array_equal(donor["C"][usable], features["C"][positions])
    shorter = {name: value[:100] for name, value in arrays.items()}
    shorter_features, shorter_scalers = standardize(shorter, rows)
    shorter_donor, shorter_noise, shorter_parameters = clip_controls(shorter_features, shorter, rows, 123)
    np.testing.assert_array_equal(noise["C"][:80], shorter_noise["C"][:80])
    np.testing.assert_array_equal(donor["C"][:80], shorter_donor["C"][:80])
    with tempfile.TemporaryDirectory(prefix="deepfake-g29-check-") as temporary:
        directory = Path(temporary)
        root = directory / "input"
        root.mkdir()
        shard = root / "clean_00000000.npz"
        np.savez_compressed(shard, **arrays)
        (root / "config.json").write_text(json.dumps({"label": "1=fake, 0=real", "arguments": {}}), encoding="utf-8")
        (root / "COMPLETE").write_text("ok\n", encoding="utf-8")
        command = [sys.executable, str(Path(__file__).with_name("vector_fusion.py")),
                   "--input", str(root), "--source", "source", "--primary-domains", "target_a", "target_b",
                   "--seeds", "7", "8", "--folds", "2", "--epochs", "3", "--trace-every", "1",
                   "--weight-decays", "0.001", "0.01", "--bootstrap", "12", "--repeats", "2"]
        first_output, changed_output = directory / "first", directory / "changed"
        subprocess.run(command + ["--output", str(first_output)], check=True)
        arrays["y"][source_count:] = 1 - arrays["y"][source_count:]
        arrays["V"][source_count:] += 10
        arrays["C"][source_count:] -= 10
        np.savez_compressed(shard, **arrays)
        subprocess.run(command + ["--output", str(changed_output)], check=True)
        first_config = json.loads((first_output / "config.json").read_text())
        changed_config = json.loads((changed_output / "config.json").read_text())
        assert first_config["selection"] == changed_config["selection"]
        assert first_config["alpha_values"] == changed_config["alpha_values"]
        assert (first_output / "training_trace.json").read_bytes() == (changed_output / "training_trace.json").read_bytes()
        for path in first_output.glob("*scalers.npz"):
            with np.load(path) as first, np.load(changed_output / path.name) as changed:
                for name in first.files:
                    np.testing.assert_array_equal(first[name], changed[name])
        with np.load(first_output / "scores.npz") as first, np.load(changed_output / "scores.npz") as changed:
            for name in first.files:
                if name not in arrays:
                    np.testing.assert_array_equal(first[name][:source_count], changed[name][:source_count])
        with np.load(first_output / "controls.npz") as first, np.load(changed_output / "controls.npz") as changed:
            for name in first.files:
                np.testing.assert_array_equal(first[name], changed[name])
        for outer in json.loads((first_output / "split.json").read_text()):
            fit_ids, validation_ids = outer["fit_row_id"], outer["validation_row_id"]
            assert not set(fit_ids) & set(validation_ids)
            assert not set(arrays["video_id"][fit_ids]) & set(arrays["video_id"][validation_ids])
            assert all(identifier < source_count for identifier in fit_ids + validation_ids)
        report = json.loads((first_output / "summary.json").read_text())
        for key in ("LEARNED|7_vs_V_MATCHED|7", "LEARNED|7_vs_FIXED|7", "LEARNED|7_vs_CONCAT|7",
                    "GRID|LEARNED|7|h1_vs_GRID|FIXED|7|h1", "LEARNED|7_vs_NOISE_RETRAIN|LEARNED|7|1",
                    "FIXED|7_vs_DONOR|FIXED|7|1"):
            assert key in report["primary_macro"]
        assert report["primary_questions"]["clip_increment"] == ["LEARNED|7_vs_V_MATCHED|7", "LEARNED|8_vs_V_MATCHED|8"]
        for name in ("V_MATCHED", "CONCAT", "FIXED", "LEARNED"):
            expected = [np.mean([report["single_seed_auc"][domain][name]["values"][index]
                                for domain in ("target_a", "target_b")]) for index in range(2)]
            np.testing.assert_allclose(report["primary_single_seed_auc"][name]["values"], expected)
        for filename, expected in first_config["code_sha256"].items():
            assert file_digest(first_output / "code_snapshot" / filename) == expected
        for filename, expected in json.loads((first_output / "artifact_hashes.json").read_text()).items():
            assert file_digest(first_output / filename) == expected
        assert (first_output / "COMPLETE").exists()
        rejected = subprocess.run(command + ["--output", str(first_output)], capture_output=True)
        assert rejected.returncode != 0
    print("PASS: matched heads, scalar fusion, gate gradients, source isolation, content controls, grid ablation, hashes, overwrite rejection")


if __name__ == "__main__":
    main()
