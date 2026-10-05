"""Synthetic G28 checks; no empirical deepfake claims."""

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from clip_readout import controlled_stores, evaluation_subset, intervention_prediction
from conditional_clip import Store, parameter_count
from pure_vit_clip import build_reader


def main():
    torch.set_num_threads(1)
    rng = np.random.default_rng(51)
    count, source_count, visual_dim, clip_dim, region_count = 120, 80, 8, 12, 4
    labels = np.arange(count) % 2
    arrays = {"row_id": np.arange(count), "y": labels.copy(),
              "path": np.array([f"synthetic/{index}.png" for index in range(count)]),
              "domain": np.array(["source"] * source_count + ["target_a"] * 20 + ["target_b"] * 20),
              "split": np.array(["train"] * source_count + ["test"] * 40),
              "video_id": np.array([f"video_{index // 4}" for index in range(count)]),
              "V": rng.normal(size=(count, visual_dim)).astype(np.float32),
              "C": rng.normal(size=(count, clip_dim)).astype(np.float32),
              "clip_regions": rng.normal(size=(count, region_count, clip_dim)).astype(np.float32),
              "logits": rng.normal(size=(count, 2)).astype(np.float32)}
    arrays["V"][:, 0] += 2 * (2 * labels - 1)
    args = SimpleNamespace(dim=4, width=8)
    models = {name: build_reader(name, visual_dim, clip_dim, args, 7)
              for name in ("POOLED", "MEAN", "ADAPTIVE", "V_ONLY")}
    assert parameter_count(models["V_ONLY"]) >= parameter_count(models["ADAPTIVE"])
    for name in ("MEAN", "POOLED"):
        for key in models["ADAPTIVE"].state_dict():
            torch.testing.assert_close(models["ADAPTIVE"].state_dict()[key], models[name].state_dict()[key])
    store = Store({"v": arrays["V"], "c": arrays["C"], "ci": arrays["clip_regions"],
                   "s": arrays["V"][:, 0].copy()}, region_count, clip_dim, "cpu")
    with torch.no_grad():
        models["ADAPTIVE"].head[-1].weight.fill_(0.1)
        models["MEAN"].load_state_dict(models["ADAPTIVE"].state_dict())
    from pure_vit_clip import predict
    original, original_weights = predict(models["ADAPTIVE"], store)
    uniform, uniform_weights = intervention_prediction(models["ADAPTIVE"], store, uniform=True)
    expected, expected_weights = predict(models["MEAN"], store)
    np.testing.assert_array_equal(uniform, expected)
    np.testing.assert_allclose(uniform_weights, 1 / region_count)
    softened, softened_weights = intervention_prediction(models["ADAPTIVE"], store, temperature=4)
    assert np.all(softened_weights.max(axis=1) <= original_weights.max(axis=1) + 1e-6)
    restored, restored_weights = predict(models["ADAPTIVE"], store)
    np.testing.assert_array_equal(original, restored)
    np.testing.assert_array_equal(original_weights, restored_weights)
    permuted = Store({"v": store.v, "c": store.c, "ci": store.ci[:, [2, 0, 3, 1]], "s": store.s},
                     region_count, clip_dim, "cpu")
    np.testing.assert_allclose(predict(models["ADAPTIVE"], permuted)[0], original, atol=1e-6)
    source_rows = np.arange(source_count)
    donor, noise, noise_training, parameters = controlled_stores(store, store, arrays, source_rows, 7)
    valid = parameters["usable"]
    positions = parameters["donors"][valid]
    assert np.all(arrays["video_id"][positions] != arrays["video_id"][valid])
    assert np.all(arrays["domain"][positions] == arrays["domain"][valid])
    assert np.all(arrays["split"][positions] == arrays["split"][valid])
    np.testing.assert_array_equal(donor.c[valid], store.c[positions])
    np.testing.assert_array_equal(donor.ci[valid], store.ci[positions])
    np.testing.assert_array_equal(donor.v, store.v)
    np.testing.assert_array_equal(donor.s, store.s)
    retained = np.arange(100)
    shorter_data = {key: value[retained] for key, value in arrays.items()}
    shorter_store = Store({"v": store.v[retained], "c": store.c[retained],
                           "ci": store.ci[retained], "s": store.s[retained]}, region_count, clip_dim, "cpu")
    shorter_donor, shorter_noise, shorter_training, shorter_parameters = controlled_stores(
        shorter_store, shorter_store, shorter_data, source_rows, 7)
    np.testing.assert_array_equal(noise.c[:source_count], shorter_noise.c[:source_count])
    np.testing.assert_array_equal(noise.ci[:source_count], shorter_noise.ci[:source_count])
    np.testing.assert_array_equal(donor.c[:source_count], shorter_donor.c[:source_count])
    with tempfile.TemporaryDirectory(prefix="deepfake-g28-check-") as temporary:
        directory = Path(temporary)
        root = directory / "input"
        root.mkdir()
        shard = root / "clean_00000000.npz"
        np.savez_compressed(shard, **arrays)
        (root / "config.json").write_text(json.dumps({"arguments": {"save_regions": True},
                                                    "label": "1=fake, 0=real"}), encoding="utf-8")
        (root / "COMPLETE").write_text("ok\n", encoding="utf-8")
        command = [sys.executable, str(Path(__file__).with_name("clip_readout.py")),
                   "--input", str(root), "--source", "source", "--primary-domains", "target_a", "target_b",
                   "--seeds", "7", "8", "--folds", "2", "--inner-folds", "2", "--epochs", "3",
                   "--dim", "4", "--width", "8", "--lambdas", "0.01", "0.1",
                   "--weight-decays", "0.001", "--repeats", "2", "--bootstrap", "12"]
        first_output, changed_output = directory / "first", directory / "changed"
        subprocess.run(command + ["--output", str(first_output)], check=True)
        original_logits = arrays["logits"].copy()
        arrays["y"][source_count:] = 1 - arrays["y"][source_count:]
        arrays["V"][source_count:] += 10
        arrays["C"][source_count:] -= 10
        arrays["clip_regions"][source_count:] *= -1
        arrays["logits"] *= -100
        np.savez_compressed(shard, **arrays)
        subprocess.run(command + ["--output", str(changed_output)], check=True)
        first_config = json.loads((first_output / "config.json").read_text())
        changed_config = json.loads((changed_output / "config.json").read_text())
        assert first_config["selection"] == changed_config["selection"]
        assert first_config["calibration"] == changed_config["calibration"]
        assert (first_output / "training_trace.json").read_bytes() == (changed_output / "training_trace.json").read_bytes()
        for filename in first_output.glob("*_scalers.npz"):
            with np.load(filename) as first, np.load(changed_output / filename.name) as changed:
                for key in first.files:
                    np.testing.assert_array_equal(first[key], changed[key])
        with np.load(first_output / "scores.npz") as first, np.load(changed_output / "scores.npz") as changed:
            for key in first.files:
                if key not in ("DETECTOR_REFERENCE",) and key not in arrays:
                    np.testing.assert_array_equal(first[key][:source_count], changed[key][:source_count])
            np.testing.assert_array_equal(first["DETECTOR_REFERENCE"], original_logits[:, 0] - original_logits[:, 1])
            for seed in (7, 8):
                np.testing.assert_array_equal(first[f"GAIN|ADAPTIVE|{seed}|0"], first["V_BASE"])
                np.testing.assert_array_equal(first[f"GAIN|POOLED|{seed}|1"], first[f"SELECTED|POOLED|{seed}"])
        split = json.loads((first_output / "split.json").read_text())
        for outer in split["outer"]:
            fit_ids, validation_ids = set(outer["fit_row_id"]), set(outer["validation_row_id"])
            assert not fit_ids & validation_ids
            assert all(identifier < source_count for identifier in fit_ids | validation_ids)
            for inner in outer["inner"]:
                inner_fit, inner_validation = set(inner["fit_row_id"]), set(inner["validation_row_id"])
                assert not inner_fit & inner_validation
                assert inner_fit | inner_validation == fit_ids
                assert not (inner_fit | inner_validation) & validation_ids
        report = json.loads((first_output / "summary.json").read_text())
        assert report["primary_macro"]["CALIBRATED_vs_V_BASE"]["delta_macro_auc"] == 0
        assert "SELECTED|POOLED|7_vs_NOISE_RETRAIN|POOLED|7|1" in report["primary_macro"]
        assert "GRID|ADAPTIVE|7|h1_vs_GRID|MEAN|7|h1" in report["primary_macro"]
        assert "UNIFORM|7_vs_SELECTED|ADAPTIVE|7" in report["primary_macro"]
        assert report["primary_questions"]["content_increment"] == [
            "SELECTED|POOLED|7_vs_V_BASE", "SELECTED|POOLED|8_vs_V_BASE"]
        expected_macro = [np.mean([report["single_seed_auc"][domain]["POOLED"]["values"][index]
                                   for domain in ("target_a", "target_b")]) for index in range(2)]
        np.testing.assert_allclose(report["primary_single_seed_auc"]["POOLED"]["values"], expected_macro)
        for filename, expected_hash in first_config["code_sha256"].items():
            assert hashlib.sha256((first_output / "code_snapshot" / filename).read_bytes()).hexdigest() == expected_hash
        for filename, expected_hash in json.loads((first_output / "artifact_hashes.json").read_text()).items():
            assert hashlib.sha256((first_output / filename).read_bytes()).hexdigest() == expected_hash
        assert (first_output / "COMPLETE").exists()
        manifest = directory / "evaluation.csv"
        manifest.write_text("row_id\n" + "\n".join(str(index) for index in range(80, 100)), encoding="utf-8")
        subset = evaluation_subset(arrays, "source", manifest)
        assert len(subset["y"]) == 100
        manifest.write_text("row_id\n80\n", encoding="utf-8")
        try:
            evaluation_subset(arrays, "source", manifest)
        except ValueError:
            pass
        else:
            raise AssertionError("Partial-video evaluation accepted")
        manifest.write_text("row_id\n0\n", encoding="utf-8")
        try:
            evaluation_subset(arrays, "source", manifest)
        except ValueError:
            pass
        else:
            raise AssertionError("Training rows accepted as new evaluation")
    print("PASS: nested OOF isolation, content controls, shared grid, frozen interventions, target isolation, artifact hashes")


if __name__ == "__main__":
    main()
