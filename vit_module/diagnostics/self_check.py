"""Synthetic verification of diagnostics; these outputs are not research results."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from analyze import donor_indices, replay_head, validate_splits


def main():
    rng = np.random.default_rng(31)
    count, dimension, projected = 192, 16, 12
    labels = np.tile([0, 1], count // 2)
    domain = np.array(["ffpp"] * 128 + ["target"] * 64)
    split = np.array(["train"] * 64 + ["test"] * 128)
    groups = np.array([f"video{index // 8}" for index in range(count)])
    features = rng.normal(size=(count, dimension)).astype(np.float32)
    features[:, 0] += 3 * (2 * labels - 1)
    semantic = rng.normal(size=(count, 8)).astype(np.float32)
    semantic[:, 0] += 2 * (2 * labels - 1)
    head = {
        "weight": rng.normal(size=(2, projected * 2 + 128)).astype(np.float32),
        "bias": rng.normal(size=2).astype(np.float32),
        "proj_weight": rng.normal(size=(projected, dimension)).astype(np.float32),
        "proj_bias": rng.normal(size=projected).astype(np.float32),
        "norm_weight": np.ones(projected, np.float32),
        "norm_bias": np.zeros(projected, np.float32),
        "norm_eps": np.array(1e-5), "fake_logit": np.array(1),
    }
    data = {
        "V": features, "C": rng.normal(size=(count, 20)).astype(np.float32), "S": semantic,
        "F": rng.normal(size=(count, projected * 2 + 128)).astype(np.float32),
        "path": np.array([f"synthetic/{index}.jpg" for index in range(count)]),
        "domain": domain, "split": split, "video_id": groups, "row_id": np.arange(count),
        "y": labels,
    }
    train = validate_splits(data, "ffpp")
    assert train.sum() == 64
    leaked = {**data, "video_id": groups.copy()}
    leaked["video_id"][64] = groups[0]
    try:
        validate_splits(leaked, "ffpp")
    except ValueError:
        pass
    else:
        raise AssertionError("Video leakage was not rejected")
    donors = donor_indices(groups[:64], rng)
    assert np.all(groups[:64][donors] != groups[:64])
    score = replay_head(data, head)
    data["logits"] = np.stack([-score / 2, score / 2], axis=1)
    script = Path(__file__).with_name("analyze.py")
    with tempfile.TemporaryDirectory(prefix="deepfake-diagnostic-check-") as temporary:
        root = Path(temporary)
        extracted = root / "extracted"
        extracted.mkdir()
        (extracted / "config.json").write_text(json.dumps({"arguments": {"variants": ["clean", "blur"]}}), encoding="utf-8")
        (extracted / "COMPLETE").write_text("synthetic", encoding="utf-8")
        np.savez_compressed(extracted / "head.npz", **head)
        for variant in ("clean", "blur"):
            changed = {**data}
            if variant != "clean":
                changed["V"] = features + rng.normal(scale=0.1, size=features.shape)
                changed["S"] = semantic + rng.normal(scale=0.1, size=semantic.shape)
                variant_score = replay_head(changed, head)
                changed["logits"] = np.stack([-variant_score / 2, variant_score / 2], axis=1)
            for start in range(0, count, 32):
                shard = {key: value[start:start + 32] for key, value in changed.items()}
                np.savez_compressed(extracted / f"{variant}_{start:08d}.npz", **shard)
        arguments = [sys.executable, str(script), "--input", str(extracted), "--repeats", "2",
                     "--bootstrap", "20", "--radii", "0.5"]
        first = root / "first"
        subprocess.run(arguments + ["--output", str(first)], check=True)
        metadata = json.loads((first / "config.json").read_text(encoding="utf-8"))
        assert metadata["replay_max_abs_error"] < 1e-4
        for intervention in metadata["interventions"]:
            assert np.isclose(intervention["l2_min"], intervention["l2_max"], atol=1e-8)
        with np.load(first / "sample_scores.npz") as archive:
            assert archive["axis_r0.5_rep0_V_S"].shape == (count,)
            assert np.all(groups[archive["donors_rep0"]] != groups)
        with np.load(first / "directions_rep0.npz") as directions, np.load(first / "source_axis.npz") as axis:
            assert np.max(np.abs(directions["orthogonal"] @ axis["axis"])) < 1e-10
            assert np.max(np.abs(directions["pc1"] @ axis["axis"])) < 1e-10
            source_axis = axis["axis"].copy()
        for file in extracted.glob("clean_*.npz"):
            with np.load(file) as archive:
                shard = {key: archive[key] for key in archive.files}
            target = shard["domain"] == "target"
            shard["V"][target] *= 2
            new_score = replay_head(shard, head)
            shard["logits"] = np.stack([-new_score / 2, new_score / 2], axis=1)
            np.savez_compressed(file, **shard)
        second = root / "second"
        subprocess.run(arguments + ["--output", str(second)], check=True)
        with np.load(second / "source_axis.npz") as axis:
            np.testing.assert_array_equal(axis["axis"], source_axis)
        for name in ("V", "S", "C", "V_S", "V_C"):
            with np.load(first / f"reader_{name}.npz") as before, np.load(second / f"reader_{name}.npz") as after:
                for key in before.files:
                    np.testing.assert_array_equal(before[key], after[key])
    print("PASS: head replay, matched energy, orthogonality, donor isolation, video leakage rejection, source-only fitting")


if __name__ == "__main__":
    main()
