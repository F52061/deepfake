"""Synthetic checks for conditional_clip.py. Not empirical evidence."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from conditional_clip import RegionReader, split_donors


def main():
    rng = np.random.default_rng(11)
    count, regions, v_dim, c_dim = 240, 9, 16, 24
    videos, domains, splits = [], [], []
    for index in range(count):
        train = index < 160
        videos.append(f"v{index // 4}" if train else f"t{index // 4}")
        domains.append("source" if train else f"target{index % 3}")
        splits.append("train" if train else "test")
    videos, domains, splits = map(np.array, (videos, domains, splits))

    # Deterministic labels: every source video carries two reals and two fakes, so
    # the both-classes-per-video precondition is exercised rather than dodged.
    labels = np.array([1 if (index % 4) >= 2 else 0 for index in range(count)])
    signed = np.where(labels == 1, 1.0, -1.0)
    signal = signed + rng.normal(size=count) * 0.3
    visual = rng.normal(size=(count, v_dim)).astype(np.float32)
    pooled = rng.normal(size=(count, c_dim)).astype(np.float32)
    region = rng.normal(size=(count, regions, c_dim)).astype(np.float32)
    region[:, :, 0] += signed[:, None] * 2.0
    visual[:, 0] += signed * 2.0
    fake_logit = signal * 3.0
    logits = np.stack([fake_logit, -fake_logit], axis=1).astype(np.float32)

    with tempfile.TemporaryDirectory(prefix="deepfake-conditional-check-") as temporary:
        directory = Path(temporary)
        shards = directory / "input"
        shards.mkdir()
        out = directory / "output"
        np.savez_compressed(
            shards / "clean_00000000.npz", row_id=np.arange(count), y=labels,
            path=np.array([f"synthetic/{index}.png" for index in range(count)]),
            domain=domains, split=splits, video_id=videos, V=visual, C=pooled,
            clip_regions=region, logits=logits, S=np.zeros((count, 1), np.float32))
        (shards / "config.json").write_text(
            json.dumps({"arguments": {"save_regions": True}, "label": "1=fake, 0=real"}),
            encoding="utf-8")
        (shards / "COMPLETE").write_text("ok\n", encoding="utf-8")

        subprocess.run([sys.executable, str(Path(__file__).with_name("conditional_clip.py")),
                        "--input", str(shards), "--output", str(out), "--source", "source",
                        "--seeds", "1", "2", "--folds", "2", "--epochs", "30",
                        "--bootstrap", "40", "--repeats", "2"], check=True)

        config = json.loads((out / "config.json").read_text(encoding="utf-8"))
        checks = config["checks"]
        assert checks["s_v_auc_source_train"] > 0.9, "s_V must point at the fake class"
        assert checks["donor_same_video_violations"] == 0
        assert checks["donor_cell_violations"] == 0
        assert checks["train_videos_with_both_classes"] == checks["train_videos"]

        with np.load(out / "scores.npz") as scores, np.load(out / "controls.npz") as controls:
            # The off control (delta = 0) must reproduce s_V exactly.
            expected = (logits[:, 0] - logits[:, 1]).astype(np.float32)
            np.testing.assert_array_equal(scores["A"], expected)
            donors = controls["donors"]
            usable = controls["donor_usable"]
            assert np.all(videos[donors[usable]] != videos[usable])
            assert np.all(domains[donors[usable]] == domains[usable])
            assert np.all(splits[donors[usable]] == splits[usable])
        assert (out / "COMPLETE").exists()

    # F must be permutation invariant, which is why the spec forbids using a
    # region-position shuffle as a control for it.
    torch.manual_seed(0)
    reader = RegionReader(v_dim, c_dim, 8, 16)
    v = torch.as_tensor(visual[:5])
    c = torch.as_tensor(pooled[:5])
    ci = torch.as_tensor(region[:5])
    s = torch.as_tensor(np.zeros(5, np.float32))
    with torch.no_grad():
        first, _ = reader(v, c, ci, s)
        second, _ = reader(v, c, ci[:, torch.randperm(regions)], s)
    np.testing.assert_allclose(first.numpy(), second.numpy(), atol=1e-6)

    # Donor selection must refuse cells with a single video instead of borrowing
    # from the same video.
    donors = split_donors(np.array(["a"] * 4), np.array(["d"] * 4), np.array(["train"] * 4),
                          np.random.default_rng(0))
    assert np.all(donors == -1)
    print("PASS: s_V orientation, off control, donor rules, both-class videos, F permutation invariance")


if __name__ == "__main__":
    main()
