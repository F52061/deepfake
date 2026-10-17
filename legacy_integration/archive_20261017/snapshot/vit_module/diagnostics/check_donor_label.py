"""Synthetic G29-A checks, not empirical evidence about CLIP content or deepfakes."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from clip_readout import file_digest, write_json
from donor_label_control import (DONOR_SEED_OFFSET, KINDS, LABEL, VARIANTS,
                                 assert_video_labels_unique, label_donors, substitute, video_key)
from vector_fusion import VectorFusion, predict, standardize

HERE = Path(__file__).resolve().parent


def video_key_checks():
    """Video identity from the path: the sample paths from the real run, verbatim."""
    cases = {
        "F:/zhj/data/FaceForensic++_raw/manipulated_sequences/Deepfakes/c23/faces23/114_102/167.png":
            "F:/zhj/data/FaceForensic++_raw/manipulated_sequences/Deepfakes/c23/114_102",
        "F:/zhj/data/FaceForensic++_raw/original_sequences/c23/faces23/308/401.png":
            "F:/zhj/data/FaceForensic++_raw/original_sequences/c23/308",
        "F:/zhj/data/Celeb-DF-v2/faces23/id10_0007/461.png": "F:/zhj/data/Celeb-DF-v2/id10_0007",
        "F:/zhj/data/Celeb-DF-v1/faces23/00011/208.png": "F:/zhj/data/Celeb-DF-v1/00011",
        "F:/zhj/data/DFDC-P/test_set_23/original_videos/2005778_C_003/175.png":
            "F:/zhj/data/DFDC-P/test_set_23/original_videos/2005778_C_003",
        "F:/zhj/data/WildDeepfake/real_test/79/real/180/3525.png": "F:/zhj/data/WildDeepfake/real_test/79/real",
        "F:/zhj/data/WildDeepfake/fake_test/33/fake/0/9.png": "F:/zhj/data/WildDeepfake/fake_test/33/fake",
        "F:/zhj/data/FaceForensic++_raw/manipulated_sequences/Face2Face/c23/faces23/003_000/157.png":
            "F:/zhj/data/FaceForensic++_raw/manipulated_sequences/Face2Face/c23/003_000",
    }
    for path, expected in cases.items():
        assert video_key(path) == expected, f"{path} -> {video_key(path)}, expected {expected}"
    # Two frames of one video must not split; the real and fake video of one person must not merge.
    assert video_key("F:/d/WildDeepfake/real_test/79/real/180/1.png") == \
        video_key("F:/d/WildDeepfake/real_test/79/real/180/999.png")
    assert video_key("F:/d/Celeb-DF-v2/faces23/id1_0007/1.png") != \
        video_key("F:/d/Celeb-DF-v2/faces23/id1_id0_0007/1.png")
    assert video_key("F:/d/FaceForensic++_raw/manipulated_sequences/Deepfakes/c23/faces23/003_000/1.png") != \
        video_key("F:/d/FaceForensic++_raw/manipulated_sequences/Face2Face/c23/faces23/003_000/1.png")


def unit_checks():
    """Donor legality: cell, video, label relation, determinism, -1 recording."""
    video_id = np.array(["a", "a", "b", "b", "c", "c", "d", "d", "e", "e", "f", "f"])
    domain = np.array(["t"] * 8 + ["u"] * 4)
    split = np.array(["test"] * 8 + ["train"] * 4)
    labels = np.array([1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1])
    for kind in KINDS:
        donors = label_donors(labels, video_id, domain, split, 11, kind)
        assert_video_labels_unique(labels, video_id, domain, split)
        usable = donors >= 0
        assert usable[:8].all(), "Well-populated cell must be fully covered"
        same_cell = (domain[donors[usable]] == domain[usable]) & (split[donors[usable]] == split[usable])
        assert same_cell.all(), "Donor escaped its (domain, split) cell"
        assert (video_id[donors[usable]] != video_id[usable]).all(), "Donor shared the row's video"
        matched = labels[donors[usable]] == labels[usable]
        assert matched.all() if kind == "SAME" else (~matched).all(), f"Wrong label relation for {kind}"
        # The all-fake train cell has no different-label video, so CROSS must record -1 there.
        assert (donors[8:] >= 0).all() if kind == "SAME" else (donors[8:] == -1).all()
        assert np.array_equal(donors, label_donors(labels, video_id, domain, split, 11, kind)), "Not deterministic"
        assert not np.array_equal(donors, label_donors(labels, video_id, domain, split, 12, kind)), "Seed ignored"
    assert not np.array_equal(label_donors(labels, video_id, domain, split, 11, "SAME"),
                              label_donors(labels, video_id, domain, split, 11, "CROSS")), "Kind ignored"

    # A single-video cell can never yield a donor of either kind.
    lonely = label_donors(np.array([1, 0]), np.array(["x", "x"]), np.array(["v", "v"]),
                          np.array(["test", "test"]), 11, "SAME")
    assert (lonely == -1).all(), "Same-video donor leaked through"

    values = np.arange(12)[:, None].repeat(3, axis=1).astype(np.float32)
    donors = np.array([-1, 1, -1, 3] + [-1] * 8)
    used, usable = substitute(values, donors)
    assert not usable[[0, 2]].any() and usable[[1, 3]].all()
    assert np.array_equal(used[0], values[0]) and np.array_equal(used[2], values[2]), "kept row was altered"
    assert np.array_equal(used[1], values[1]) and np.array_equal(used[3], values[3]), "donor row not substituted"

    bad_video = np.array(["a", "a"] + ["b"] * 10)
    try:
        assert_video_labels_unique(labels, bad_video, domain, split)
        raise AssertionError("Multi-label video was accepted")
    except ValueError as error:
        assert "carries labels" in str(error)


def build_reference(root):
    """Minimal but structurally faithful stand-in for a finished G29 run."""
    root.mkdir(parents=True)
    rng = np.random.default_rng(29)
    layout = [("ffpp", "train", 40), ("cd1", "test", 12), ("cd2", "test", 24),
              ("dfdcp", "test", 24), ("wild", "test", 24)]
    domain = np.concatenate([np.array([name] * count) for name, _, count in layout])
    split = np.concatenate([np.array([part] * count) for _, part, count in layout])
    count = len(domain)
    row_id = np.arange(count)
    video_id = np.array([f"{name}_v{index // 2}" for name, _, count_ in layout
                         for index in range(count_)])
    labels = np.array([(index // 2) % 2 for _, _, count_ in layout for index in range(count_)])
    data = {"row_id": row_id, "domain": domain, "split": split, "video_id": video_id, "y": labels,
            "path": np.array([f"synthetic/{name}/group_{index // 2}/frame_{index}.png"
                              for name, _, count_ in layout for index in range(count_)]),
            "V": rng.normal(size=(count, 8)).astype(np.float32),
            "C": rng.normal(size=(count, 12)).astype(np.float32)}
    data["V"][:, 0] += 2 * (2 * labels - 1)
    data["C"][:, 0] += (2 * labels - 1)
    train = np.flatnonzero(split == "train")
    features, scalers = standardize(data, train)

    seeds = [5, 6]
    scores = {}
    for variant in VARIANTS:
        for seed in seeds:
            model = VectorFusion(data["V"].shape[1], data["C"].shape[1], variant, seed)
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(torch.as_tensor(rng.normal(size=parameter.shape) * 0.01,
                                                   dtype=parameter.dtype))
            torch.save(model.state_dict(), root / f"model_{variant}_{seed}.pt")
            scores[f"{variant}|{seed}"], _ = predict(model, features)
            for repeat in range(2):
                rolled = np.roll(features["C"], 1, axis=0)
                scores[f"DONOR|{variant}|{seed}|{repeat}"], _ = predict(
                    model, {"V": features["V"], "C": rolled})
    identity = {key: data[key] for key in ("row_id", "path", "domain", "split", "video_id", "y")}
    np.savez_compressed(root / "features.npz", **identity, V=data["V"], C=data["C"])
    np.savez_compressed(root / "scalers.npz", **scalers)
    np.savez_compressed(root / "scores.npz", **identity, **scores)
    write_json(root / "input_config.json", {"label": LABEL})
    write_json(root / "config.json", {"status": "complete"})
    write_json(root / "artifact_hashes.json", {
        path.name: file_digest(path) for path in sorted(root.iterdir())
        if path.is_file() and path.name != "artifact_hashes.json"})
    (root / "COMPLETE").write_text("synthetic reference\n", encoding="utf-8")
    return len(seeds)


def run(reference, output, expect_success=True):
    command = [sys.executable, str(HERE / "donor_label_control.py"), "--reference", str(reference),
               "--output", str(output), "--primary-domains", "cd2", "dfdcp", "wild",
               "--seeds", "5", "6", "--repeats", "2", "--bootstrap", "20", "--threads", "1"]
    process = subprocess.run(command, cwd=HERE, capture_output=True, text=True)
    if expect_success:
        assert process.returncode == 0, f"Runner failed:\n{process.stdout}\n{process.stderr}"
    else:
        assert process.returncode != 0, "Runner should have refused this reference"
    return process


def integration_checks(directory):
    reference, output = directory / "reference", directory / "run01"
    build_reference(reference)
    run(reference, output)
    report = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert len(report["primary_macro"]) == len(VARIANTS) * 2 * 2 * len(KINDS), "Wrong number of comparisons"
    for key, value in report["primary_macro"].items():
        assert value["valid_bootstraps"] >= max(10, 20 // 2) and value["ci95"] is not None, f"No CI for {key}"
        assert value["ci95"][0] <= value["delta_macro_auc"] <= value["ci95"][1], f"Inverted CI for {key}"
        assert value["estimator"] == "mean of per-domain AUCs, not pooled AUC"
    assert report["comparison_roles"], "Roles unrecorded"
    for key, value in report["decomposition"].items():
        # The synthetic DONOR column is unrelated to the SAME/CROSS draws, so only
        # the shape and finiteness of the decomposition are program invariants here.
        for field in ("recorded_donor_delta", "same_label_delta", "cross_label_delta",
                      "mean_of_two", "residual_recorded_minus_mean"):
            assert np.isfinite(value[field]), f"{key} has non-finite {field}"
        assert len(value["recorded_donor_ci95"]) == 2 and len(value["same_label_ci95"]) == 2
    assert set(report["decomposition"]) == {f"{name}|{seed}|{repeat}" for name in VARIANTS
                                            for seed in (5, 6) for repeat in range(2)}
    with np.load(output / "scores.npz", allow_pickle=False) as archive:
        keys = set(archive.files)
        assert set(("row_id", "path", "domain", "split", "video_id", "y")) <= keys
        for variant in VARIANTS:
            for seed in (5, 6):
                assert f"{variant}|{seed}" in keys
                for repeat in range(2):
                    for kind in KINDS:
                        assert f"{kind}|{variant}|{seed}|{repeat}" in keys
    with np.load(output / "controls.npz", allow_pickle=False) as archive:
        assert f"donor_seed|SAME|5|0" in archive.files
        assert int(archive["donor_seed|SAME|5|0"]) == 5 + DONOR_SEED_OFFSET
        match = float(archive["label_match|SAME|5|0"])
        cross = float(archive["label_match|CROSS|5|0"])
        assert match == 1.0 and cross == 0.0, f"Label relations broken: {match}, {cross}"
    hashes = json.loads((output / "artifact_hashes.json").read_text(encoding="utf-8"))
    for name, digest in hashes.items():
        assert file_digest(output / name) == digest, f"Unhashed or mutated output: {name}"
    assert (output / "COMPLETE").exists() and (output / "code_snapshot" / "donor_label_control.py").exists()
    assert "COMPLETE" not in hashes and "config.json" in hashes
    run(reference, output, expect_success=False)  # Must refuse to overwrite a finished run

    tampered = directory / "tampered"
    build_reference(tampered)
    with (tampered / "features.npz").open("ab") as stream:
        stream.write(b"tamper")
    run(tampered, directory / "run02", expect_success=False)


def main():
    torch.set_num_threads(1)
    video_key_checks()
    unit_checks()
    with tempfile.TemporaryDirectory() as name:
        integration_checks(Path(name))
    print("check_donor_label: path-derived video identity, donor legality, determinism, -1 recording, "
          "substitution identity and the end-to-end run (hash gate, refusal to overwrite, tamper "
          "detection) all passed")


if __name__ == "__main__":
    main()
