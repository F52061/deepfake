"""G29-A: donor label control (same-label vs different-label donors) on G29's frozen weights.

Reads the finished G29 run read-only and trains nothing: this is a "fix the trained
model, swap only the input" intervention, the same shape as G29's DONOR control.
See DONOR_LABEL_CONTROL_EXPERIMENT.md for the pre-registered design.
"""

import argparse
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch

from analyze import summarize
from clip_readout import cell_seed, file_digest, write_json
from pure_vit_clip import macro_comparison
from vector_fusion import VectorFusion, predict, standardize

IDENTITY = ("row_id", "path", "domain", "split", "video_id", "y")
VARIANTS = ("FIXED", "LEARNED")
KINDS = ("SAME", "CROSS")
DONOR_SEED_OFFSET = 400000
LABEL = "1=fake, 0=real"


def check(condition, message):
    if not condition:
        raise ValueError(message)


def video_key(path):
    """Video identity derived from the file path, not from the `video_id` column.

    `video_id` is a person id in cd1/cd2/dfdcp/ffpp (one person owns both a real and a
    fake video) and a frame index in wild, so it cannot separate videos.  The path
    folder is the video: drop the `faces23` extraction marker, the file name and, where
    frames sit under a `real`/`fake` directory (wild), the frame directory as well.
    """
    parts = [part for part in str(path).replace("\\", "/").split("/") if part != "faces23"]
    folder = parts[:-1]
    if len(folder) >= 2 and folder[-2].lower() in {"real", "fake"}:
        folder = folder[:-1]
    return "/".join(folder)


def assert_video_labels_unique(labels, video, domain, split):
    """Within a cell, a video must carry one label, otherwise "different label"
    no longer implies "different video" and CROSS would not be a valid control."""
    for cell_domain in np.unique(domain):
        for cell_split in np.unique(split):
            cell = np.flatnonzero((domain == cell_domain) & (split == cell_split))
            for name in np.unique(video[cell]):
                seen = np.unique(labels[cell][video[cell] == name])
                check(len(seen) == 1,
                      f"Video {name} in {cell_domain}/{cell_split} carries labels {seen.tolist()}")


def label_donors(labels, video_id, domain, split, seed, kind):
    """Donor row per row: same (domain, split) cell, different video, and the
    requested label relation (SAME = equal label, CROSS = different label).

    Returns -1 where a cell holds no eligible donor, recorded rather than
    silently falling back to an ineligible donor.
    """
    donors = np.full(len(labels), -1, dtype=int)
    for cell_domain in np.unique(domain):
        for cell_split in np.unique(split):
            cell = np.flatnonzero((domain == cell_domain) & (split == cell_split))
            if not len(cell):
                continue
            rng = np.random.default_rng(cell_seed(seed, kind, f"{cell_domain}|{cell_split}"))
            cell_labels, cell_videos = labels[cell], video_id[cell]
            for video in np.unique(cell_videos):
                for label in np.unique(cell_labels[cell_videos == video]):
                    selected = np.flatnonzero((cell_videos == video) & (cell_labels == label))
                    eligible = (cell_labels == label) if kind == "SAME" else (cell_labels != label)
                    candidates = np.flatnonzero(eligible & (cell_videos != video))
                    if not len(candidates):
                        continue
                    donors[cell[selected]] = cell[rng.choice(candidates, size=len(selected), replace=True)]
    return donors


def substitute(clip_values, donor_rows):
    """Rows without an eligible donor keep their own C; the replacement is a no-op
    there rather than a silent donor substitution."""
    usable = donor_rows >= 0
    kept = np.where(usable, donor_rows, np.arange(len(donor_rows)))
    return clip_values[kept], usable


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, help="finished G29 run directory (read-only)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--primary-domains", nargs="+", default=["cd2", "dfdcp", "wild"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[20261030, 20261031, 20261032])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if (args.repeats < 1 or args.bootstrap < 0 or args.threads < 1 or len(args.seeds) != len(set(args.seeds))
            or "ffpp" in args.primary_domains or {"cd1", "cd2"}.issubset(args.primary_domains)):
        parser.error("Invalid repeats, bootstrap, threads, seeds or primary domains")
    return args


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)

    reference, output = Path(args.reference), Path(args.output)
    check((reference / "COMPLETE").exists(), "Reference run incomplete; frozen features and weights must exist")
    check((reference / "artifact_hashes.json").exists(), "Reference artifact hashes missing")
    if output.exists():
        raise ValueError(f"Refusing to overwrite {output}")
    reference_hashes = json.loads((reference / "artifact_hashes.json").read_text(encoding="utf-8"))

    def verified(name):
        path = reference / name
        check(path.exists(), f"Reference artifact missing: {name}")
        stored = reference_hashes.get(name)
        check(stored is not None, f"Reference artifact not covered by its own hash list: {name}")
        digest = file_digest(path)
        check(digest == stored, f"Reference artifact changed since its run: {name}")
        return path

    with np.load(verified("features.npz"), allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    extraction = json.loads((reference / "input_config.json").read_text(encoding="utf-8"))
    check(extraction.get("label") == LABEL, "Expected extraction labels 1=fake, 0=real")
    for key in (*IDENTITY, "V", "C"):
        check(key in data, f"features.npz missing {key}")
    check(np.array_equal(data["row_id"], np.arange(len(data["y"]))), "Reordered or missing feature rows")
    check(set(np.unique(data["split"])) == {"train", "test"}, "Expected both train and test rows")
    videos = np.array([video_key(value) for value in data["path"]])
    assert_video_labels_unique(data["y"], videos, data["domain"], data["split"])

    rows = np.flatnonzero(data["split"] == "train")
    features, scalers = standardize(data, rows)
    with np.load(verified("scalers.npz"), allow_pickle=False) as archive:
        stored = {key: archive[key] for key in archive.files}
    for key in ("V_mean", "V_scale", "C_mean", "C_scale"):
        check(np.array_equal(scalers[key], stored[key]),
              f"Rebuilt {key} differs from the reference scaler; refusing to continue")
    del scalers, stored

    models = {}
    for variant in VARIANTS:
        for seed in args.seeds:
            model = VectorFusion(data["V"].shape[1], data["C"].shape[1], variant, seed)
            state = torch.load(verified(f"model_{variant}_{seed}.pt"), map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            models[f"{variant}|{seed}"] = model

    with np.load(verified("scores.npz"), allow_pickle=False) as archive:
        recorded = {key: archive[key] for key in archive.files}

    scores, controls, donor_masks = {}, {}, {}
    largest_gap = 0.0
    for variant in VARIANTS:
        for seed in args.seeds:
            record = f"{variant}|{seed}"
            check(record in recorded, f"Reference scores lack the real column {record}")
            value, _ = predict(models[record], features)
            gap = float(np.max(np.abs(value - recorded[record])))
            largest_gap = max(largest_gap, gap)
            check(gap <= 1e-5,
                  f"Rebuilt real scores for {record} differ from the reference by {gap}; "
                  "weights or standardization do not reproduce the reference run")
            scores[record] = value
            for repeat in range(args.repeats):
                for kind in KINDS:
                    donor_seed = seed + DONOR_SEED_OFFSET + repeat
                    donor_rows = label_donors(data["y"], videos, data["domain"],
                                              data["split"], donor_seed, kind)
                    used_c, usable = substitute(features["C"], donor_rows)
                    check(not np.any(videos[donor_rows[usable]] == videos[usable]),
                          f"{kind}|{variant}|{seed}|{repeat} drew a donor from the row's own video")
                    key = f"{kind}|{variant}|{seed}|{repeat}"
                    scores[key], _ = predict(models[record], {"V": features["V"], "C": used_c})
                    donor_masks[key] = usable
                    controls[f"donors|{kind}|{seed}|{repeat}"] = donor_rows
                    controls[f"usable|{kind}|{seed}|{repeat}"] = usable
                    controls[f"donor_y|{kind}|{seed}|{repeat}"] = np.where(
                        usable, data["y"][np.clip(donor_rows, 0, None)], -1)
                    controls[f"label_match|{kind}|{seed}|{repeat}"] = np.array(float(np.mean(
                        data["y"][donor_rows[usable]] == data["y"][usable])))
                    controls[f"donor_seed|{kind}|{seed}|{repeat}"] = np.array(donor_seed)
        print(f"Rebuilt and intervened on seed {seed} (variants {len(VARIANTS)})", flush=True)
    print(f"Rebuilt real scores reproduce the reference within {largest_gap:.3e}", flush=True)

    primary = (data["split"] == "test") & np.isin(data["domain"], args.primary_domains)
    check(primary.any(), "No primary test rows found")
    for variant in VARIANTS:
        for seed in args.seeds:
            for repeat in range(args.repeats):
                for kind in KINDS:
                    key = f"{kind}|{variant}|{seed}|{repeat}"
                    dropped = int(np.sum(primary & ~controls[f"usable|{kind}|{seed}|{repeat}"]))
                    check(dropped == 0, f"{key} drops {dropped} primary target rows; report instead of masking")

    comparisons, roles = {}, {}
    for variant in VARIANTS:
        for seed in args.seeds:
            record = f"{variant}|{seed}"
            for repeat in range(args.repeats):
                for kind in KINDS:
                    key = f"{kind}|{variant}|{seed}|{repeat}"
                    comparisons[key + "_vs_" + record] = (key, record)
                    roles[key] = ("same_label_different_video_donor_primary" if kind == "SAME"
                                  else "different_label_donor_positive_control")

    report = {"per_domain": {}, "primary_macro": {}, "comparison_roles": roles, "decomposition": {},
              "design": "delta = real - control; positive means the donor lost information"}
    cache = {}
    for key, (base, proposed) in comparisons.items():
        usable = np.ones(len(data["y"]), dtype=bool)
        for name in (base, proposed):
            if name in donor_masks:
                usable &= donor_masks[name]
        for domain in np.unique(data["domain"][data["split"] == "test"]):
            mask = (data["domain"] == domain) & (data["split"] == "test") & usable
            report["per_domain"].setdefault(domain, {})[key] = (
                summarize(data, scores[base], scores[proposed], mask, args.bootstrap, args.seeds[0])
                if mask.any() else {"status": "no usable donor rows"})
        check(usable[primary].all(), f"{key} has incomplete donor coverage on primary domains")
        report["primary_macro"][key] = macro_comparison(
            data, scores[base], scores[proposed], args.primary_domains, args.bootstrap, args.seeds[0], cache)
        print(f"Summarized {key}", flush=True)

    for variant in VARIANTS:
        for seed in args.seeds:
            for repeat in range(args.repeats):
                name = f"{variant}|{seed}"
                recorded_key = f"DONOR|{name}|{repeat}"
                check(recorded_key in recorded, f"Reference scores lack {recorded_key}; decomposition skipped")
                reference_donor = macro_comparison(
                    data, recorded[recorded_key], scores[name], args.primary_domains,
                    args.bootstrap, args.seeds[0], cache)
                pair = {}
                for kind in KINDS:
                    key = f"{kind}|{variant}|{seed}|{repeat}"
                    pair[kind] = report["primary_macro"][key + "_vs_" + name]
                report["decomposition"][f"{name}|{repeat}"] = {
                    "recorded_donor_delta": reference_donor["delta_macro_auc"],
                    "recorded_donor_ci95": reference_donor["ci95"],
                    "same_label_delta": pair["SAME"]["delta_macro_auc"],
                    "same_label_ci95": pair["SAME"]["ci95"],
                    "cross_label_delta": pair["CROSS"]["delta_macro_auc"],
                    "cross_label_ci95": pair["CROSS"]["ci95"],
                    "mean_of_two": float(np.mean([pair["SAME"]["delta_macro_auc"],
                                                  pair["CROSS"]["delta_macro_auc"]])),
                    "residual_recorded_minus_mean": float(
                        reference_donor["delta_macro_auc"]
                        - np.mean([pair["SAME"]["delta_macro_auc"], pair["CROSS"]["delta_macro_auc"]]))}

    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "code_snapshot"
    snapshot.mkdir()
    hashes = {}
    for filename in ("donor_label_control.py", "DONOR_LABEL_CONTROL_EXPERIMENT.md", "vector_fusion.py",
                     "analyze.py", "clip_readout.py", "pure_vit_clip.py", "conditional_clip.py"):
        path = Path(__file__).with_name(filename)
        shutil.copy2(path, snapshot / filename)
        hashes[filename] = file_digest(path)
    identity = {key: data[key] for key in IDENTITY}
    np.savez_compressed(output / "scores.npz", **identity, **scores)
    np.savez_compressed(output / "controls.npz", **controls)
    write_json(output / "comparisons.json", {key: list(value) for key, value in comparisons.items()})
    write_json(output / "summary.json", report)
    write_json(output / "config.json", {
        "arguments": vars(args), "status": "complete", "label": LABEL,
        "feature_source": "read-only re-use of the finished G29 run",
        "normalization": "same source StandardScaler as the reference run, rebuilt and byte-compared",
        "video_identity": "path-derived folder (faces23 dropped; the frame directory dropped where frames sit "
                          "under real/fake, i.e. wild); the video_id column is a person id in cd1/cd2/dfdcp/ffpp "
                          "and a frame index in wild, so it is not used",
        "donor_rule": "same (domain, split), different path-derived video, SAME|CROSS label relation; "
                      "default_rng(cell_seed(seed + 400000 + repeat, kind, f'{domain}|{split}')); -1 recorded",
        "no_training": "fixed-model intervention on the saved G29 weights; no model is fitted here",
        "delta_sign": "delta = real - control; positive means the substituted C lost information",
        "code_sha256": hashes,
        "input_hashes": {name: reference_hashes[name] for name in sorted(reference_hashes)
                         if name != "artifact_hashes.json"},
        "versions": {"numpy": np.__version__, "torch": str(torch.__version__)}})
    write_json(output / "artifact_hashes.json", {
        str(path.relative_to(output)).replace("\\", "/"): file_digest(path)
        for path in sorted(output.rglob("*")) if path.is_file()})
    (output / "COMPLETE").write_text("G29-A donor label control completed\n", encoding="utf-8")
    print(f"Saved {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
