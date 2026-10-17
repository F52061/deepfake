"""Analyze frozen local ViT/CLIP descriptors and spatial controls."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from analyze import fit_reader, load_observations, save_reader, summarize, validate_splits
from local_features import relations


def load_local(directory, variant):
    files = sorted(Path(directory).glob(f"{variant}_*.npz"))
    if not files:
        raise ValueError(f"Missing {variant} shards")
    keys = ("row_id", "path", "domain", "split", "video_id", "y", "V", "C", "F", "logits",
            "vit_regions", "clip_regions")
    values = {key: [] for key in keys}
    for file in files:
        with np.load(file, allow_pickle=False) as shard:
            if "vit_regions" not in shard or "clip_regions" not in shard:
                raise ValueError("Extraction must use --save-regions")
            for key in keys:
                values[key].append(shard[key])
    data = {key: np.concatenate(items) for key, items in values.items()}
    if not np.array_equal(data["row_id"], np.arange(len(data["y"]))):
        raise ValueError("Rows are missing, duplicated or reordered")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source", default="ffpp")
    parser.add_argument("--regularization", type=float, default=1e-3)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--permutations", type=int, default=5)
    args = parser.parse_args()
    if args.permutations < 1 or args.bootstrap < 0 or args.regularization <= 0:
        parser.error("Invalid permutations, bootstrap or regularization")
    root = Path(args.input)
    if not (root / "COMPLETE").exists():
        raise ValueError("Extraction incomplete")
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if not config["arguments"].get("save_regions", False):
        raise ValueError("Input was not extracted with --save-regions")
    data = load_local(root, "clean")
    train = validate_splits(data, args.source)
    evaluation = data["split"] == "test"
    if not evaluation.any():
        raise ValueError("Test rows required")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    v_region = data["vit_regions"].astype(np.float32)
    c_region = data["clip_regions"].astype(np.float32)
    v_relation = relations(v_region).astype(np.float32)
    c_relation = relations(c_region).astype(np.float32)
    v_region /= np.linalg.norm(v_region, axis=-1, keepdims=True).clip(1e-8)
    c_region /= np.linalg.norm(c_region, axis=-1, keepdims=True).clip(1e-8)
    v_flat, c_flat = v_region.reshape(len(v_region), -1), c_region.reshape(len(c_region), -1)
    sets = {
        "V": data["V"], "C": data["C"], "F": data["F"],
        "V_C": np.concatenate([data["V"], data["C"]], axis=1),
        "V_regions": v_flat, "C_regions": c_flat, "V_C_regions": np.concatenate([v_flat, c_flat], axis=1),
        "V_C_relations": np.concatenate([data["V"], data["C"], v_relation, c_relation], axis=1),
        "V_relations": np.concatenate([data["V"], v_relation], axis=1),
        "C_relations": np.concatenate([data["C"], c_relation], axis=1),
    }
    scores, readers = {}, {}
    for name, features in sets.items():
        reader = fit_reader(features, data["y"], train, args.regularization, args.seed)
        readers[name] = reader
        scores[name] = reader.decision_function(features)
        save_reader(output / f"reader_{name}.npz", reader)
    rng = np.random.default_rng(args.seed)
    controls = {}
    for repeat in range(args.permutations):
        c_perm = np.empty_like(c_region)
        for domain in np.unique(data["domain"]):
            for split in np.unique(data["split"]):
                ids = np.flatnonzero((data["domain"] == domain) & (data["split"] == split))
                if len(ids) == 0:
                    # Skip empty cells. np.unique(data["split"]) is global, so this
                    # loop visits every (domain, split) combination, but a given
                    # domain need not populate every split: in our manifest only
                    # ffpp has "train" rows, so cd1/train etc. are empty. There is
                    # nothing to permute there, and failing would make the control
                    # unrunnable. A cell with exactly one row is still an error,
                    # because permutation needs a partner.
                    continue
                if len(ids) < 2:
                    raise ValueError(f"Spatial permutation needs >=2 rows: {domain}/{split}")
                c_perm[ids] = c_region[ids[rng.permutation(len(ids))]]
        perm = np.concatenate([v_flat, c_perm.reshape(len(c_perm), -1)], axis=1)
        scores[f"spatial_shuffle_{repeat}"] = readers["V_C_regions"].decision_function(perm)
        controls[f"c_regions_shuffle_{repeat}"] = c_perm
        noise = rng.normal(size=c_region.shape) * c_region[train].std(axis=0) + c_region[train].mean(axis=0)
        noise_features = np.concatenate([v_flat, noise.reshape(len(noise), -1)], axis=1)
        scores[f"region_noise_fixed_{repeat}"] = readers["V_C_regions"].decision_function(noise_features)
        noise_reader = fit_reader(noise_features, data["y"], train, args.regularization, args.seed)
        scores[f"region_noise_retrain_{repeat}"] = noise_reader.decision_function(noise_features)
        save_reader(output / f"reader_region_noise_{repeat}.npz", noise_reader)
        controls[f"region_noise_{repeat}"] = noise
    identity = {key: data[key] for key in ("row_id", "path", "domain", "split", "video_id", "y")}
    np.savez_compressed(output / "features.npz", **identity, vit_regions=v_region,
                        clip_regions=c_region, vit_relations=v_relation, clip_relations=c_relation)
    np.savez_compressed(output / "scores.npz", **identity, **scores)
    np.savez_compressed(output / "controls.npz", **controls)
    report = {}
    for domain in np.unique(data["domain"][evaluation]):
        mask = evaluation & (data["domain"] == domain)
        report[domain] = {name: summarize(data, scores["V"], score, mask, args.bootstrap, args.seed)
                          for name, score in scores.items()}
        report[domain]["V_C_regions_vs_V_C"] = summarize(
            data, scores["V_C"], scores["V_C_regions"], mask, args.bootstrap, args.seed)
    metadata = {
        "arguments": vars(args), "input_config_sha256": hashlib.sha256(
            (root / "config.json").read_bytes()).hexdigest(),
        "grid": int(round(v_region.shape[1] ** 0.5)), "region_count": int(v_region.shape[1]),
        "descriptors": "normalized 3x3 pooled token regions; pairwise and region-to-global cosine relations",
        "label": "1=fake", "interpretation": "diagnostic probes only; no architecture training",
    }
    (output / "config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    (output / "COMPLETE").write_text("Analysis completed\n", encoding="utf-8")
    print(f"Saved {output.resolve()}")


if __name__ == "__main__":
    main()
