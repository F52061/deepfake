"""Source-only axis interventions and semantic incremental-information diagnostics."""

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import sklearn


def load_observations(directory, variant):
    files = sorted(Path(directory).glob(f"{variant}_*.npz"))
    if not files:
        raise ValueError(f"No shards for {variant}")
    keys = ("V", "C", "S", "F", "logits", "row_id", "path", "y", "split", "domain", "video_id")
    columns = {key: [] for key in keys}
    for file in files:
        with np.load(file, allow_pickle=False) as shard:
            for key in keys:
                columns[key].append(shard[key])
    data = {key: np.concatenate(values) for key, values in columns.items()}
    if not np.array_equal(data["row_id"], np.arange(len(data["y"]))):
        raise ValueError("Missing, duplicate or unordered rows")
    for key in ("V", "C", "S", "F", "logits"):
        if not np.isfinite(data[key]).all():
            raise ValueError(f"Nonfinite {key}")
    return data


def validate_splits(data, source):
    train = (data["split"] == "train") & (data["domain"] == source)
    if np.any((data["split"] == "train") & ~train):
        raise ValueError("Only source-domain training rows allowed")
    if set(data["y"].tolist()) != {0, 1} or len(np.unique(data["y"][train])) != 2:
        raise ValueError("Both classes required; labels must be 0/1")
    if len(np.unique(data["path"])) != len(data["path"]):
        raise ValueError("Duplicate paths")
    ownership = {}
    for domain, video, split in zip(data["domain"], data["video_id"], data["split"]):
        if not video:
            raise ValueError("Missing video ID")
        key = (domain, video)
        if key in ownership and ownership[key] != split:
            raise ValueError(f"Video overlaps splits: {key}")
        ownership[key] = split
    return train


def replay_head(data, head, features=None):
    features = data["V"] if features is None else features
    projected = features @ head["proj_weight"].T + head["proj_bias"]
    projected = (projected - projected.mean(axis=1, keepdims=True)) / np.sqrt(
        projected.var(axis=1, keepdims=True) + head["norm_eps"])
    projected = projected * head["norm_weight"] + head["norm_bias"]
    combined = data["F"].astype(projected.dtype, copy=True)
    combined[:, -projected.shape[1]:] = projected
    logits = combined @ head["weight"].T + head["bias"]
    fake = int(head["fake_logit"])
    return logits[:, fake] - logits[:, 1 - fake]


def fit_reader(features, labels, train, regularization, seed):
    reader = make_pipeline(StandardScaler(), LogisticRegression(
        C=regularization, max_iter=3000, random_state=seed))
    reader.fit(features[train], labels[train])
    return reader


def save_reader(path, reader):
    scaler, logistic = reader.steps[0][1], reader.steps[1][1]
    np.savez_compressed(path, mean=scaler.mean_, scale=scaler.scale_,
                        coef=logistic.coef_, intercept=logistic.intercept_, classes=logistic.classes_)


def metric(labels, scores):
    return float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None


def donor_indices(groups, rng):
    if len(np.unique(groups)) < 2:
        raise ValueError("Semantic shuffle requires >=2 videos per domain/split")
    donors = np.empty(len(groups), dtype=int)
    for group in np.unique(groups):
        selected = np.flatnonzero(groups == group)
        candidates = np.flatnonzero(groups != group)
        donors[selected] = rng.choice(candidates, size=len(selected), replace=True)
    return donors


def paired_interval(labels, baseline, candidate, groups, repeats, seed):
    unique = np.unique(groups)
    if len(unique) < 2 or repeats == 0:
        return None
    rng = np.random.default_rng(seed)
    blocks = [np.flatnonzero(groups == group) for group in unique]
    deltas = []
    for iteration in range(repeats):
        indices = np.concatenate([blocks[index] for index in rng.integers(0, len(blocks), len(blocks))])
        if len(np.unique(labels[indices])) == 2:
            deltas.append(metric(labels[indices], candidate[indices]) - metric(labels[indices], baseline[indices]))
    return np.quantile(deltas, [0.025, 0.975]).tolist() if len(deltas) >= max(10, repeats // 2) else None


def summarize(data, baseline, candidate, mask, bootstrap, seed):
    labels, base, score = data["y"][mask], baseline[mask], candidate[mask]
    videos = data["video_id"][mask]
    wrong, new_wrong = (base >= 0) != labels, (score >= 0) != labels
    base_auc, new_auc = metric(labels, base), metric(labels, score)
    return {
        "n": len(labels), "videos": len(np.unique(videos)), "auc": new_auc,
        "baseline_auc": base_auc,
        "delta_auc": None if base_auc is None else new_auc - base_auc,
        "delta_auc_video_bootstrap_ci95": paired_interval(labels, base, score, videos, bootstrap, seed),
        "accuracy": float(np.mean(~new_wrong)), "baseline_errors": int(wrong.sum()),
        "rescued": int(np.sum(wrong & ~new_wrong)), "harmed": int(np.sum(~wrong & new_wrong)),
        "error_subset_auc": metric(labels[wrong], score[wrong]),
        "prediction_flip_rate": float(np.mean((base >= 0) != (score >= 0))),
        "mean_abs_score_change": float(np.mean(np.abs(score - base))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source", default="ffpp")
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--regularization", type=float, default=1e-3)
    parser.add_argument("--radii", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0])
    args = parser.parse_args()
    if args.repeats < 1 or args.bootstrap < 0 or args.regularization <= 0 or any(radius <= 0 for radius in args.radii):
        parser.error("Invalid repetitions, bootstrap, regularization or radii")
    directory = Path(args.input)
    if not (directory / "COMPLETE").exists():
        raise ValueError("Extraction incomplete")
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    data = load_observations(directory, "clean")
    train = validate_splits(data, args.source)
    evaluation = data["split"] == "test"
    if not evaluation.any():
        raise ValueError("Test rows required")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    with np.load(directory / "head.npz", allow_pickle=False) as archive:
        head = {key: archive[key] for key in archive.files}
    mean = data["V"][train].mean(axis=0, dtype=np.float64)
    centered = data["V"].astype(np.float64) - mean
    _, singular, basis = np.linalg.svd(centered[train], full_matrices=False)
    axis = basis[0]
    if singular[0] <= 1e-12 or data["V"].shape[1] < 2:
        raise ValueError("Degenerate source representation")
    axial = centered @ axis
    axis_std = float(axial[train].std())
    detector = replay_head(data, head)
    fake = int(head["fake_logit"])
    original = data["logits"][:, fake] - data["logits"][:, 1 - fake]
    replay_error = float(np.max(np.abs(detector - original)))
    if not np.allclose(detector, original, atol=1e-4, rtol=1e-4):
        raise ValueError(f"Detector head replay mismatch: {replay_error}")
    np.savez_compressed(output / "source_axis.npz", mean=mean, axis=axis, basis=basis, singular=singular,
                        axis_std=np.array(axis_std), source_row_id=data["row_id"][train])
    scores = {"detector": detector}
    readers = {}
    feature_sets = {"V": data["V"], "S": data["S"], "C": data["C"],
                    "V_S": np.concatenate([data["V"], data["S"]], axis=1),
                    "V_C": np.concatenate([data["V"], data["C"]], axis=1)}
    for name, features in feature_sets.items():
        reader = fit_reader(features, data["y"], train, args.regularization, args.seed)
        readers[name] = reader
        scores[name] = reader.decision_function(features)
        save_reader(output / f"reader_{name}.npz", reader)
    interventions, donors_saved = [], {}
    for repeat in range(args.repeats):
        rng = np.random.default_rng(args.seed + repeat)
        signs = rng.choice([-1.0, 1.0], size=len(axial))
        orthogonal = rng.normal(size=centered.shape)
        orthogonal -= (orthogonal @ axis)[:, None] * axis
        orthogonal /= np.linalg.norm(orthogonal, axis=1, keepdims=True).clip(1e-12)
        random_direction = rng.normal(size=centered.shape)
        random_direction /= np.linalg.norm(random_direction, axis=1, keepdims=True).clip(1e-12)
        fixed_orthogonal = rng.normal(size=len(axis))
        fixed_orthogonal -= (fixed_orthogonal @ axis) * axis
        fixed_orthogonal /= np.linalg.norm(fixed_orthogonal)
        directions = {
            "axis": signs[:, None] * axis, "orthogonal": orthogonal, "random": random_direction,
            "fixed_orthogonal": signs[:, None] * fixed_orthogonal,
            "pc1": signs[:, None] * basis[1],
        }
        np.savez_compressed(output / f"directions_rep{repeat}.npz", **directions)
        for radius in args.radii:
            length = radius * axis_std
            for name, direction in directions.items():
                features = data["V"] + length * direction
                key = f"{name}_r{radius:g}_rep{repeat}"
                scores[key] = replay_head(data, head, features)
                scores[key + "_probe"] = readers["V"].decision_function(features)
                if name == "axis":
                    scores[key + "_V_S"] = readers["V_S"].decision_function(
                        np.concatenate([features, data["S"]], axis=1))
                norms = np.linalg.norm(features - data["V"], axis=1)
                interventions.append({"condition": key, "radius": radius, "repeat": repeat,
                                      "l2_min": float(norms.min()), "l2_max": float(norms.max())})
        donors = np.arange(len(axial))
        for domain in np.unique(data["domain"]):
            for split in np.unique(data["split"]):
                indices = np.flatnonzero((data["domain"] == domain) & (data["split"] == split))
                if len(indices):
                    donors[indices] = indices[donor_indices(data["video_id"][indices], rng)]
        donors_saved[f"donors_rep{repeat}"] = donors
        shuffled = np.concatenate([data["V"], data["S"][donors]], axis=1)
        scores[f"V_S_evalshuffle_rep{repeat}"] = readers["V_S"].decision_function(shuffled)
        shuffled_reader = fit_reader(shuffled, data["y"], train, args.regularization, args.seed)
        scores[f"V_S_trainshuffle_rep{repeat}"] = shuffled_reader.decision_function(shuffled)
        save_reader(output / f"reader_shuffle_rep{repeat}.npz", shuffled_reader)
        noise = rng.normal(size=data["S"].shape)
        noise = noise * data["S"][train].std(axis=0) + data["S"][train].mean(axis=0)
        noise_features = np.concatenate([data["V"], noise], axis=1)
        noise_reader = fit_reader(noise_features, data["y"], train, args.regularization, args.seed)
        scores[f"V_S_noise_rep{repeat}"] = noise_reader.decision_function(noise_features)
        donors_saved[f"semantic_noise_rep{repeat}"] = noise
        save_reader(output / f"reader_noise_rep{repeat}.npz", noise_reader)
    for variant in config["arguments"]["variants"]:
        if variant == "clean":
            continue
        changed = load_observations(directory, variant)
        for key in ("row_id", "path", "y", "split", "domain", "video_id"):
            if not np.array_equal(changed[key], data[key]):
                raise ValueError(f"Variant metadata differs: {variant}/{key}")
        delta = changed["V"] - data["V"]
        axis_delta = delta @ axis
        orthogonal_delta = delta - axis_delta[:, None] * axis
        scores[f"image_{variant}"] = changed["logits"][:, fake] - changed["logits"][:, 1 - fake]
        scores[f"image_{variant}_V_S"] = readers["V_S"].decision_function(np.concatenate([changed["V"], changed["S"]], axis=1))
        scores[f"image_{variant}_V"] = readers["V"].decision_function(changed["V"])
        scores[f"image_{variant}_axis_only"] = replay_head(data, head, data["V"] + axis_delta[:, None] * axis)
        scores[f"image_{variant}_orthogonal_only"] = replay_head(data, head, data["V"] + orthogonal_delta)
        scores[f"image_{variant}_V_replay"] = replay_head(data, head, changed["V"])
        np.savez_compressed(output / f"image_{variant}_decomposition.npz", row_id=data["row_id"],
                            axis_delta=axis_delta, orthogonal_l2=np.linalg.norm(orthogonal_delta, axis=1),
                            total_l2=np.linalg.norm(delta, axis=1), semantic_delta=changed["S"] - data["S"])
    comparisons = {}
    for domain in np.unique(data["domain"][evaluation]):
        mask = evaluation & (data["domain"] == domain)
        comparisons[domain] = {}
        for name, score in scores.items():
            if name.startswith("axis_") and name.endswith("_V_S"):
                baseline = scores[name[:-4] + "_probe"]
            elif name in feature_sets or name.startswith("V_S_") or name.endswith("_probe"):
                baseline = scores["V"]
            elif name.startswith("image_") and (name.endswith("_V_S") or name.endswith("_V")):
                suffix_length = 4 if name.endswith("_V_S") else 2
                variant = name[len("image_"):-suffix_length]
                baseline = scores[f"image_{variant}_V"]
            else:
                baseline = detector
            comparisons[domain][name] = summarize(data, baseline, score, mask, args.bootstrap, args.seed)
        comparisons[domain]["V_S_on_detector_failures"] = summarize(data, detector, scores["V_S"], mask, args.bootstrap, args.seed)
    metadata = {"arguments": vars(args), "extraction": config, "replay_max_abs_error": replay_error,
                "source_pc0_variance_fraction": float(singular[0] ** 2 / np.sum(singular ** 2)),
                "numpy": np.__version__, "sklearn": sklearn.__version__, "interventions": interventions,
                "scope": "CLS interventions hold CLIP/bridge fixed; image variants change all branches",
                "threshold": "source-trained logistic decision=0; fixed detector fake-minus-real logit=0",
                "interpretation": "error subsets are descriptive, never fitted or used for tuning"}
    (output / "config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(comparisons, indent=2, allow_nan=False), encoding="utf-8")
    columns = {key: data[key] for key in ("row_id", "path", "y", "split", "domain", "video_id")}
    np.savez_compressed(output / "sample_scores.npz", **columns, axis_coordinate=axial, **scores, **donors_saved)
    with open(output / "samples.csv", "w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        csv_columns = {**columns, "axis_coordinate": axial, **scores}
        writer.writerow(csv_columns)
        writer.writerows(zip(*csv_columns.values()))
    print(f"Saved diagnostics to {output.resolve()}; replay error={replay_error:.3g}")


if __name__ == "__main__":
    main()
