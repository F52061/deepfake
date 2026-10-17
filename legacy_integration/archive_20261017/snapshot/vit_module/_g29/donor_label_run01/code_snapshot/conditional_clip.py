"""G26 phase 1: read CLIP image regions conditionally, correct a frozen ViT score.

Implements vit_module/diagnostics/CONDITIONAL_CLIP_EXPERIMENT.md.

The ViT decision s_V is FIXED for the whole experiment; every trainable reader only
predicts a correction delta, so s(x) = s_V(x) + delta(x). Nothing here updates the
ViT, CLIP or the original classification head -- only the small readers are fitted.

Everything is fitted on FF++ train only, grouped by video, and evaluated on the
target domains. Target-domain labels are never used for training, model selection
or donor selection.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn

from analyze import summarize, validate_splits


SHARDS = ("row_id", "path", "domain", "split", "video_id", "y", "V", "C", "logits", "clip_regions")
VARIANTS = ("B", "C", "D", "E", "F")


def load_local(directory, variant="clean"):
    """Read the per-sample shards written by extract.py --save-regions."""
    files = sorted(Path(directory).glob(f"{variant}_*.npz"))
    if not files:
        raise ValueError(f"Missing {variant} shards")
    values = {key: [] for key in SHARDS}
    for file in files:
        with np.load(file, allow_pickle=False) as shard:
            if "clip_regions" not in shard:
                raise ValueError("Extraction must use --save-regions")
            for key in SHARDS:
                values[key].append(shard[key])
    data = {key: np.concatenate(items) for key, items in values.items()}
    if not np.array_equal(data["row_id"], np.arange(len(data["y"]))):
        raise ValueError("Rows are missing, duplicated or reordered")
    return data


def normalize_regions(regions):
    """Per-region L2 normalisation, as in the local-structure experiment."""
    regions = np.asarray(regions, dtype=np.float32)
    return regions / np.linalg.norm(regions, axis=-1, keepdims=True).clip(1e-8)


def standardize_features(data, fit_rows):
    """Standardize reader inputs using only the supplied fit rows."""
    from sklearn.preprocessing import StandardScaler

    scalers = {key: StandardScaler().fit(data[key][fit_rows]) for key in ("v", "c")}
    region_values = data["ci"].reshape(len(data["y"]), -1)
    region_scaler = StandardScaler().fit(region_values[fit_rows])
    return {
        "v": scalers["v"].transform(data["v"]).astype(np.float32),
        "c": scalers["c"].transform(data["c"]).astype(np.float32),
        "ci": region_scaler.transform(region_values).astype(np.float32).reshape(
            len(data["y"]), data["ci"].shape[1], data["ci"].shape[2]),
        "s": data["s"],
    }


def split_donors(video_id, domain, split, rng):
    """Donor rows within the same (domain, split) but never the same video.

    Returns -1 where a cell holds fewer than two videos, recorded rather than
    silently falling back to a same-video donor.
    """
    donors = np.full(len(video_id), -1, dtype=int)
    for cell_domain in np.unique(domain):
        for cell_split in np.unique(split):
            cell = np.flatnonzero((domain == cell_domain) & (split == cell_split))
            if len(cell) == 0:
                continue
            videos = video_id[cell]
            if len(np.unique(videos)) < 2:
                continue
            for group in np.unique(videos):
                selected = np.flatnonzero(videos == group)
                candidates = np.flatnonzero(videos != group)
                donors[cell[selected]] = cell[rng.choice(candidates, size=len(selected), replace=True)]
    return donors


def split_cross_domain_donors(video_id, domain, split, rng):
    """Choose donors from another domain and the same split.

    This is a stronger domain-mismatch control than ``split_donors``.  It is
    intentionally allowed to return -1 for a cell with only one domain (the
    FF++ source train cell is such a case); callers must report and fall back
    without using those rows as evidence.
    """
    donors = np.full(len(video_id), -1, dtype=int)
    for cell_split in np.unique(split):
        cell = np.flatnonzero(split == cell_split)
        for cell_domain in np.unique(domain[cell]):
            target = cell[domain[cell] == cell_domain]
            candidates = cell[domain[cell] != cell_domain]
            if len(target) == 0 or len(candidates) == 0:
                continue
            for row in target:
                eligible = candidates[video_id[candidates] != video_id[row]]
                if len(eligible):
                    donors[row] = rng.choice(eligible)
    return donors


def correction_diagnostics(labels, baseline, candidate):
    """Describe whether the correction itself carries domain-conditional signal."""
    from sklearn.metrics import roc_auc_score

    labels = np.asarray(labels)
    baseline = np.asarray(baseline)
    candidate = np.asarray(candidate)
    delta = candidate - baseline
    result = {
        "delta_mean": float(delta.mean()),
        "delta_std": float(delta.std()),
        "delta_abs_p50": float(np.quantile(np.abs(delta), 0.50)),
        "delta_abs_p90": float(np.quantile(np.abs(delta), 0.90)),
    }
    if len(np.unique(labels)) == 2:
        result["delta_auc"] = float(roc_auc_score(labels, delta))
        result["baseline_auc"] = float(roc_auc_score(labels, baseline))
        result["candidate_auc"] = float(roc_auc_score(labels, candidate))
    else:
        result.update({"delta_auc": None, "baseline_auc": None, "candidate_auc": None})
    baseline_wrong = (baseline >= 0) != labels.astype(bool)
    candidate_wrong = (candidate >= 0) != labels.astype(bool)
    result["rescue_rate"] = float(np.mean(baseline_wrong & ~candidate_wrong))
    result["harm_rate"] = float(np.mean(~baseline_wrong & candidate_wrong))
    result["net_correction_rate"] = result["rescue_rate"] - result["harm_rate"]
    return result


class RegionReader(nn.Module):
    """F: V queries the nine CLIP regions and returns one correction.

    Permutation invariant by construction: the query comes from V alone and the
    regions enter only through a softmax-weighted sum, so reordering regions cannot
    change the output. The spec therefore forbids region-position shuffling as a
    control here (section 4.6), and this implementation does not run it.
    """

    def __init__(self, v_dim, c_dim, dim, width):
        super().__init__()
        self.norm_v = nn.LayerNorm(v_dim)
        self.norm_c = nn.LayerNorm(c_dim)
        self.project = nn.Linear(c_dim, dim)
        self.query = nn.Linear(v_dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        head = nn.Sequential(nn.Linear(v_dim + dim + 1, width), nn.ReLU(), nn.Linear(width, 1))
        nn.init.zeros_(head[-1].weight)
        nn.init.zeros_(head[-1].bias)
        self.head = head
        self.scale = dim ** -0.5

    def forward(self, v, c, ci, s):
        visual = self.norm_v(v)
        hidden = self.project(self.norm_c(ci))
        logits = (self.key(hidden) * self.query(visual).unsqueeze(1)).sum(-1) * self.scale
        weights = torch.softmax(logits, dim=1)
        pooled = (weights.unsqueeze(-1) * self.value(hidden)).sum(1)
        delta = self.head(torch.cat([visual, pooled, s.unsqueeze(-1)], dim=1))
        return delta.squeeze(-1), weights


class VectorReader(nn.Module):
    """B/C/D/E: a plain head over one already-assembled feature vector."""

    def __init__(self, in_dim, width, hidden_layers):
        super().__init__()
        layers, current = [], in_dim
        for _ in range(hidden_layers):
            layers += [nn.Linear(current, width), nn.ReLU()]
            current = width
        layers += [nn.Linear(current, 1)]
        nn.init.zeros_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)
        self.net = nn.Sequential(*layers)

    def forward_input(self, features, name):
        if name == "B":
            return features["v"]
        if name == "C":
            return torch.cat([features["v"], features["c"]], dim=1)
        if name == "D":
            return torch.cat([features["v"], features["ci"].mean(1)], dim=1)
        if name == "E":
            return torch.cat([features["v"], features["ci"].reshape(len(features["v"]), -1)], dim=1)
        raise ValueError(f"VectorReader does not handle {name}")

    def forward(self, v, c, ci, s, name):
        return self.net(self.forward_input({"v": v, "c": c, "ci": ci}, name)).squeeze(-1), None


def build(name, v_dim, c_dim, regions, width, dim, depth):
    if name == "B":
        return VectorReader(v_dim, width, depth)
    if name == "C":
        return VectorReader(v_dim + c_dim, width, depth)
    if name == "D":
        return VectorReader(v_dim + c_dim, width, depth)
    if name == "E":
        # Linear residual head over the concatenated local features, per the spec.
        return VectorReader(v_dim + regions * c_dim, width, 0)
    if name == "F":
        return RegionReader(v_dim, c_dim, dim, width)
    raise ValueError(f"Unknown variant {name}")


def parameter_count(model):
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def capacity_matched_widths(v_dim, c_dim, regions, dim, width, depth, target):
    """Choose hidden widths so B/C/D hold at least as many parameters as F.

    The spec makes "F must also beat B" the control that rules out a pure capacity
    or nonlinearity explanation, and requires the variants to be parameter-matched.
    That control is only meaningful if B is not the smaller model, so each width is
    the smallest one whose parameter count reaches F's, never falls short of it.
    """
    widths = {}
    for name in ("B", "C", "D"):
        low, high, best = 1, 8192, None
        while low <= high:
            middle = (low + high) // 2
            candidate = parameter_count(build(name, v_dim, c_dim, regions, middle, dim, depth))
            if candidate >= target:
                best, high = middle, middle - 1
            else:
                low = middle + 1
        if best is None:
            raise ValueError(f"Cannot match F's parameter count for variant {name}")
        widths[name] = best
    return widths


class Store:
    """Standardised features held as float32 numpy; batches become torch on demand."""

    def __init__(self, features, regions, c_dim, device):
        self.v, self.c, self.ci, self.s = (features[key] for key in ("v", "c", "ci", "s"))
        self.regions, self.c_dim, self.device = regions, c_dim, device

    def batch(self, index):
        return (torch.as_tensor(self.v[index], device=self.device),
                torch.as_tensor(self.c[index], device=self.device),
                torch.as_tensor(self.ci[index].reshape(len(index), self.regions, self.c_dim),
                                device=self.device),
                torch.as_tensor(self.s[index], device=self.device))


def forward(model, name, batch):
    return model(*batch, name) if isinstance(model, VectorReader) else model(*batch)


def train_reader(model, name, store, index, labels, hyper, args, device):
    torch.manual_seed(args.seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=hyper["weight_decay"])
    objective = nn.BCEWithLogitsLoss()
    batch = store.batch(index)
    target = torch.as_tensor(labels, device=device).float()
    model.train()
    for _ in range(args.epochs):
        optimizer.zero_grad()
        delta, _ = forward(model, name, batch)
        loss = objective(batch[3] + delta, target) + hyper["lambda_delta"] * delta.pow(2).mean()
        loss.backward()
        optimizer.step()
    return model


def score_reader(model, name, store, index):
    model.eval()
    with torch.no_grad():
        delta, weights = forward(model, name, store.batch(index))
    return (store.s[np.asarray(index)] + delta.cpu().numpy()), \
           (None if weights is None else weights.cpu().numpy())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source", default="ffpp")
    parser.add_argument("--seeds", type=int, nargs="+", default=[20261004, 20261005, 20261006])
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 1e-4, 1e-3, 1e-2])
    parser.add_argument("--weight-decays", type=float, nargs="+", default=[1e-4, 1e-3])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if min(args.lambdas) < 0 or min(args.weight_decays) <= 0 or args.epochs < 1 or args.folds < 2:
        parser.error("Invalid regularisation or training budget")
    if args.bootstrap < 0 or args.repeats < 1 or len(args.seeds) < 1 or args.threads < 1:
        parser.error("Invalid bootstrap, repeats, seeds or threads")

    torch.set_num_threads(args.threads)
    device = torch.device("cpu")
    root = Path(args.input)
    if not (root / "COMPLETE").exists():
        raise ValueError("Extraction incomplete")
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if not config["arguments"].get("save_regions", False):
        raise ValueError("Input was not extracted with --save-regions")
    data = load_local(root, "clean")
    train = validate_splits(data, args.source)
    if data["y"][train].sum() in (0, int(train.sum())):
        raise ValueError("Both classes required in the source training rows")

    # s_V is the frozen ViT decision; a reader never replaces it, only corrects it.
    data["s"] = (data["logits"][:, 0] - data["logits"][:, 1]).astype(np.float32)
    regions = int(data["clip_regions"].shape[1])
    c_dim = int(data["clip_regions"].shape[2])
    data["v"] = data["V"].astype(np.float32)
    data["c"] = data["C"].astype(np.float32)
    data["ci"] = normalize_regions(data["clip_regions"]).reshape(len(data["y"]), regions, c_dim)
    evaluation = data["split"] == "test"
    if not evaluation.any():
        raise ValueError("Test rows required")
    if not np.isfinite(data["s"]).all():
        raise ValueError("Nonfinite s_V")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)

    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score

    train_index = np.flatnonzero(train)
    videos = data["video_id"][train]

    # ---- preconditions that must hold before anything is fitted ----
    mixed = sum(1 for video in np.unique(videos)
                if len(set(data["y"][(data["video_id"] == video) & train])) == 2)
    rng = np.random.default_rng(args.seeds[0])
    donors = split_donors(data["video_id"], data["domain"], data["split"], rng)
    cross_domain_donors = split_cross_domain_donors(
        data["video_id"], data["domain"], data["split"], rng)
    usable = donors >= 0
    cross_usable = cross_domain_donors >= 0
    checks = {
        "s_v_auc_source_train": float(roc_auc_score(data["y"][train], data["s"][train])),
        "regions": regions, "region_dim": c_dim,
        "train_rows": int(train.sum()), "train_videos": int(len(np.unique(videos))),
        "train_videos_with_both_classes": int(mixed),
        "rows_without_donor": int((~usable).sum()),
        "rows_without_cross_domain_donor": int((~cross_usable).sum()),
        "donor_same_video_violations": int(np.sum(
            data["video_id"][donors[usable]] == data["video_id"][usable])),
        "donor_cell_violations": int(np.sum(
            (data["domain"][donors[usable]] != data["domain"][usable]) |
            (data["split"][donors[usable]] != data["split"][usable]))),
        "cross_domain_same_domain_violations": int(np.sum(
            data["domain"][cross_domain_donors[cross_usable]] ==
            data["domain"][cross_usable])),
        "cross_domain_same_video_violations": int(np.sum(
            data["video_id"][cross_domain_donors[cross_usable]] ==
            data["video_id"][cross_usable])),
        "cross_domain_split_violations": int(np.sum(
            data["split"][cross_domain_donors[cross_usable]] !=
            data["split"][cross_usable])),
    }
    if checks["s_v_auc_source_train"] < 0.5:
        raise ValueError("s_V is oriented backwards; fake score must be logits[:,0]-logits[:,1]")
    if checks["train_videos_with_both_classes"] != checks["train_videos"]:
        raise ValueError("Some source videos do not carry both classes; grouping by video would leak")
    if checks["donor_same_video_violations"] or checks["donor_cell_violations"]:
        raise ValueError("Donor construction violated the same-video or same-cell rule")
    if (checks["cross_domain_same_domain_violations"] or
            checks["cross_domain_same_video_violations"] or
            checks["cross_domain_split_violations"]):
        raise ValueError("Cross-domain donor construction violated its domain/video/split rule")

    # Final reader/controls use source-train statistics. CV folds below create
    # their own stores from only the fitting rows, preventing validation leakage.
    standardised = standardize_features(data, train)
    store = Store(standardised, regions, c_dim, device)

    # ---- parameter matching: B/C/D must be able to match F before it is fitted ----
    v_dim = data["v"].shape[1]
    reference = parameter_count(build("F", v_dim, c_dim, regions, args.width, args.dim, args.depth))
    widths = capacity_matched_widths(v_dim, c_dim, regions, args.dim, args.width, args.depth, reference)
    widths["F"] = args.width
    widths["E"] = args.width
    checks["f_parameter_target"] = reference
    checks["matched_widths"] = widths

    # ---- fit every variant, every seed ----
    parameter_counts, selection, scores, adapters, weights_store = {}, {}, {}, {}, {}
    scores["A"] = data["s"].copy()
    splits = {}
    for seed in args.seeds:
        args.seed = seed
        fold_count = min(args.folds, len(np.unique(videos)))
        folds = GroupKFold(n_splits=fold_count)
        splits[str(seed)] = {"folds": [], "lambda_delta_choice": {}}
        for fitting, validation in folds.split(train_index, groups=videos):
            splits[str(seed)]["folds"].append({
                "fit_videos": sorted(np.unique(videos[fitting]).tolist()),
                "validation_videos": sorted(np.unique(videos[validation]).tolist()),
                "fit_rows": int(len(fitting)), "validation_rows": int(len(validation))})
        grid = [(lam, wd) for lam in args.lambdas for wd in args.weight_decays]
        for name in VARIANTS:
            losses = {}
            for lam, wd in grid:
                squared, elements = 0.0, 0
                for fitting, validation in folds.split(train_index, groups=videos):
                    fit_rows, val_rows = train_index[fitting], train_index[validation]
                    fold_store = Store(standardize_features(data, fit_rows), regions, c_dim, device)
                    model = build(name, v_dim, c_dim, regions, widths[name], args.dim, args.depth)
                    train_reader(model, name, fold_store, fit_rows, data["y"][fit_rows],
                                 {"lambda_delta": lam, "weight_decay": wd}, args, device)
                    model.eval()
                    val_batch = fold_store.batch(val_rows)
                    with torch.no_grad():
                        delta, _ = forward(model, name, val_batch)
                    target = torch.as_tensor(data["y"][val_rows], device=device).float()
                    error = nn.functional.binary_cross_entropy_with_logits(
                        val_batch[3] + delta, target)
                    squared += float(error) * len(val_rows)
                    elements += len(val_rows)
                losses[f"{lam}|{wd}"] = squared / elements
            best = min(grid, key=lambda pair: losses[f"{pair[0]}|{pair[1]}"])
            selection[f"{name}|{seed}"] = {
                "lambda_delta": best[0], "weight_decay": best[1],
                "validation_bce": losses[f"{best[0]}|{best[1]}"], "grid": losses}
            splits[str(seed)]["lambda_delta_choice"][name] = {
                "lambda_delta": best[0], "weight_decay": best[1]}
            model = build(name, v_dim, c_dim, regions, widths[name], args.dim, args.depth)
            if name not in parameter_counts:
                parameter_counts[name] = parameter_count(model)
            train_reader(model, name, store, train_index, data["y"][train_index],
                         {"lambda_delta": best[0], "weight_decay": best[1]}, args, device)
            predicted, weights = score_reader(model, name, store, np.arange(len(data["y"])))
            key = f"{name}|{seed}"
            scores[key] = predicted
            adapters[key] = model
            if weights is not None:
                weights_store[key] = weights
            torch.save(model.state_dict(), output / f"adapter_{name}_{seed}.pt")

    # ---- controls on the trained readers ----
    region_mean = standardised["ci"].reshape(len(data["y"]), -1)[train].mean(axis=0)
    region_std = standardised["ci"].reshape(len(data["y"]), -1)[train].std(axis=0)
    donor_features = {"v": standardised["v"], "c": standardised["c"],
                      "ci": standardised["ci"][np.where(usable, donors, np.arange(len(usable)))],
                      "s": standardised["s"]}
    donor_store = Store(donor_features, regions, c_dim, device)
    cross_donor_features = {
        "v": standardised["v"], "c": standardised["c"],
        "ci": standardised["ci"][np.where(
            cross_usable, cross_domain_donors, np.arange(len(cross_usable)))],
        "s": standardised["s"],
    }
    cross_donor_store = Store(cross_donor_features, regions, c_dim, device)
    controls = {
        "donors": donors, "donor_usable": usable,
        "cross_domain_donors": cross_domain_donors,
        "cross_domain_donor_usable": cross_usable,
    }
    for seed in args.seeds:
        args.seed = seed
        key = f"F|{seed}"
        chosen = selection[key]
        hyper = {"lambda_delta": chosen["lambda_delta"], "weight_decay": chosen["weight_decay"]}
        model = adapters[key]
        scores[f"F_donor|{seed}"], _ = score_reader(model, "F", donor_store, np.arange(len(data["y"])))
        scores[f"F_cross_domain|{seed}"], _ = score_reader(
            model, "F", cross_donor_store, np.arange(len(data["y"])))
        noise_rng = np.random.default_rng(seed)
        for repeat in range(args.repeats):
            noise = (noise_rng.normal(size=(len(data["y"]), regions * c_dim)).astype(np.float32)
                     * region_std + region_mean).reshape(len(data["y"]), regions, c_dim)
            noise_features = {"v": standardised["v"], "c": standardised["c"], "ci": noise,
                              "s": standardised["s"]}
            noise_store = Store(noise_features, regions, c_dim, device)
            scores[f"F_noise_fixed|{seed}|{repeat}"], _ = score_reader(
                model, "F", noise_store, np.arange(len(data["y"])))
            retrained = build("F", v_dim, c_dim, regions, widths["F"], args.dim, args.depth)
            train_reader(retrained, "F", noise_store, train_index, data["y"][train_index],
                         hyper, args, device)
            scores[f"F_noise_retrain|{seed}|{repeat}"], _ = score_reader(
                retrained, "F", noise_store, np.arange(len(data["y"])))
            torch.save(retrained.state_dict(), output / f"adapter_F_noise_{seed}_{repeat}.pt")
            controls[f"noise|{seed}|{repeat}"] = noise.reshape(len(data["y"]), -1)

    # ---- persist before the (slow) bootstrap statistics ----
    identity = {key: data[key] for key in ("row_id", "path", "domain", "split", "video_id", "y")}
    np.savez_compressed(output / "features.npz", **identity, s_V=data["s"], V=data["v"],
                        C=data["c"], C_regions=data["ci"])
    np.savez_compressed(output / "scores.npz", **identity, **scores)
    np.savez_compressed(output / "controls.npz", **controls)
    if weights_store:
        np.savez_compressed(output / "region_weights.npz", **weights_store)

    report = {}
    for domain in np.unique(data["domain"][evaluation]):
        mask = evaluation & (data["domain"] == domain)
        entry = {"A": summarize(data, scores["A"], scores["A"], mask, args.bootstrap, args.seeds[0])}
        for name in VARIANTS:
            # Average the full score vectors across seeds; summarize applies the mask.
            per_seed = [scores[f"{name}|{seed}"] for seed in args.seeds]
            entry[f"{name}|mean"] = summarize(
                data, scores["A"], np.mean(per_seed, axis=0), mask, args.bootstrap, args.seeds[0])
            for seed in args.seeds:
                entry[f"{name}|{seed}"] = summarize(
                    data, scores["A"], scores[f"{name}|{seed}"], mask, args.bootstrap, seed)
        for seed in args.seeds:
            entry[f"F_vs_E|{seed}"] = summarize(
                data, scores[f"E|{seed}"], scores[f"F|{seed}"], mask, args.bootstrap, seed)
            entry[f"F_donor|{seed}"] = summarize(
                data, scores["A"], scores[f"F_donor|{seed}"], mask, args.bootstrap, seed)
            entry[f"F_cross_domain|{seed}"] = summarize(
                data, scores["A"], scores[f"F_cross_domain|{seed}"], mask,
                args.bootstrap, seed)
            entry[f"F_noise_fixed|{seed}"] = summarize(
                data, scores["A"], scores[f"F_noise_fixed|{seed}|0"], mask, args.bootstrap, seed)
            entry[f"F_noise_retrain|{seed}"] = summarize(
                data, scores["A"], scores[f"F_noise_retrain|{seed}|0"], mask, args.bootstrap, seed)
        diagnostics = {}
        for name in ("F|mean", *[f"F|{seed}" for seed in args.seeds],
                     *[f"F_donor|{seed}" for seed in args.seeds],
                     *[f"F_cross_domain|{seed}" for seed in args.seeds]):
            candidate = (np.mean([scores[f"F|{seed}"] for seed in args.seeds], axis=0)
                         if name == "F|mean" else scores[name])
            diagnostics[name] = correction_diagnostics(
                data["y"][mask], scores["A"][mask], candidate[mask])
        entry["correction_diagnostics"] = diagnostics
        report[domain] = entry

    metadata = {
        "arguments": vars(args),
        "input_config_sha256": hashlib.sha256((root / "config.json").read_bytes()).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "label": "1=fake", "s_v": "logits[:,0]-logits[:,1] (frozen ViT decision)",
        "checks": checks, "parameter_counts": parameter_counts, "selection": selection,
        "region_position_control": (
            "SKIPPED by design: F is permutation invariant (query from V only, regions enter via a "
            "softmax-weighted sum), so region-order shuffling cannot change its output and would not "
            "be a valid control (spec section 4.6)."),
        "domain_conditional_controls": {
            "same_domain_donor": "same domain/split, different video; tests current-image dependence",
            "cross_domain_donor": "different domain, same split, different video; tests domain compatibility",
            "correction_diagnostics": "per-domain delta AUC, rescue/harm rates and correction magnitude",
        },
        "standardization": (
            "final reader uses source-train statistics; each GroupKFold fit/validation selection "
            "uses statistics fitted on the fit fold only"),
        "interpretation": "frozen-representation diagnostic; no architecture training",
    }
    (output / "config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output / "split.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    (output / "COMPLETE").write_text("Analysis completed\n", encoding="utf-8")
    print(f"Saved {output.resolve()}")
    print(f"s_V source-train AUC={checks['s_v_auc_source_train']:.4f} "
          f"params={parameter_counts} rows_without_donor={checks['rows_without_donor']} "
          f"rows_without_cross_domain_donor={checks['rows_without_cross_domain_donor']}")


if __name__ == "__main__":
    main()
