"""G27: source-only ViT baseline and controlled CLIP image region readout."""

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import sklearn
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from torch import nn

from analyze import fit_reader, save_reader, summarize, validate_splits
from conditional_clip import Store, load_local, normalize_regions, parameter_count, split_donors


VARIANTS = ("V_ONLY", "POOLED", "MEAN", "ADAPTIVE")


class RegionCorrection(nn.Module):
    def __init__(self, visual_dim, clip_dim, dim, width, mode):
        super().__init__()
        self.mode = mode
        self.visual_norm = nn.LayerNorm(visual_dim)
        self.region_norm = nn.LayerNorm(clip_dim)
        self.query = nn.Linear(visual_dim, dim)
        self.project = nn.Linear(clip_dim, dim)
        self.head = nn.Sequential(
            nn.Linear(visual_dim + 2 * dim + 1, width), nn.ReLU(), nn.Linear(width, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.scale = dim ** -0.5

    def forward(self, visual, pooled_clip, regions, baseline):
        visual = self.visual_norm(visual)
        query = self.query(visual)
        tokens = pooled_clip.unsqueeze(1) if self.mode == "POOLED" else regions
        hidden = self.project(self.region_norm(tokens))
        if self.mode == "ADAPTIVE":
            weights = torch.softmax((hidden * query.unsqueeze(1)).sum(-1) * self.scale, dim=1)
        else:
            weights = torch.full_like(hidden[:, :, 0], 1.0 / hidden.shape[1])
        pooled = (weights.unsqueeze(-1) * hidden).sum(1)
        correction = self.head(torch.cat([visual, query, pooled, baseline.unsqueeze(1)], dim=1))
        return correction.squeeze(1), weights


class VisualCorrection(nn.Module):
    def __init__(self, visual_dim, width):
        super().__init__()
        self.norm = nn.LayerNorm(visual_dim)
        self.head = nn.Sequential(nn.Linear(visual_dim + 1, width), nn.ReLU(), nn.Linear(width, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, visual, pooled_clip, regions, baseline):
        correction = self.head(torch.cat([self.norm(visual), baseline.unsqueeze(1)], dim=1))
        return correction.squeeze(1), None


def build_reader(name, visual_dim, clip_dim, args, seed):
    torch.manual_seed(seed)
    if name != "V_ONLY":
        return RegionCorrection(visual_dim, clip_dim, args.dim, args.width, name)
    target = parameter_count(RegionCorrection(visual_dim, clip_dim, args.dim, args.width, "ADAPTIVE"))
    width = max(1, int(np.ceil((target - 2 * visual_dim - 1) / (visual_dim + 3))))
    return VisualCorrection(visual_dim, width)


def prepare_store(data, fit_rows, reader):
    scalers = {name: StandardScaler().fit(data[name][fit_rows]) for name in ("V", "C")}
    flat = data["regions"].reshape(len(data["y"]), -1)
    region_scaler = StandardScaler().fit(flat[fit_rows])
    features = {
        "v": scalers["V"].transform(data["V"]).astype(np.float32),
        "c": scalers["C"].transform(data["C"]).astype(np.float32),
        "ci": region_scaler.transform(flat).astype(np.float32).reshape(data["regions"].shape),
        "s": reader.decision_function(data["V"]).astype(np.float32),
    }
    parameters = {}
    for name, scaler in {**scalers, "regions": region_scaler}.items():
        parameters[name + "_mean"] = scaler.mean_
        parameters[name + "_scale"] = scaler.scale_
    return Store(features, data["regions"].shape[1], data["regions"].shape[2], "cpu"), parameters


def fit_correction(model, store, rows, labels, hyper, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=hyper[1])
    batch = store.batch(rows)
    target = torch.as_tensor(labels[rows], dtype=torch.float32)
    model.train()
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        correction, weights = model(*batch)
        loss = nn.functional.binary_cross_entropy_with_logits(batch[3] + correction, target)
        loss = loss + hyper[0] * correction.square().mean()
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite correction loss")
        loss.backward()
        optimizer.step()
    return model


def predict(model, store):
    model.eval()
    with torch.no_grad():
        correction, weights = model(*store.batch(np.arange(len(store.s))))
    return store.s + correction.numpy(), None if weights is None else weights.numpy()


def calibrate(baseline, labels):
    baseline = torch.as_tensor(baseline, dtype=torch.float64)
    target = torch.as_tensor(labels, dtype=torch.float64)
    log_scale = nn.Parameter(torch.zeros((), dtype=torch.float64))
    bias = nn.Parameter(torch.zeros((), dtype=torch.float64))
    optimizer = torch.optim.LBFGS([log_scale, bias], max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(log_scale.exp() * baseline + bias, target)
        loss.backward()
        return loss

    optimizer.step(closure)
    scale = float(log_scale.detach().exp())
    offset = float(bias.detach())
    if not np.isfinite([scale, offset]).all() or scale <= 0:
        raise ValueError("Invalid positive-affine calibration")
    return scale, offset


def macro_comparison(data, baseline, candidate, domains, bootstrap, seed, cache=None):
    blocks, masks = {}, {}
    for domain in domains:
        mask = (data["domain"] == domain) & (data["split"] == "test")
        rows = np.flatnonzero(mask)
        if len(np.unique(data["y"][rows])) != 2:
            raise ValueError(f"Both test classes required for macro domain {domain}")
        videos = np.unique(data["video_id"][rows])
        if len(videos) < 2:
            raise ValueError(f"At least two videos required for macro domain {domain}")
        blocks[domain] = [rows[data["video_id"][rows] == video] for video in videos]
        masks[domain] = rows

    def average_auc(score, selected):
        return float(np.mean([roc_auc_score(data["y"][rows], score[rows]) for rows in selected.values()]))

    cache = {} if cache is None else cache

    def estimates(score):
        key = (id(score), tuple(domains), bootstrap, seed)
        if key not in cache:
            rng = np.random.default_rng(seed)
            replicas = []
            for repeat in range(bootstrap):
                selected = {domain: np.concatenate([domain_blocks[index] for index in
                            rng.integers(0, len(domain_blocks), len(domain_blocks))])
                            for domain, domain_blocks in blocks.items()}
                if all(len(np.unique(data["y"][rows])) == 2 for rows in selected.values()):
                    replicas.append(average_auc(score, selected))
            cache[key] = (average_auc(score, masks), np.asarray(replicas))
        return cache[key]

    base, baseline_replicas = estimates(baseline)
    proposed, candidate_replicas = estimates(candidate)
    differences = candidate_replicas - baseline_replicas
    interval = (np.quantile(differences, [0.025, 0.975]).tolist()
                if len(differences) >= max(10, bootstrap // 2) else None)
    return {"domains": list(domains), "baseline_macro_auc": base, "macro_auc": proposed,
            "delta_macro_auc": proposed - base, "ci95": interval,
            "valid_bootstraps": len(differences), "estimator": "mean of per-domain AUCs, not pooled AUC"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source", default="ffpp")
    parser.add_argument("--primary-domains", nargs="+", default=["cd2", "dfdcp", "wild"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[20261010, 20261011, 20261012])
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0, 1e-3, 1e-2, 1e-1])
    parser.add_argument("--weight-decays", type=float, nargs="+", default=[1e-4, 1e-3])
    parser.add_argument("--baseline-c", type=float, default=1e-3)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if (min(args.epochs, args.width, args.dim, args.repeats, args.threads) < 1 or args.folds < 2
            or args.bootstrap < 0 or args.lr <= 0 or args.baseline_c <= 0
            or min(args.lambdas) < 0 or min(args.weight_decays) < 0):
        parser.error("Invalid training/statistical settings")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.primary_domains)) != len(args.primary_domains):
        parser.error("Seeds and primary domains must be unique")
    if args.source in args.primary_domains or "ffiw" in args.primary_domains:
        parser.error("Source and single-video FFIW are not primary target domains")
    if "cd1" in args.primary_domains and "cd2" in args.primary_domains:
        parser.error("CD1 and CD2 cannot both enter the primary macro estimate")
    torch.set_num_threads(args.threads)
    root = Path(args.input)
    if not (root / "COMPLETE").exists():
        raise ValueError("Extraction incomplete")
    extraction = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if extraction.get("label") != "1=fake, 0=real" or not extraction["arguments"].get("save_regions"):
        raise ValueError("Expected region extraction with label 1=fake, 0=real")
    data = load_local(root)
    train = validate_splits(data, args.source)
    for key in ("V", "C", "clip_regions", "logits"):
        if not np.isfinite(data[key]).all():
            raise ValueError(f"Nonfinite {key}")
    if data["C"].shape[1] != data["clip_regions"].shape[2]:
        raise ValueError("Pooled/region CLIP dimensions differ")
    data["regions"] = normalize_regions(data["clip_regions"])
    train_rows = np.flatnonzero(train)
    fold_count = min(args.folds, len(np.unique(data["video_id"][train])))
    if fold_count < 2:
        raise ValueError("At least two source video groups required")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    metadata = {"arguments": vars(args), "status": "running", "label": "1=fake, 0=real",
                "baseline": "source-trained StandardScaler+LogisticRegression on V only; not detector logits",
                "versions": {"numpy": np.__version__, "sklearn": sklearn.__version__, "torch": str(torch.__version__)},
                "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "input_config_sha256": hashlib.sha256((root / "config.json").read_bytes()).hexdigest(),
                "helper_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                  for name in ("analyze.py", "conditional_clip.py")}}
    (output / "config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    pure_reader = fit_reader(data["V"], data["y"], train, args.baseline_c, args.seeds[0])
    save_reader(output / "reader_V_ONLY.npz", pure_reader)
    store, scalers = prepare_store(data, train_rows, pure_reader)
    np.savez_compressed(output / "scalers.npz", **scalers)
    scores = {"V_BASE": store.s.copy(),
              "DETECTOR_REFERENCE": data["logits"][:, 0] - data["logits"][:, 1]}
    macro_comparison(data, scores["V_BASE"], scores["V_BASE"], args.primary_domains, 0, args.seeds[0])
    concat = np.concatenate([data["V"], data["C"]], axis=1)
    concat_reader = fit_reader(concat, data["y"], train, args.baseline_c, args.seeds[0])
    save_reader(output / "reader_V_C_LINEAR.npz", concat_reader)
    scores["V_C_LINEAR"] = concat_reader.decision_function(concat)
    local_concat = np.concatenate([data["V"], data["regions"].reshape(len(train), -1)], axis=1)
    local_reader = fit_reader(local_concat, data["y"], train, args.baseline_c, args.seeds[0])
    save_reader(output / "reader_V_REGIONS_LINEAR.npz", local_reader)
    scores["V_REGIONS_LINEAR"] = local_reader.decision_function(local_concat)
    folds, split_record = [], []
    out_of_fold = np.full(len(train_rows), np.nan)
    splitter = GroupKFold(n_splits=fold_count)
    for fitting, validation in splitter.split(train_rows, groups=data["video_id"][train]):
        fit_rows, val_rows = train_rows[fitting], train_rows[validation]
        fold_mask = np.zeros(len(train), dtype=bool)
        fold_mask[fit_rows] = True
        fold_reader = fit_reader(data["V"], data["y"], fold_mask, args.baseline_c, args.seeds[0])
        fold_store, fold_scalers = prepare_store(data, fit_rows, fold_reader)
        out_of_fold[validation] = fold_reader.decision_function(data["V"][val_rows])
        folds.append((fit_rows, val_rows, fold_store))
        split_record.append({"fit_row_id": data["row_id"][fit_rows].tolist(),
                             "validation_row_id": data["row_id"][val_rows].tolist(),
                             "fit_video_id": np.unique(data["video_id"][fit_rows]).tolist(),
                             "validation_video_id": np.unique(data["video_id"][val_rows]).tolist()})
    scale, offset = calibrate(out_of_fold, data["y"][train])
    scores["CALIBRATED"] = scale * scores["V_BASE"] + offset
    metadata["calibration"] = {"scale": scale, "bias": offset, "fit": "source out-of-fold V scores"}
    np.savez_compressed(output / "baseline_oof.npz", row_id=data["row_id"][train], scores=out_of_fold)
    (output / "split.json").write_text(json.dumps(split_record, indent=2), encoding="utf-8")
    selection, counts, weights_saved = {}, {}, {}
    models = {}
    grid = [(lam, decay) for lam in args.lambdas for decay in args.weight_decays]
    for seed in args.seeds:
        for name in VARIANTS:
            losses = {}
            for hyper in grid:
                total, count = 0.0, 0
                for fold_index, (fit_rows, val_rows, fold_store) in enumerate(folds):
                    model = build_reader(name, data["V"].shape[1], data["C"].shape[1], args, seed + fold_index)
                    fit_correction(model, fold_store, fit_rows, data["y"], hyper, args)
                    with torch.no_grad():
                        batch = fold_store.batch(val_rows)
                        correction, weights = model(*batch)
                        error = nn.functional.binary_cross_entropy_with_logits(
                            batch[3] + correction, torch.as_tensor(data["y"][val_rows], dtype=torch.float32))
                    total += float(error) * len(val_rows)
                    count += len(val_rows)
                losses[str(hyper)] = total / count
            best = min(grid, key=lambda hyper: losses[str(hyper)])
            key = f"{name}|{seed}"
            selection[key] = {"lambda_delta": best[0], "weight_decay": best[1], "cv_bce": losses[str(best)], "grid": losses}
            model = build_reader(name, data["V"].shape[1], data["C"].shape[1], args, seed)
            counts[name] = parameter_count(model)
            fit_correction(model, store, train_rows, data["y"], best, args)
            scores[key], weights = predict(model, store)
            models[key] = model
            if weights is not None:
                weights_saved[key] = weights
            torch.save(model.state_dict(), output / f"adapter_{name}_{seed}.pt")
            print(f"Fitted {key}: source CV BCE={losses[str(best)]:.5f}", flush=True)
    controls = {"noise_mean": store.ci[train].mean(axis=0),
                "noise_std": store.ci[train].std(axis=0)}
    for seed in args.seeds:
        model = models[f"ADAPTIVE|{seed}"]
        chosen = selection[f"ADAPTIVE|{seed}"]
        hyper = (chosen["lambda_delta"], chosen["weight_decay"])
        for repeat in range(args.repeats):
            rng = np.random.default_rng(seed + 10000 + repeat)
            donors = split_donors(data["video_id"], data["domain"], data["split"], rng)
            usable = donors >= 0
            donor_regions = store.ci[np.where(usable, donors, np.arange(len(donors)))]
            donor_store = Store({"v": store.v, "c": store.c, "ci": donor_regions, "s": store.s},
                                store.regions, store.c_dim, "cpu")
            scores[f"DONOR|{seed}|{repeat}"], weights = predict(model, donor_store)
            controls[f"donors|{seed}|{repeat}"] = donors
            controls[f"donor_usable|{seed}|{repeat}"] = usable
            noise_seed = seed + 20000 + repeat
            noise = (np.random.default_rng(noise_seed).normal(size=store.ci.shape).astype(np.float32)
                     * controls["noise_std"] + controls["noise_mean"])
            noise_store = Store({"v": store.v, "c": store.c, "ci": noise, "s": store.s},
                                store.regions, store.c_dim, "cpu")
            scores[f"NOISE_FIXED|{seed}|{repeat}"], weights = predict(model, noise_store)
            controls[f"noise_seed|{seed}|{repeat}"] = np.array(noise_seed)
            noise_model = build_reader("ADAPTIVE", data["V"].shape[1], data["C"].shape[1], args, seed)
            fit_correction(noise_model, noise_store, train_rows, data["y"], hyper, args)
            scores[f"NOISE_RETRAIN|{seed}|{repeat}"], weights = predict(noise_model, noise_store)
            torch.save(noise_model.state_dict(), output / f"adapter_noise_{seed}_{repeat}.pt")
    for name in VARIANTS:
        scores[f"{name}|ensemble"] = np.mean([scores[f"{name}|{seed}"] for seed in args.seeds], axis=0)
    identity = {key: data[key] for key in ("row_id", "path", "domain", "split", "video_id", "y")}
    np.savez_compressed(output / "scores.npz", **identity, **scores)
    np.savez_compressed(output / "features.npz", **identity, V=data["V"], C=data["C"], C_regions=data["regions"])
    np.savez_compressed(output / "controls.npz", **controls)
    np.savez_compressed(output / "region_weights.npz", **weights_saved)
    metadata.update({"selection": selection, "parameter_counts": counts,
                     "noise_generator": "default_rng(seed).normal(size=regions.shape).astype(float32)*std+mean",
                     "status": "statistics_pending"})
    (output / "config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    comparisons = {key + "_vs_V_BASE": ("V_BASE", key) for key in scores}
    for seed in args.seeds:
        candidate = f"ADAPTIVE|{seed}"
        for baseline in (f"V_ONLY|{seed}", f"MEAN|{seed}", f"POOLED|{seed}",
                         "V_C_LINEAR", "V_REGIONS_LINEAR", "CALIBRATED"):
            comparisons[candidate + "_vs_" + baseline] = (baseline, candidate)
        for repeat in range(args.repeats):
            for control in ("DONOR", "NOISE_FIXED", "NOISE_RETRAIN"):
                baseline = f"{control}|{seed}|{repeat}"
                comparisons[candidate + "_vs_" + baseline] = (baseline, candidate)
    for baseline in ("V_ONLY|ensemble", "MEAN|ensemble", "POOLED|ensemble"):
        comparisons["ADAPTIVE|ensemble_vs_" + baseline] = (baseline, "ADAPTIVE|ensemble")
    report = {"per_domain": {}, "primary_macro": {}, "single_seed_auc": {}}
    for domain in np.unique(data["domain"][data["split"] == "test"]):
        mask = (data["domain"] == domain) & (data["split"] == "test")
        report["per_domain"][domain] = {}
        for name, (base_key, candidate_key) in comparisons.items():
            comparison_mask = mask.copy()
            for key in (base_key, candidate_key):
                if key.startswith("DONOR|"):
                    comparison_mask &= controls["donor_usable|" + key.split("|", 1)[1]]
            if not comparison_mask.any():
                report["per_domain"][domain][name] = {"status": "no usable donor rows"}
            else:
                report["per_domain"][domain][name] = summarize(
                    data, scores[base_key], scores[candidate_key], comparison_mask, args.bootstrap, args.seeds[0])
        report["single_seed_auc"][domain] = {}
        for name in VARIANTS:
            values = [roc_auc_score(data["y"][mask], scores[f"{name}|{seed}"][mask]) for seed in args.seeds]
            report["single_seed_auc"][domain][name] = {
                "values": values, "mean": float(np.mean(values)), "std": float(np.std(values)),
                "ensemble_auc": float(roc_auc_score(data["y"][mask], scores[f"{name}|ensemble"][mask]))}
    macro_cache = {}
    for name, (base_key, candidate_key) in comparisons.items():
        if base_key.startswith("DONOR|") or candidate_key.startswith("DONOR|"):
            usable = controls["donor_usable|" + (base_key if base_key.startswith("DONOR|") else candidate_key).split("|", 1)[1]]
            if not usable[np.isin(data["domain"], args.primary_domains) & (data["split"] == "test")].all():
                report["primary_macro"][name] = {"status": "incomplete donor coverage"}
                continue
        report["primary_macro"][name] = macro_comparison(
            data, scores[base_key], scores[candidate_key], args.primary_domains,
            args.bootstrap, args.seeds[0], macro_cache)
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    metadata["status"] = "complete"
    (output / "config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output / "COMPLETE").write_text("G27 completed\n", encoding="utf-8")
    print(f"Saved {output.resolve()}")


if __name__ == "__main__":
    main()
