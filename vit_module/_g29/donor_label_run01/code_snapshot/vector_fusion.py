"""G29: frozen CLS vectors, matched normalization and scalar convex fusion."""

import argparse
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import sklearn
import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn

from analyze import fit_reader, save_reader, summarize, validate_splits
from clip_readout import cell_seed, evaluation_subset, file_digest, grouped_folds, write_json
from conditional_clip import parameter_count, split_donors
from pure_vit_clip import macro_comparison


VARIANTS = ("V_MATCHED", "CONCAT", "FIXED", "LEARNED")
FUSIONS = ("FIXED", "LEARNED")
IDENTITY = ("row_id", "path", "domain", "split", "video_id", "y")


class VectorFusion(nn.Module):
    def __init__(self, visual_dim, clip_dim, mode, seed):
        super().__init__()
        if mode not in VARIANTS:
            raise ValueError(f"Unknown vector fusion mode: {mode}")
        self.mode = mode
        self.visual_norm = nn.LayerNorm(visual_dim, elementwise_affine=False)
        self.clip_norm = nn.LayerNorm(visual_dim if mode in FUSIONS else clip_dim,
                                      elementwise_affine=False)
        if mode in FUSIONS:
            torch.manual_seed(seed)
            self.clip_projection = nn.Linear(clip_dim, visual_dim)
            if mode == "LEARNED":
                self.alpha_logit = nn.Parameter(torch.zeros(()))
        torch.manual_seed(seed + 100000)
        self.head = nn.Linear(visual_dim + clip_dim if mode == "CONCAT" else visual_dim, 1)

    def alpha(self):
        if self.mode == "LEARNED":
            return self.alpha_logit.sigmoid()
        return self.head.weight.new_tensor(0.5)

    def forward(self, visual, clip, alpha_override=None):
        visual = self.visual_norm(visual)
        if self.mode == "V_MATCHED":
            combined = visual
            clip_value = None
        elif self.mode == "CONCAT":
            clip_value = self.clip_norm(clip)
            combined = torch.cat([visual, clip_value], dim=-1)
        else:
            clip_value = self.clip_norm(self.clip_projection(clip))
            alpha = self.alpha() if alpha_override is None else self.head.weight.new_tensor(alpha_override)
            combined = alpha * clip_value + (1 - alpha) * visual
        return self.head(combined).squeeze(-1), combined, visual, clip_value


def load_vectors(root):
    shards = sorted(root.glob("clean_*.npz"))
    if not shards:
        raise ValueError("Missing clean_*.npz; summary JSON cannot replace frozen features")
    columns = {key: [] for key in (*IDENTITY, "V", "C")}
    for path in shards:
        with np.load(path, allow_pickle=False) as shard:
            for key in columns:
                columns[key].append(shard[key])
    data = {key: np.concatenate(values) for key, values in columns.items()}
    if not np.array_equal(data["row_id"], np.arange(len(data["y"]))):
        raise ValueError("Missing, duplicated or reordered extraction rows")
    for name in ("V", "C"):
        if data[name].ndim != 2 or data[name].shape[1] < 2 or not np.isfinite(data[name]).all():
            raise ValueError(f"Invalid frozen vector: {name}")
    if any(len(value) != len(data["y"]) for value in data.values()):
        raise ValueError("Inconsistent vector/identity lengths")
    return data


def standardize(data, rows):
    features, parameters = {}, {}
    for name in ("V", "C"):
        scaler = StandardScaler().fit(data[name][rows])
        features[name] = scaler.transform(data[name]).astype(np.float32)
        parameters[name + "_mean"] = scaler.mean_
        parameters[name + "_scale"] = scaler.scale_
    return features, parameters


def train_model(model, features, rows, labels, decay, args, validation_rows=None):
    regularized = [parameter for name, parameter in model.named_parameters() if name != "alpha_logit"]
    groups = [{"params": regularized, "weight_decay": decay}]
    if model.mode == "LEARNED":
        groups.append({"params": [model.alpha_logit], "weight_decay": 0.0})
    optimizer = torch.optim.Adam(groups, lr=args.lr)
    visual, clip = (torch.as_tensor(features[name][rows]) for name in ("V", "C"))
    target = torch.as_tensor(labels[rows], dtype=torch.float32)
    trace = []
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        logits, combined, visual_value, clip_value = model(visual, clip)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, target)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite fusion loss")
        loss.backward()
        optimizer.step()
        if epoch == 0 or (epoch + 1) % args.trace_every == 0 or epoch + 1 == args.epochs:
            model.eval()
            record = {"epoch": epoch + 1}
            if model.mode in FUSIONS:
                record["alpha"] = float(model.alpha().detach())
            with torch.no_grad():
                for label, selected in (("fit", rows), ("validation", validation_rows)):
                    if selected is None:
                        continue
                    score = model(*(torch.as_tensor(features[name][selected]) for name in ("V", "C")))[0]
                    record[label + "_bce"] = float(nn.functional.binary_cross_entropy_with_logits(
                        score, torch.as_tensor(labels[selected], dtype=torch.float32)))
            trace.append(record)
    return model, trace


def predict(model, features, alpha_override=None):
    model.eval()
    with torch.no_grad():
        score, combined, visual, clip = model(
            *(torch.as_tensor(features[name]) for name in ("V", "C")), alpha_override=alpha_override)
        diagnostics = {}
        if clip is not None and model.mode in FUSIONS:
            alpha = float(model.alpha()) if alpha_override is None else alpha_override
            diagnostics = {
                "projected_cosine": nn.functional.cosine_similarity(visual, clip, dim=-1).numpy(),
                "visual_norm": visual.norm(dim=-1).numpy(), "clip_norm": clip.norm(dim=-1).numpy(),
                "combined_norm": combined.norm(dim=-1).numpy(),
                "visual_head_component": ((1 - alpha) * (visual @ model.head.weight[0])).numpy(),
                "clip_head_component": (alpha * (clip @ model.head.weight[0])).numpy()}
    if not np.isfinite(score.numpy()).all():
        raise ValueError("Nonfinite fusion prediction")
    return score.numpy(), diagnostics


def clip_controls(features, data, train_rows, seed):
    mean = features["C"][train_rows].mean(axis=0)
    std = features["C"][train_rows].std(axis=0)
    donors = np.full(len(data["y"]), -1, dtype=int)
    noise = np.empty_like(features["C"])
    for domain in np.unique(data["domain"]):
        for split in np.unique(data["split"]):
            rows = np.flatnonzero((data["domain"] == domain) & (data["split"] == split))
            if not len(rows):
                continue
            rng = np.random.default_rng(cell_seed(seed, domain, split))
            local = split_donors(data["video_id"][rows], data["domain"][rows], data["split"][rows], rng)
            usable = local >= 0
            donors[rows[usable]] = rows[local[usable]]
            noise_rng = np.random.default_rng(cell_seed(seed + 100000, domain, split))
            noise[rows] = noise_rng.normal(size=(len(rows), features["C"].shape[1])).astype(np.float32) * std + mean
    usable = donors >= 0
    donor_rows = np.where(usable, donors, np.arange(len(donors)))
    return ({"V": features["V"], "C": features["C"][donor_rows]},
            {"V": features["V"], "C": noise},
            {"donors": donors, "usable": usable, "noise_mean": mean, "noise_std": std,
             "noise_seed": np.array(seed + 100000), "donor_seed": np.array(seed)})


def report_results(data, scores, comparisons, donor_masks, args):
    report = {"per_domain": {}, "primary_macro": {}, "single_seed_auc": {}, "primary_single_seed_auc": {},
              "comparison_roles": {}, "primary_questions": {
                  "clip_increment": [f"LEARNED|{seed}_vs_V_MATCHED|{seed}" for seed in args.seeds],
                  "learned_weight_increment": [f"LEARNED|{seed}_vs_FIXED|{seed}" for seed in args.seeds],
                  "over_concatenation": [f"LEARNED|{seed}_vs_CONCAT|{seed}" for seed in args.seeds]}}
    cache = {}
    for key, (base, proposed, role) in comparisons.items():
        report["comparison_roles"][key] = role
        usable = np.ones(len(data["y"]), dtype=bool)
        for name in (base, proposed):
            if name in donor_masks:
                usable &= donor_masks[name]
        for domain in np.unique(data["domain"][data["split"] == "test"]):
            mask = (data["domain"] == domain) & (data["split"] == "test") & usable
            report["per_domain"].setdefault(domain, {})[key] = (
                summarize(data, scores[base], scores[proposed], mask, args.bootstrap, args.seeds[0])
                if mask.any() else {"status": "no usable donor rows"})
        primary = (data["split"] == "test") & np.isin(data["domain"], args.primary_domains)
        report["primary_macro"][key] = (macro_comparison(
            data, scores[base], scores[proposed], args.primary_domains, args.bootstrap, args.seeds[0], cache)
            if usable[primary].all() else {"status": "incomplete donor coverage"})
    for name in VARIANTS:
        macro_values = []
        for seed in args.seeds:
            values = []
            for domain in np.unique(data["domain"][data["split"] == "test"]):
                mask = (data["domain"] == domain) & (data["split"] == "test")
                auc = float(roc_auc_score(data["y"][mask], scores[f"{name}|{seed}"][mask]))
                report["single_seed_auc"].setdefault(domain, {}).setdefault(name, {"values": []})["values"].append(auc)
                if domain in args.primary_domains:
                    values.append(auc)
            macro_values.append(float(np.mean(values)))
        for domain in report["single_seed_auc"]:
            result = report["single_seed_auc"][domain][name]
            mask = (data["domain"] == domain) & (data["split"] == "test")
            result.update({"mean": float(np.mean(result["values"])), "std": float(np.std(result["values"])),
                           "ensemble_auc": float(roc_auc_score(data["y"][mask], scores[f"{name}|ensemble"][mask]))})
        ensemble_values = [report["single_seed_auc"][domain][name]["ensemble_auc"] for domain in args.primary_domains]
        report["primary_single_seed_auc"][name] = {"values": macro_values, "mean": float(np.mean(macro_values)),
                                                  "std": float(np.std(macro_values)),
                                                  "ensemble_auc": float(np.mean(ensemble_values))}
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source", default="ffpp")
    parser.add_argument("--primary-domains", nargs="+", default=["cd2", "dfdcp", "wild"])
    parser.add_argument("--evaluation-manifest")
    parser.add_argument("--evaluation-status", choices=["exploratory", "new-holdout"], default="exploratory")
    parser.add_argument("--seeds", type=int, nargs="+", default=[20261030, 20261031, 20261032])
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--trace-every", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decays", type=float, nargs="+", default=[0.0001, 0.001, 0.01])
    parser.add_argument("--baseline-c", type=float, default=0.001)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if (min(args.epochs, args.trace_every, args.threads, args.repeats) < 1 or args.folds < 2
            or args.bootstrap < 0 or min(args.seeds) < 0 or min(args.lr, args.baseline_c) <= 0
            or min(args.weight_decays) < 0 or not np.isfinite([args.lr, args.baseline_c, *args.weight_decays]).all()):
        parser.error("Invalid training/statistical settings")
    for values in (args.seeds, args.primary_domains, args.weight_decays):
        if len(values) != len(set(values)):
            parser.error("Duplicate settings are not allowed")
    if (args.source in args.primary_domains or "ffiw" in args.primary_domains
            or {"cd1", "cd2"}.issubset(args.primary_domains)):
        parser.error("Source/FFIW/overlapping CD1+CD2 cannot be primary domains")
    if args.evaluation_status == "new-holdout" and not args.evaluation_manifest:
        parser.error("new-holdout requires a predeclared evaluation manifest")
    return args


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    root, output = Path(args.input), Path(args.output)
    if not (root / "COMPLETE").exists():
        raise ValueError("Extraction incomplete")
    extraction = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if extraction.get("label") != "1=fake, 0=real":
        raise ValueError("Expected extraction labels 1=fake, 0=real")
    data = evaluation_subset(load_vectors(root), args.source, args.evaluation_manifest)
    train = validate_splits(data, args.source)
    rows = np.flatnonzero(train)
    folds = grouped_folds(data, rows, args.folds)
    for domain in np.unique(data["domain"][data["split"] == "test"]):
        mask = (data["domain"] == domain) & (data["split"] == "test")
        if len(np.unique(data["y"][mask])) != 2:
            raise ValueError(f"Both test classes required: {domain}")
    placeholder = np.zeros(len(data["y"]))
    macro_comparison(data, placeholder, placeholder.copy(), args.primary_domains, 0, args.seeds[0])
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "code_snapshot"
    snapshot.mkdir()
    hashes = {}
    for filename in ("vector_fusion.py", "VECTOR_FUSION_EXPERIMENT.md", "clip_readout.py",
                     "pure_vit_clip.py", "conditional_clip.py", "analyze.py"):
        path = Path(__file__).with_name(filename)
        shutil.copy2(path, snapshot / filename)
        hashes[filename] = file_digest(path)
    shutil.copy2(root / "config.json", output / "input_config.json")
    metadata = {"arguments": vars(args), "status": "running", "label": "1=fake, 0=real",
                "feature_definition": {"V": "frozen ViT CLS", "C": "CLIP hidden_states[-2][:,0], 1024 in current project",
                                       "fusion_dim": data["V"].shape[1], "clip_dim": data["C"].shape[1]},
                "normalization": "source StandardScaler then non-affine LayerNorm; projected C normalized after Linear; no post-mixture normalization",
                "alpha": "global sigmoid(logit), initial 0.5; no gate weight decay; not an information contribution percentage",
                "training": "fresh classifier, not residual correction; encoders remain frozen",
                "noise_selection": "reuse each real-source-selected weight decay; not independently tuned",
                "checkpoint_selection": "inherited checkpoint; source-only reader selection does not remove prior FFIW selection",
                "code_sha256": hashes, "input_config_sha256": file_digest(root / "config.json"),
                "versions": {"numpy": np.__version__, "sklearn": sklearn.__version__, "torch": str(torch.__version__)}}
    if args.evaluation_manifest:
        shutil.copy2(args.evaluation_manifest, output / "evaluation_manifest.csv")
        metadata["evaluation_manifest_sha256"] = file_digest(Path(args.evaluation_manifest))
    write_json(output / "config.json", metadata)
    write_json(output / "input_hashes.json", {path.name: file_digest(path) for path in sorted(root.glob("clean_*.npz"))})
    features, scalers = standardize(data, rows)
    np.savez_compressed(output / "scalers.npz", **scalers)
    contexts, split_record = [], []
    for fold_index, (fit_rows, val_rows) in enumerate(folds):
        fold_features, fold_scalers = standardize(data, fit_rows)
        np.savez_compressed(output / f"fold{fold_index}_scalers.npz", **fold_scalers)
        contexts.append((fit_rows, val_rows, fold_features))
        split_record.append({"fit_row_id": data["row_id"][fit_rows].tolist(),
                             "validation_row_id": data["row_id"][val_rows].tolist(),
                             "fit_video_id": np.unique(data["video_id"][fit_rows]).tolist(),
                             "validation_video_id": np.unique(data["video_id"][val_rows]).tolist()})
    write_json(output / "split.json", split_record)
    scores, selection, models, traces, diagnostics, counts, alphas, comparisons = {}, {}, {}, {}, {}, {}, {}, {}

    def compare(base, candidate, role):
        comparisons[candidate + "_vs_" + base] = (base, candidate, role)

    for name, values in (("V_REFERENCE", data["V"]), ("V_C_REFERENCE", np.concatenate([data["V"], data["C"]], axis=1))):
        reader = fit_reader(values, data["y"], train, args.baseline_c, args.seeds[0])
        save_reader(output / f"reader_{name}.npz", reader)
        scores[name] = reader.decision_function(values)
    for seed in args.seeds:
        for name in VARIANTS:
            losses = []
            for hyper_index, decay in enumerate(args.weight_decays):
                total, count = 0.0, 0
                for fold_index, (fit_rows, val_rows, fold_features) in enumerate(contexts):
                    model = VectorFusion(data["V"].shape[1], data["C"].shape[1], name, seed + fold_index)
                    model, trace = train_model(model, fold_features, fit_rows, data["y"], decay, args, val_rows)
                    traces[f"CV|{name}|{seed}|h{hyper_index}|fold{fold_index}"] = trace
                    total += trace[-1]["validation_bce"] * len(val_rows)
                    count += len(val_rows)
                losses.append(total / count)
            chosen = int(np.argmin(losses))
            key = f"{name}|{seed}"
            selection[key] = {"weight_decay": args.weight_decays[chosen], "grid_index": chosen, "cv_bce": losses[chosen],
                              "grid_cv_bce": losses, "at_upper_decay": args.weight_decays[chosen] == max(args.weight_decays)}
            model = VectorFusion(data["V"].shape[1], data["C"].shape[1], name, seed)
            counts[name] = parameter_count(model)
            if name in FUSIONS:
                for hyper_index, decay in enumerate(args.weight_decays):
                    grid_key = f"GRID|{name}|{seed}|h{hyper_index}"
                    grid_model = VectorFusion(data["V"].shape[1], data["C"].shape[1], name, seed)
                    grid_model, traces[grid_key] = train_model(grid_model, features, rows, data["y"], decay, args)
                    scores[grid_key], values = predict(grid_model, features)
                    alphas[grid_key] = float(grid_model.alpha().detach())
                    torch.save(grid_model.state_dict(), output / f"grid_{name}_{seed}_h{hyper_index}.pt")
                    if hyper_index == chosen:
                        model = grid_model
                        traces[key] = traces[grid_key]
            else:
                model, traces[key] = train_model(model, features, rows, data["y"], args.weight_decays[chosen], args)
            scores[key], values = predict(model, features)
            for diagnostic, value in values.items():
                diagnostics[key + "|" + diagnostic] = value
            if name in FUSIONS:
                alphas[key] = float(model.alpha().detach())
            models[key] = model
            torch.save(model.state_dict(), output / f"model_{name}_{seed}.pt")
            print(f"Fitted {key}: source CV BCE={losses[chosen]:.6f}", flush=True)
        for name in ("CONCAT", "FIXED", "LEARNED"):
            compare(f"V_MATCHED|{seed}", f"{name}|{seed}", "matched_baseline_increment")
        compare(f"FIXED|{seed}", f"LEARNED|{seed}", "learned_vs_fixed_weight")
        compare(f"CONCAT|{seed}", f"LEARNED|{seed}", "learned_vs_normalized_concatenation")
        compare(f"CONCAT|{seed}", f"FIXED|{seed}", "fixed_vs_normalized_concatenation")
        for hyper_index in range(len(args.weight_decays)):
            compare(f"GRID|FIXED|{seed}|h{hyper_index}", f"GRID|LEARNED|{seed}|h{hyper_index}",
                    "same_regularization_gate_ablation_not_target_selection")
        for name in VARIANTS:
            compare("V_REFERENCE", f"{name}|{seed}", "historical_probe_reference_not_matched_primary")
        compare("V_C_REFERENCE", f"LEARNED|{seed}", "historical_linear_fusion_reference")
        for name in FUSIONS:
            disabled = f"CLIP_OFF|{name}|{seed}"
            scores[disabled], values = predict(models[f"{name}|{seed}"], features, alpha_override=0.0)
            compare(disabled, f"{name}|{seed}", "fixed_model_clip_off_not_retrained_v_baseline")
    controls, donor_masks = {}, {}
    for seed in args.seeds:
        for repeat in range(args.repeats):
            donor_features, noise_features, parameters = clip_controls(features, data, rows, seed + 200000 + repeat)
            for name, value in parameters.items():
                controls[f"{name}|{seed}|{repeat}"] = value
            for name in FUSIONS:
                real = f"{name}|{seed}"
                for label, changed in (("DONOR", donor_features), ("NOISE_FIXED", noise_features)):
                    key = f"{label}|{name}|{seed}|{repeat}"
                    scores[key], values = predict(models[real], changed)
                    if label == "DONOR":
                        donor_masks[key] = parameters["usable"]
                    compare(key, real, "current_image_content_control")
                key = f"NOISE_RETRAIN|{name}|{seed}|{repeat}"
                model = VectorFusion(data["V"].shape[1], data["C"].shape[1], name, seed)
                model, traces[key] = train_model(model, noise_features, rows, data["y"], selection[real]["weight_decay"], args)
                scores[key], values = predict(model, noise_features)
                alphas[key] = float(model.alpha().detach())
                torch.save(model.state_dict(), output / f"noise_{name}_{seed}_{repeat}.pt")
                compare(key, real, "matched_parameter_noise_retraining")
                compare(f"V_MATCHED|{seed}", key, "noise_capacity_vs_matched_baseline")
        print(f"Saved all content controls for seed {seed}", flush=True)
    for name in VARIANTS:
        key = f"{name}|ensemble"
        scores[key] = np.mean([scores[f"{name}|{seed}"] for seed in args.seeds], axis=0)
        if name != "V_MATCHED":
            compare("V_MATCHED|ensemble", key, "ensemble_secondary")
    compare("FIXED|ensemble", "LEARNED|ensemble", "ensemble_secondary")
    compare("CONCAT|ensemble", "LEARNED|ensemble", "ensemble_secondary")
    compare("V_REFERENCE", "V_MATCHED|ensemble", "normalization_reference")
    identity = {key: data[key] for key in IDENTITY}
    np.savez_compressed(output / "scores.npz", **identity, **scores)
    np.savez_compressed(output / "features.npz", **identity, V=data["V"], C=data["C"])
    np.savez_compressed(output / "controls.npz", **controls)
    np.savez_compressed(output / "vector_diagnostics.npz", **identity, **diagnostics)
    write_json(output / "training_trace.json", traces)
    write_json(output / "comparisons.json", comparisons)
    metadata.update({"selection": selection, "parameter_counts": counts, "alpha_values": alphas,
                     "noise_generator": "per(domain,split): default_rng(cell_seed(noise_seed,domain,split)).normal(shape).astype(float32)*noise_std+noise_mean",
                     "status": "statistics_pending"})
    write_json(output / "config.json", metadata)
    report = report_results(data, scores, comparisons, donor_masks, args)
    report["alpha_values"] = alphas
    report["vector_diagnostics"] = {}
    for domain in np.unique(data["domain"][data["split"] == "test"]):
        mask = (data["domain"] == domain) & (data["split"] == "test")
        report["vector_diagnostics"][domain] = {
            key: {"mean": float(value[mask].mean()), "std": float(value[mask].std())}
            for key, value in diagnostics.items()}
    write_json(output / "summary.json", report)
    metadata["status"] = "complete"
    write_json(output / "config.json", metadata)
    write_json(output / "artifact_hashes.json", {
        str(path.relative_to(output)).replace("\\", "/"): file_digest(path)
        for path in sorted(output.rglob("*")) if path.is_file()})
    (output / "COMPLETE").write_text("G29 completed\n", encoding="utf-8")
    print(f"Saved {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
