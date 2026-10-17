"""G28: CLIP content controls and controlled readout-failure diagnostics."""

import argparse
import csv
import hashlib
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
from sklearn.model_selection import GroupKFold
from torch import nn

from analyze import fit_reader, save_reader, summarize, validate_splits
from conditional_clip import Store, load_local, normalize_regions, parameter_count, split_donors
from pure_vit_clip import VARIANTS, build_reader, calibrate, macro_comparison, predict, prepare_store


IDENTITY = ("row_id", "path", "domain", "split", "video_id", "y")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def grouped_folds(data, rows, count):
    count = min(count, len(np.unique(data["video_id"][rows])))
    if count < 2:
        raise ValueError("At least two videos required in every nested source fit")
    folds = []
    for fitting, validation in GroupKFold(count).split(rows, groups=data["video_id"][rows]):
        fit_rows, val_rows = rows[fitting], rows[validation]
        if len(np.unique(data["y"][fit_rows])) != 2:
            raise ValueError("Each baseline fitting fold needs both classes")
        folds.append((fit_rows, val_rows))
    return folds


def source_reader(data, rows, args):
    mask = np.zeros(len(data["y"]), dtype=bool)
    mask[rows] = True
    return fit_reader(data["V"], data["y"], mask, args.baseline_c, args.seeds[0])


def baseline_context(data, rows, args, output, prefix):
    reader = source_reader(data, rows, args)
    save_reader(output / f"{prefix}_baseline.npz", reader)
    store, scalers = prepare_store(data, rows, reader)
    np.savez_compressed(output / f"{prefix}_scalers.npz", **scalers)
    out_of_fold = store.s.copy()
    records = []
    for fold_index, (fit_rows, val_rows) in enumerate(grouped_folds(data, rows, args.inner_folds)):
        inner_reader = source_reader(data, fit_rows, args)
        out_of_fold[val_rows] = inner_reader.decision_function(data["V"][val_rows])
        save_reader(output / f"{prefix}_inner{fold_index}_baseline.npz", inner_reader)
        records.append({"fit_row_id": data["row_id"][fit_rows].tolist(),
                        "validation_row_id": data["row_id"][val_rows].tolist()})
    training = Store({"v": store.v, "c": store.c, "ci": store.ci, "s": out_of_fold},
                     store.regions, store.c_dim, "cpu")
    np.savez_compressed(output / f"{prefix}_oof.npz", row_id=data["row_id"][rows],
                        score=out_of_fold[rows])
    return store, training, records


def fit_traced(model, store, rows, labels, hyper, args, validation_rows=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=hyper[1])
    batch = store.batch(rows)
    target = torch.as_tensor(labels[rows], dtype=torch.float32)
    trace = []
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        correction, weights = model(*batch)
        bce = nn.functional.binary_cross_entropy_with_logits(batch[3] + correction, target)
        loss = bce + hyper[0] * correction.square().mean()
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        loss.backward()
        optimizer.step()
        if epoch == 0 or (epoch + 1) % args.trace_every == 0 or epoch + 1 == args.epochs:
            model.eval()
            record = {"epoch": epoch + 1}
            with torch.no_grad():
                for name, indices in (("fit", rows), ("validation", validation_rows)):
                    if indices is None:
                        continue
                    current = store.batch(indices)
                    delta, attention = model(*current)
                    labels_tensor = torch.as_tensor(labels[indices], dtype=torch.float32)
                    record[name + "_bce"] = float(nn.functional.binary_cross_entropy_with_logits(
                        current[3] + delta, labels_tensor))
                    record[name + "_baseline_bce"] = float(nn.functional.binary_cross_entropy_with_logits(
                        current[3], labels_tensor))
                    record[name + "_delta_abs_mean"] = float(delta.abs().mean())
                    record[name + "_delta_rms"] = float(delta.square().mean().sqrt())
            trace.append(record)
    return model, trace


def intervention_prediction(model, store, temperature=1.0, uniform=False):
    original_mode, original_scale = model.mode, model.scale
    try:
        model.scale = original_scale / temperature
        if uniform:
            model.mode = "MEAN"
        return predict(model, store)
    finally:
        model.mode, model.scale = original_mode, original_scale


def cell_seed(seed, domain, split):
    digest = hashlib.sha256(f"{seed}|{domain}|{split}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def controlled_stores(store, training, data, train_rows, seed):
    donors = np.full(len(data["y"]), -1, dtype=int)
    noise_c = np.empty_like(store.c)
    noise_regions = np.empty_like(store.ci)
    parameters = {"c_mean": store.c[train_rows].mean(axis=0),
                  "c_std": store.c[train_rows].std(axis=0),
                  "regions_mean": store.ci[train_rows].mean(axis=0),
                  "regions_std": store.ci[train_rows].std(axis=0)}
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
            noise_c[rows] = (noise_rng.normal(size=store.c[rows].shape).astype(np.float32)
                             * parameters["c_std"] + parameters["c_mean"])
            noise_regions[rows] = (noise_rng.normal(size=store.ci[rows].shape).astype(np.float32)
                                   * parameters["regions_std"] + parameters["regions_mean"])
    usable = donors >= 0
    donor_rows = np.where(usable, donors, np.arange(len(donors)))

    def replace(visual_store, pooled, regions):
        return Store({"v": visual_store.v, "c": pooled, "ci": regions, "s": visual_store.s},
                     store.regions, store.c_dim, "cpu")

    return (replace(store, store.c[donor_rows], store.ci[donor_rows]),
            replace(store, noise_c, noise_regions), replace(training, noise_c, noise_regions),
            {**parameters, "donors": donors, "usable": usable, "seed": np.array(seed)})


def evaluation_subset(data, source, manifest):
    if manifest is None:
        return data
    with open(manifest, encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "row_id" not in reader.fieldnames:
            raise ValueError("Evaluation manifest requires a row_id column")
        selected = [int(row["row_id"]) for row in reader]
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("Evaluation row_ids must be nonempty and unique")
    chosen = np.isin(data["row_id"], selected)
    if chosen.sum() != len(selected) or np.any(chosen & (data["split"] != "test")):
        raise ValueError("Evaluation manifest must select existing test rows only")
    selected_groups = set(zip(data["domain"][chosen], data["video_id"][chosen]))
    for domain, video in selected_groups:
        group = (data["domain"] == domain) & (data["video_id"] == video) & (data["split"] == "test")
        if not chosen[group].all():
            raise ValueError("Evaluation manifest must include every extracted frame of each chosen video")
    keep = chosen | ((data["domain"] == source) & (data["split"] == "train"))
    return {key: value[keep] for key, value in data.items()}


def score_diagnostics(data, baseline, candidate, weights, mask):
    delta = candidate[mask] - baseline[mask]
    baseline_sd = float(baseline[mask].std())
    result = {"delta_abs_p50": float(np.quantile(np.abs(delta), 0.5)),
              "delta_abs_p90": float(np.quantile(np.abs(delta), 0.9)),
              "delta_rms": float(np.sqrt(np.mean(delta ** 2))), "baseline_sd": baseline_sd,
              "delta_abs_p50_over_baseline_sd": None if baseline_sd == 0 else
              float(np.quantile(np.abs(delta), 0.5) / baseline_sd)}
    result["delta_auc"] = (float(roc_auc_score(data["y"][mask], delta))
                           if len(np.unique(data["y"][mask])) == 2 else None)
    if weights is not None:
        values = weights[mask]
        entropy = -(values * np.log(values.clip(1e-12))).sum(axis=1)
        result.update({"attention_entropy_mean": float(entropy.mean()),
                       "attention_max_mean": float(values.max(axis=1).mean()),
                       "attention_mean": values.mean(axis=0).tolist()})
    return result


def make_report(data, scores, comparisons, weights, donor_masks, args):
    report = {"per_domain": {}, "primary_macro": {}, "single_seed_auc": {},
              "score_diagnostics": {}, "comparison_roles": {}, "primary_single_seed_auc": {},
              "primary_questions": {
                  "content_increment": [f"SELECTED|POOLED|{seed}_vs_V_BASE" for seed in args.seeds],
                  "adaptive_weight_increment": [f"SELECTED|ADAPTIVE|{seed}_vs_SELECTED|MEAN|{seed}"
                                                for seed in args.seeds]}}
    cache = {}
    for key, (base_key, candidate_key, role) in comparisons.items():
        report["comparison_roles"][key] = role
        usable = np.ones(len(data["y"]), dtype=bool)
        for score_key in (base_key, candidate_key):
            if score_key in donor_masks:
                usable &= donor_masks[score_key]
        for domain in np.unique(data["domain"][data["split"] == "test"]):
            mask = (data["domain"] == domain) & (data["split"] == "test") & usable
            report["per_domain"].setdefault(domain, {})[key] = (
                summarize(data, scores[base_key], scores[candidate_key], mask,
                          args.bootstrap, args.seeds[0]) if mask.any() else {"status": "no usable rows"})
        primary_rows = (data["split"] == "test") & np.isin(data["domain"], args.primary_domains)
        if not usable[primary_rows].all():
            report["primary_macro"][key] = {"status": "incomplete donor coverage"}
        else:
            report["primary_macro"][key] = macro_comparison(
                data, scores[base_key], scores[candidate_key], args.primary_domains,
                args.bootstrap, args.seeds[0], cache)
    for domain in np.unique(data["domain"][data["split"] == "test"]):
        mask = (data["domain"] == domain) & (data["split"] == "test")
        report["score_diagnostics"][domain] = {
            key: score_diagnostics(data, scores["V_BASE"], score, weights.get(key), mask)
            for key, score in scores.items() if not key.startswith(("DONOR", "NOISE"))}
        report["single_seed_auc"][domain] = {}
        for name in VARIANTS:
            values = [roc_auc_score(data["y"][mask], scores[f"SELECTED|{name}|{seed}"][mask])
                      for seed in args.seeds]
            report["single_seed_auc"][domain][name] = {
                "values": values, "mean": float(np.mean(values)), "std": float(np.std(values)),
                "ensemble_auc": float(roc_auc_score(data["y"][mask], scores[f"SELECTED|{name}|ensemble"][mask]))}
    for name in VARIANTS:
        values = [report["primary_macro"][f"SELECTED|{name}|{seed}_vs_V_BASE"]["macro_auc"]
                  for seed in args.seeds]
        report["primary_single_seed_auc"][name] = {
            "values": values, "mean": float(np.mean(values)), "std": float(np.std(values)),
            "ensemble_auc": report["primary_macro"][f"SELECTED|{name}|ensemble_vs_V_BASE"]["macro_auc"]}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source", default="ffpp")
    parser.add_argument("--primary-domains", nargs="+", default=["cd2", "dfdcp", "wild"])
    parser.add_argument("--evaluation-manifest")
    parser.add_argument("--evaluation-status", choices=["exploratory", "new-holdout"], default="exploratory")
    parser.add_argument("--seeds", type=int, nargs="+", default=[20261020, 20261021, 20261022])
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--trace-every", type=int, default=20)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.01, 0.1, 1.0])
    parser.add_argument("--weight-decays", type=float, nargs="+", default=[0.001])
    parser.add_argument("--gains", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0])
    parser.add_argument("--temperatures", type=float, nargs="+", default=[2.0, 4.0])
    parser.add_argument("--baseline-c", type=float, default=0.001)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    numeric = [args.lr, args.baseline_c, *args.lambdas, *args.weight_decays, *args.gains, *args.temperatures]
    if (not np.isfinite(numeric).all() or min(args.epochs, args.trace_every, args.width, args.dim,
            args.repeats, args.threads) < 1 or min(args.folds, args.inner_folds) < 2 or args.bootstrap < 0
            or min(args.lr, args.baseline_c) <= 0 or min(args.lambdas + args.weight_decays) < 0
            or any(gain < 0 or gain > 1 for gain in args.gains) or min(args.temperatures) <= 1
            or min(args.seeds) < 0):
        parser.error("Invalid training, intervention or statistical settings")
    for values in (args.seeds, args.primary_domains, args.lambdas, args.weight_decays, args.gains, args.temperatures):
        if len(values) != len(set(values)):
            parser.error("Duplicate settings are not allowed")
    if (args.source in args.primary_domains or "ffiw" in args.primary_domains
            or {"cd1", "cd2"}.issubset(args.primary_domains)):
        parser.error("Source/FFIW/overlapping CD1+CD2 are not valid primary domains")
    if args.evaluation_status == "new-holdout" and args.evaluation_manifest is None:
        parser.error("new-holdout requires a predeclared evaluation manifest")
    torch.set_num_threads(args.threads)
    root, output = Path(args.input), Path(args.output)
    if not (root / "COMPLETE").exists():
        raise ValueError("Extraction incomplete")
    extraction = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if extraction.get("label") != "1=fake, 0=real" or not extraction["arguments"].get("save_regions"):
        raise ValueError("Need region extraction with label 1=fake, 0=real")
    data = evaluation_subset(load_local(root), args.source, args.evaluation_manifest)
    train = validate_splits(data, args.source)
    for name in ("V", "C", "clip_regions", "logits"):
        if not np.isfinite(data[name]).all():
            raise ValueError(f"Nonfinite input: {name}")
    if data["C"].shape[1] != data["clip_regions"].shape[2]:
        raise ValueError("Global/region CLIP dimensions differ")
    data["regions"] = normalize_regions(data["clip_regions"])
    rows = np.flatnonzero(train)
    source_folds = grouped_folds(data, rows, args.folds)
    for fit_rows, val_rows in source_folds:
        grouped_folds(data, fit_rows, args.inner_folds)
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "code_snapshot"
    snapshot.mkdir()
    code_hashes = {}
    for filename in ("clip_readout.py", "pure_vit_clip.py", "conditional_clip.py", "analyze.py",
                     "CLIP_READOUT_EXPERIMENT.md"):
        source_path = Path(__file__).with_name(filename)
        shutil.copy2(source_path, snapshot / filename)
        code_hashes[filename] = file_digest(source_path)
    input_hashes = {path.name: file_digest(path)
                    for path in sorted(root.glob("clean_*.npz"))}
    metadata = {"arguments": vars(args), "status": "running", "label": "1=fake, 0=real",
                "code_sha256": code_hashes, "input_config_sha256": file_digest(root / "config.json"),
                "versions": {"numpy": np.__version__, "sklearn": sklearn.__version__, "torch": str(torch.__version__)},
                "baseline_training": "nested source-video OOF scores train correction; full-fit V scores evaluate",
                "checkpoint_selection": "inherited extraction; new-holdout does not certify source-only checkpoint selection",
                "controls": "real-source-selected hyperparameters; noise controls are not independently tuned",
                "noise_generator": "per domain/split: default_rng(cell_seed(control_seed+100000,domain,split)); C then regions normal float32, times source std plus mean"}
    if args.evaluation_manifest:
        shutil.copy2(args.evaluation_manifest, output / "evaluation_manifest.csv")
        metadata["evaluation_manifest_sha256"] = file_digest(Path(args.evaluation_manifest))
    write_json(output / "config.json", metadata)
    write_json(output / "input_hashes.json", input_hashes)
    shutil.copy2(root / "config.json", output / "input_config.json")
    store, training, inner_record = baseline_context(data, rows, args, output, "final")
    scores = {"V_BASE": store.s.copy(), "DETECTOR_REFERENCE": data["logits"][:, 0] - data["logits"][:, 1]}
    scale, bias = calibrate(training.s[rows], data["y"][rows])
    scores["CALIBRATED"] = scale * scores["V_BASE"].astype(np.float64) + bias
    metadata["calibration"] = {"scale": scale, "bias": bias, "fit": "source OOF only"}
    concat_reader = fit_reader(np.concatenate([data["V"], data["C"]], axis=1), data["y"], train,
                              args.baseline_c, args.seeds[0])
    save_reader(output / "reader_V_C_LINEAR.npz", concat_reader)
    scores["V_C_LINEAR"] = concat_reader.decision_function(np.concatenate([data["V"], data["C"]], axis=1))
    split_record = {"final_inner": inner_record, "outer": []}
    contexts = []
    for fold_index, (fit_rows, val_rows) in enumerate(source_folds):
        fold_store, fold_training, inner = baseline_context(data, fit_rows, args, output, f"fold{fold_index}")
        contexts.append((fit_rows, val_rows, fold_training))
        split_record["outer"].append({"fit_row_id": data["row_id"][fit_rows].tolist(),
                                      "validation_row_id": data["row_id"][val_rows].tolist(), "inner": inner})
    write_json(output / "split.json", split_record)
    grid = [(lam, decay) for lam in args.lambdas for decay in args.weight_decays]
    selection, models, weights, traces, counts = {}, {}, {}, {}, {}
    comparisons = {}

    def compare(base, candidate, role):
        comparisons[candidate + "_vs_" + base] = (base, candidate, role)

    for seed in args.seeds:
        for name in VARIANTS:
            validation_losses = []
            keys = []
            for grid_index, hyper in enumerate(grid):
                cv_total, cv_count = 0.0, 0
                for fold_index, (fit_rows, val_rows, fold_training) in enumerate(contexts):
                    model = build_reader(name, data["V"].shape[1], data["C"].shape[1], args, seed + fold_index)
                    model, trace = fit_traced(model, fold_training, fit_rows, data["y"], hyper, args, val_rows)
                    traces[f"CV|{name}|{seed}|h{grid_index}|fold{fold_index}"] = trace
                    cv_total += trace[-1]["validation_bce"] * len(val_rows)
                    cv_count += len(val_rows)
                validation_losses.append(cv_total / cv_count)
                key = f"GRID|{name}|{seed}|h{grid_index}"
                keys.append(key)
                model = build_reader(name, data["V"].shape[1], data["C"].shape[1], args, seed)
                counts[name] = parameter_count(model)
                model, trace = fit_traced(model, training, rows, data["y"], hyper, args)
                traces[key] = trace
                scores[key], attention = predict(model, store)
                if attention is not None:
                    weights[key] = attention
                models[key] = model
                torch.save(model.state_dict(), output / f"adapter_{name}_{seed}_h{grid_index}.pt")
                compare("V_BASE", key, "fixed_hyper_diagnostic_not_target_selection")
            chosen = int(np.argmin(validation_losses))
            selection[f"{name}|{seed}"] = {"grid_index": chosen, "lambda_delta": grid[chosen][0],
                                          "weight_decay": grid[chosen][1], "cv_bce": validation_losses[chosen],
                                          "grid_cv_bce": validation_losses,
                                          "at_lambda_upper_bound": grid[chosen][0] == max(args.lambdas)}
            selected_key = f"SELECTED|{name}|{seed}"
            scores[selected_key] = scores[keys[chosen]]
            models[selected_key] = models[keys[chosen]]
            if keys[chosen] in weights:
                weights[selected_key] = weights[keys[chosen]]
            compare("V_BASE", selected_key, "source_selected_readout")
            print(f"Fitted {selected_key}; source CV chose h{chosen}", flush=True)
        for grid_index in range(len(grid)):
            for baseline in ("POOLED", "MEAN", "V_ONLY"):
                compare(f"GRID|{baseline}|{seed}|h{grid_index}", f"GRID|ADAPTIVE|{seed}|h{grid_index}",
                        "same_hyper_readout_diagnostic")
        for name in ("POOLED", "ADAPTIVE"):
            key = f"SELECTED|{name}|{seed}"
            for baseline in ("CALIBRATED", "V_C_LINEAR", f"SELECTED|V_ONLY|{seed}"):
                compare(baseline, key, "capacity_calibration_or_linear_control")
            for gain in args.gains:
                changed = f"GAIN|{name}|{seed}|{gain:g}"
                scores[changed] = (scores["V_BASE"].copy() if gain == 0 else scores[key].copy() if gain == 1
                                   else scores["V_BASE"] + gain * (scores[key] - scores["V_BASE"]))
                compare(key, changed, "fixed_model_correction_amplitude_intervention")
                compare("V_BASE", changed, "amplitude_vs_base_not_deployment_selection")
        adaptive_key = f"SELECTED|ADAPTIVE|{seed}"
        for temperature in args.temperatures:
            key = f"TEMPERATURE|{seed}|{temperature:g}"
            scores[key], weights[key] = intervention_prediction(models[adaptive_key], store, temperature)
            compare(adaptive_key, key, "fixed_model_attention_intervention")
            compare("V_BASE", key, "attention_intervention_vs_base")
        key = f"UNIFORM|{seed}"
        scores[key], weights[key] = intervention_prediction(models[adaptive_key], store, uniform=True)
        compare(adaptive_key, key, "fixed_model_attention_intervention")
        compare("V_BASE", key, "attention_intervention_vs_base")
        compare(f"SELECTED|MEAN|{seed}", adaptive_key, "source_selected_attention_vs_uniform_training")
        compare(f"SELECTED|POOLED|{seed}", adaptive_key, "source_selected_local_vs_global")
    metadata.update({"selection": selection, "parameter_counts": counts,
                     "hyper_grid": [{"lambda_delta": hyper[0], "weight_decay": hyper[1]} for hyper in grid]})
    write_json(output / "config.json", metadata)
    write_json(output / "training_trace.json", traces)
    controls, donor_masks = {}, {}
    for seed in args.seeds:
        for repeat in range(args.repeats):
            control_seed = seed + 10000 + repeat
            donor_store, noise_store, noise_training, parameters = controlled_stores(
                store, training, data, rows, control_seed)
            for key, value in parameters.items():
                controls[f"{key}|{seed}|{repeat}"] = value
            for name in ("POOLED", "ADAPTIVE"):
                selected_key = f"SELECTED|{name}|{seed}"
                chosen = selection[f"{name}|{seed}"]["grid_index"]
                for label, changed_store in (("DONOR", donor_store), ("NOISE_FIXED", noise_store)):
                    key = f"{label}|{name}|{seed}|{repeat}"
                    scores[key], attention = predict(models[selected_key], changed_store)
                    if label == "DONOR":
                        donor_masks[key] = parameters["usable"]
                    compare(key, selected_key, "selected_readout_content_control")
                model = build_reader(name, data["V"].shape[1], data["C"].shape[1], args, seed)
                key = f"NOISE_RETRAIN|{name}|{seed}|{repeat}"
                model, trace = fit_traced(model, noise_training, rows, data["y"], grid[chosen], args)
                scores[key], attention = predict(model, noise_store)
                traces[key] = trace
                torch.save(model.state_dict(), output / f"noise_{name}_{seed}_{repeat}.pt")
                compare(key, selected_key, "selected_readout_content_control")
                compare("V_BASE", key, "noise_capacity_control")
    for name in VARIANTS:
        key = f"SELECTED|{name}|ensemble"
        scores[key] = np.mean([scores[f"SELECTED|{name}|{seed}"] for seed in args.seeds], axis=0)
        compare("V_BASE", key, "ensemble_secondary_not_single_model_mean")
    compare("SELECTED|POOLED|ensemble", "SELECTED|ADAPTIVE|ensemble", "ensemble_secondary")
    compare("V_BASE", "V_C_LINEAR", "linear_reference")
    compare("V_BASE", "CALIBRATED", "calibration_rank_check")
    compare("V_BASE", "DETECTOR_REFERENCE", "detector_reference_only")
    identity = {key: data[key] for key in IDENTITY}
    np.savez_compressed(output / "scores.npz", **identity, **scores)
    np.savez_compressed(output / "features.npz", **identity, V=data["V"], C=data["C"],
                        clip_regions=data["clip_regions"])
    np.savez_compressed(output / "controls.npz", **controls)
    np.savez_compressed(output / "region_weights.npz", **weights)
    write_json(output / "training_trace.json", traces)
    write_json(output / "comparisons.json", comparisons)
    metadata["status"] = "statistics_pending"
    write_json(output / "config.json", metadata)
    report = make_report(data, scores, comparisons, weights, donor_masks, args)
    write_json(output / "summary.json", report)
    metadata["status"] = "complete"
    write_json(output / "config.json", metadata)
    artifacts = {str(path.relative_to(output)).replace("\\", "/"): file_digest(path)
                 for path in sorted(output.rglob("*")) if path.is_file()}
    write_json(output / "artifact_hashes.json", artifacts)
    (output / "COMPLETE").write_text("G28 completed\n", encoding="utf-8")
    print(f"Saved {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
