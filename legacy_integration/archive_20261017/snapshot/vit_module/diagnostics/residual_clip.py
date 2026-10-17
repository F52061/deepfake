"""Test source-fitted CLIP residuals with reversible-transform controls."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from analyze import donor_indices, fit_reader, load_observations, save_reader, summarize, validate_splits


def residualize(visual, clip, train, groups, alphas):
    indices = np.flatnonzero(train)
    folds = GroupKFold(n_splits=min(5, len(np.unique(groups[train]))))
    if folds.n_splits < 2:
        raise ValueError("At least two source training videos required")
    losses = {}
    for alpha in alphas:
        squared, elements = 0.0, 0
        for fitting, validation in folds.split(indices, groups=groups[train]):
            fitting, validation = indices[fitting], indices[validation]
            scaler = StandardScaler().fit(visual[fitting])
            regression = Ridge(alpha=alpha).fit(scaler.transform(visual[fitting]), clip[fitting])
            error = regression.predict(scaler.transform(visual[validation])) - clip[validation]
            squared += float(np.square(error).sum())
            elements += error.size
        losses[str(alpha)] = squared / elements
    alpha = min(alphas, key=lambda value: losses[str(value)])
    scaler = StandardScaler().fit(visual[train])
    regression = Ridge(alpha=alpha).fit(scaler.transform(visual[train]), clip[train])
    predicted = regression.predict(scaler.transform(visual))
    return clip - predicted, predicted, scaler, regression, losses, alpha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Extraction directory or standardized feature NPZ")
    parser.add_argument("--output", required=True)
    parser.add_argument("--source", default="ffpp")
    parser.add_argument("--label-one", choices=["fake", "real"], required=True)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.1, 1, 10, 100, 1000])
    parser.add_argument("--regularization", type=float, default=0.001)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261002)
    args = parser.parse_args()
    if min(args.alphas) <= 0 or args.regularization <= 0 or args.repeats < 1 or args.bootstrap < 0:
        parser.error("Invalid alpha, regularization or repetition settings")
    source = Path(args.input)
    if source.is_dir():
        if not (source / "COMPLETE").exists():
            raise ValueError("Extraction incomplete")
        data = load_observations(source, "clean")
        config = json.loads((source / "config.json").read_text(encoding="utf-8"))
        if config["label"] != "1=fake, 0=real" or args.label_one != "fake":
            raise ValueError("Extraction schema requires --label-one fake")
        digest = hashlib.sha256((source / "config.json").read_bytes()).hexdigest()
    else:
        with np.load(source, allow_pickle=False) as archive:
            data = {key: archive[key] for key in ("V", "C", "y", "path", "domain", "split", "video_id", "row_id")}
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if args.label_one == "real":
            data["y"] = 1 - data["y"]
    train = validate_splits(data, args.source)
    if not np.array_equal(data["row_id"], np.arange(len(data["y"]))):
        raise ValueError("row_id must be contiguous")
    for key in ("V", "C"):
        if data[key].ndim != 2 or not np.isfinite(data[key]).all():
            raise ValueError(f"Invalid {key} feature matrix")
    evaluation = data["split"] == "test"
    if not evaluation.any():
        raise ValueError("Test observations required")
    residual, predicted, scaler, regression, losses, alpha = residualize(
        data["V"], data["C"], train, data["video_id"], args.alphas)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    mapping = regression.coef_ / scaler.scale_[None, :]
    offset = regression.intercept_ - mapping @ scaler.mean_
    reconstruction = float(np.max(np.abs(residual + data["V"] @ mapping.T + offset - data["C"])))
    if reconstruction > 1e-6:
        raise ValueError("Residual reconstruction failed")
    np.savez_compressed(output / "ridge.npz", coef=mapping, intercept=offset,
                        source_row_id=data["row_id"][train], alpha=np.array(alpha))
    sets = {"V": data["V"], "C": data["C"], "Cres": residual,
            "V_C": np.concatenate([data["V"], data["C"]], axis=1),
            "V_Cres": np.concatenate([data["V"], residual], axis=1)}
    scores, readers = {}, {}
    for name, features in sets.items():
        reader = fit_reader(features, data["y"], train, args.regularization, args.seed)
        readers[name] = reader
        scores[name] = reader.decision_function(features)
        save_reader(output / f"reader_{name}.npz", reader)
    fitted_scaler, logistic = readers["V_C"].steps[0][1], readers["V_C"].steps[1][1]
    raw_weight = logistic.coef_[0] / fitted_scaler.scale_
    raw_bias = float(logistic.intercept_[0] - raw_weight @ fitted_scaler.mean_)
    dimension = data["V"].shape[1]
    equivalent_weight = np.concatenate([raw_weight[:dimension] + raw_weight[dimension:] @ mapping,
                                       raw_weight[dimension:]])
    equivalent_bias = raw_bias + raw_weight[dimension:] @ offset
    scores["V_Cres_equivalent"] = sets["V_Cres"] @ equivalent_weight + equivalent_bias
    # Compared as a RELATIVE error. The identity below is exact algebra, but the
    # two sides are evaluated along different floating-point paths, and the ridge
    # solve is ill conditioned here (V's spectrum is steep: PC0 alone carries
    # ~62% of the variance, so the Gram matrix has rcond ~3e-8 and sklearn emits
    # LinAlgWarning). That amplifies roundoff to ~1e-8 relative, which on scores
    # of magnitude ~7 lands just above an ABSOLUTE 1e-7 threshold. The absolute
    # error still passed 1e-7 on the synthetic check because those features are
    # 8-dimensional and the scores are O(1); it does not generalise to this data.
    # Scaling by the score magnitude keeps the guard's intent — a coding error in
    # the transform would be orders of magnitude larger than this — while making
    # it independent of the score scale.
    score_scale = max(1.0, float(np.max(np.abs(scores["V_C"]))))
    equivalent_error = float(np.max(np.abs(scores["V_Cres_equivalent"] - scores["V_C"])))
    equivalent_relative_error = equivalent_error / score_scale
    if equivalent_relative_error > 1e-7:
        raise ValueError("Equivalent classifier does not preserve predictions")
    np.savez_compressed(output / "equivalent_head.npz", weight=equivalent_weight, bias=np.array(equivalent_bias))
    random_artifacts = {}
    for repeat in range(args.repeats):
        rng = np.random.default_rng(args.seed + repeat)
        donors = np.arange(len(data["y"]))
        for domain in np.unique(data["domain"]):
            for split in np.unique(data["split"]):
                indices = np.flatnonzero((data["domain"] == domain) & (data["split"] == split))
                if len(indices):
                    donors[indices] = indices[donor_indices(data["video_id"][indices], rng)]
        shuffled = np.concatenate([data["V"], residual[donors]], axis=1)
        scores[f"evalshuffle_{repeat}"] = readers["V_Cres"].decision_function(shuffled)
        shuffled_reader = fit_reader(shuffled, data["y"], train, args.regularization, args.seed)
        scores[f"trainshuffle_{repeat}"] = shuffled_reader.decision_function(shuffled)
        save_reader(output / f"reader_shuffle_{repeat}.npz", shuffled_reader)
        noise = rng.normal(size=residual.shape) * residual[train].std(axis=0) + residual[train].mean(axis=0)
        noise_features = np.concatenate([data["V"], noise], axis=1)
        noise_reader = fit_reader(noise_features, data["y"], train, args.regularization, args.seed)
        scores[f"noise_{repeat}"] = noise_reader.decision_function(noise_features)
        save_reader(output / f"reader_noise_{repeat}.npz", noise_reader)
        random_artifacts[f"donors_{repeat}"] = donors
        random_artifacts[f"noise_{repeat}"] = noise
    identity = {key: data[key] for key in ("row_id", "path", "domain", "split", "video_id", "y")}
    np.savez_compressed(output / "features.npz", **identity, V=data["V"], C=data["C"],
                        Cres=residual, Cpred=predicted)
    np.savez_compressed(output / "scores.npz", **identity, **scores)
    np.savez_compressed(output / "controls.npz", **random_artifacts)
    metadata = {"arguments": vars(args), "input_metadata_sha256": digest,
                "label": "1=fake", "ridge_cv_mse": losses, "selected_alpha": alpha,
                "reconstruction_error": reconstruction, "equivalent_score_error": equivalent_error,
                "equivalent_score_relative_error": equivalent_relative_error,
                "equivalent_score_scale": score_scale,
                "equivalent_check": "relative to max|score|; threshold 1e-7",
                "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "interpretation": "Residual fusion is an invertible reparameterization; gains are not proof of new information."}
    (output / "config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    report = {}
    for domain in np.unique(data["domain"][evaluation]):
        mask = evaluation & (data["domain"] == domain)
        report[domain] = {name: summarize(data, scores["V"], score, mask, args.bootstrap, args.seed)
                          for name, score in scores.items()}
        report[domain]["V_Cres_vs_V_C"] = summarize(
            data, scores["V_C"], scores["V_Cres"], mask, args.bootstrap, args.seed)
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    (output / "COMPLETE").write_text("Analysis completed\n", encoding="utf-8")
    print(f"Saved {output.resolve()}; equivalent-score error={equivalent_error:.3g} "
          f"(relative {equivalent_relative_error:.3g}, scale {score_scale:.3g})")


if __name__ == "__main__":
    main()
