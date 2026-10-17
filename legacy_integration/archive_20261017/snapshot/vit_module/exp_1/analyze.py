"""Paired, source-thresholded error complementarity; no model training."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

import numpy as np


FIELDS = ("row_id", "path", "domain", "split", "video_id", "group_id", "y", "vit_score", "clip_score")
GROUPS = ("both_correct", "potential_rescue", "potential_harm", "both_wrong")


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def auc(labels, scores):
    positive = labels == 1
    count = int(positive.sum())
    if count == 0 or count == len(labels):
        return None
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1]
    ends = np.r_[starts[1:], len(scores)]
    sorted_ranks = np.repeat((starts + ends + 1) / 2, ends - starts)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = sorted_ranks
    return float((ranks[positive].sum() - count * (count + 1) / 2) / (count * (len(labels) - count)))


def threshold(labels, scores):
    unique = np.unique(scores)
    candidates = np.r_[np.nextafter(unique[0], -np.inf),
                       unique[:-1] / 2 + unique[1:] / 2,
                       np.nextafter(unique[-1], np.inf)]
    order = np.argsort(scores, kind="stable")
    sorted_labels = labels[order]
    prefix = np.r_[0, np.cumsum(sorted_labels)]
    positions = np.searchsorted(scores[order], candidates, side="left")
    positives = labels.sum()
    negatives = len(labels) - positives
    balanced = ((positives - prefix[positions]) / positives
                + (positions - prefix[positions]) / negatives) / 2
    selected = np.lexsort((candidates, np.abs(candidates), -balanced))[0]
    return float(candidates[selected]), float(balanced[selected])


def load(input_path, provenance_path, source):
    with np.load(input_path, allow_pickle=False) as archive:
        missing = set(FIELDS) - set(archive.files)
        if missing:
            raise ValueError(f"Missing fields: {sorted(missing)}")
        data = {key: archive[key] for key in FIELDS}
    size = len(data["y"])
    if not size or any(value.ndim != 1 or len(value) != size for value in data.values()):
        raise ValueError("All fields must be nonempty, equal-length 1D arrays")
    for name in ("row_id", "path"):
        if len(np.unique(data[name])) != size:
            raise ValueError(f"Duplicate {name}")
    if set(data["y"].tolist()) != {0, 1}:
        raise ValueError("Labels must contain both classes; 1=fake")
    for name in ("vit_score", "clip_score"):
        data[name] = data[name].astype(np.float64)
        if not np.isfinite(data[name]).all():
            raise ValueError(f"Nonfinite {name}")
    for name in ("domain", "video_id", "group_id", "path", "split"):
        if data[name].dtype.kind not in "US" or np.any(data[name] == ""):
            raise ValueError(f"{name} must contain nonempty strings")
    if not set(data["split"].tolist()) <= {"val", "test"}:
        raise ValueError("Only held-out val/test predictions allowed; no training predictions")
    calibration = (data["domain"] == source) & (data["split"] == "val")
    if np.any((data["split"] == "val") & (data["domain"] != source)):
        raise ValueError("Only source-domain validation rows allowed")
    if set(data["y"][calibration].tolist()) != {0, 1}:
        raise ValueError("Source validation requires both classes")
    if len(np.unique(data["group_id"][calibration])) < 2:
        raise ValueError("Source validation requires at least two independent groups")
    if not np.any(data["split"] == "test"):
        raise ValueError("Test predictions required")
    for identity in ("video_id", "group_id"):
        ownership = {}
        for domain, value, split in zip(data["domain"], data[identity], data["split"]):
            key = (domain, value)
            if key in ownership and ownership[key] != split:
                raise ValueError(f"{identity} crosses val/test: {key}")
            ownership[key] = split
    video_groups = {}
    for domain, video, group in zip(data["domain"], data["video_id"], data["group_id"]):
        key = (domain, video)
        if key in video_groups and video_groups[key] != group:
            raise ValueError("A video must belong to exactly one bootstrap group")
        video_groups[key] = group
    provenance = json.loads(Path(provenance_path).read_text(encoding="utf-8-sig"))
    if provenance.get("source") != source or provenance.get("label_definition") != "1=fake":
        raise ValueError("Provenance source/label definition mismatch")
    if provenance.get("video_identity_audited") is not True:
        raise ValueError("Audit actual video/group identities before running")
    for name, kinds in (("vit", {"native", "vit_only_probe"}), ("clip", {"clip_only_probe"})):
        branch = provenance.get(name, {})
        if branch.get("kind") not in kinds or branch.get("higher_score") != "fake":
            raise ValueError(f"Invalid independent {name} branch definition")
        if branch.get("evaluation_labels_used_for_training") is not False:
            raise ValueError("Evaluation labels cannot be used for training")
        if branch.get("validation_groups_excluded_from_training") is not True:
            raise ValueError("Calibration videos must be excluded from branch fitting")
        for key in ("checkpoint_sha256", "preprocessing", "training_description", "score_definition", "selection_history"):
            if not isinstance(branch.get(key), str) or not branch[key].strip():
                raise ValueError(f"Missing provenance {name}.{key}")
        if re.fullmatch(r"[0-9a-fA-F]{64}", branch["checkpoint_sha256"]) is None:
            raise ValueError(f"Invalid SHA256 for {name}")
        if any(branch[key].startswith("REPLACE_") for key in
               ("preprocessing", "training_description", "score_definition", "selection_history")):
            raise ValueError("Replace provenance placeholders with audited facts")
    return data, calibration, provenance


def measurements(labels, vit, clip, vit_threshold, clip_threshold):
    vit_right = (vit >= vit_threshold) == labels
    clip_right = (clip >= clip_threshold) == labels
    masks = (vit_right & clip_right, ~vit_right & clip_right,
             vit_right & ~clip_right, ~vit_right & ~clip_right)
    vit_auc, clip_auc = auc(labels, vit), auc(labels, clip)
    values = {"vit_auc": vit_auc, "clip_auc": clip_auc,
              "delta_auc_clip_minus_vit": None if vit_auc is None else clip_auc - vit_auc,
              "vit_accuracy": float(vit_right.mean()), "clip_accuracy": float(clip_right.mean()),
              "oracle_correctness_accuracy": float((vit_right | clip_right).mean()),
              "rescue_fraction_of_vit_errors": float(masks[1].sum() / (~vit_right).sum()) if (~vit_right).any() else None,
              "harm_fraction_of_vit_correct": float(masks[2].sum() / vit_right.sum()) if vit_right.any() else None,
              "rescue_minus_harm_fraction": float((masks[1].sum() - masks[2].sum()) / len(labels)),
              "disagreement_fraction": float((masks[1] | masks[2]).mean())}
    values.update({name + "_fraction": float(mask.mean()) for name, mask in zip(GROUPS, masks)})
    return values, masks


def intervals(draws):
    result = {}
    for name in draws[0] if draws else ():
        valid = [draw[name] for draw in draws if draw[name] is not None]
        result[name] = {"valid_bootstraps": len(valid),
                        "ci95": np.quantile(valid, [.025, .975]).tolist()
                        if len(valid) >= max(10, len(draws) // 2) else None}
    return result


def analyze(data, calibration, args):
    thresholds = {}
    for branch in ("vit", "clip"):
        value, balanced = threshold(data["y"][calibration], data[branch + "_score"][calibration])
        thresholds[branch] = {"value": value, "validation_balanced_accuracy": balanced}
    vit_threshold, clip_threshold = (thresholds[branch]["value"] for branch in ("vit", "clip"))
    per_domain, domain_draws = {}, {}
    group_codes = np.full(len(data["y"]), "calibration", dtype="U20")
    for domain_index, domain in enumerate(sorted(np.unique(data["domain"][data["split"] == "test"]))):
        rows = np.flatnonzero((data["domain"] == domain) & (data["split"] == "test"))
        labels, vit, clip = (data[key][rows] for key in ("y", "vit_score", "clip_score"))
        if len(np.unique(labels)) != 2:
            raise ValueError(f"Test domain {domain} requires both classes")
        values, masks = measurements(labels, vit, clip, vit_threshold, clip_threshold)
        counts = {}
        for name, mask in zip(GROUPS, masks):
            group_codes[rows[mask]] = name
            counts[name] = {"n": int(mask.sum()), "fraction": float(mask.mean()),
                            "real": int(np.sum(mask & (labels == 0))), "fake": int(np.sum(mask & (labels == 1))),
                            "videos": len(np.unique(data["video_id"][rows[mask]])),
                            "independent_groups": len(np.unique(data["group_id"][rows[mask]]))}
        blocks = [np.flatnonzero(data["group_id"][rows] == group)
                  for group in np.unique(data["group_id"][rows])]
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, domain_index]))
        draws = []
        if len(blocks) >= 2:
            for _ in range(args.bootstrap):
                indices = np.concatenate([blocks[index] for index in rng.integers(len(blocks), size=len(blocks))])
                draws.append(measurements(labels[indices], vit[indices], clip[indices], vit_threshold, clip_threshold)[0])
        domain_draws[domain] = draws
        per_domain[domain] = {"n": len(rows), "videos": len(np.unique(data["video_id"][rows])),
                              "independent_groups": len(blocks), "metrics": values,
                              "groups": counts, "bootstrap": intervals(draws)}
    missing = sorted(set(args.primary_domains) - set(per_domain))
    macro = {"domains": args.primary_domains, "missing_domains": missing, "complete": not missing}
    if not missing:
        names = ("vit_auc", "clip_auc", "delta_auc_clip_minus_vit")
        macro["metrics"] = {name: float(np.mean([per_domain[domain]["metrics"][name]
                                               for domain in args.primary_domains])) for name in names}
        draws = []
        if all(len(domain_draws[domain]) == args.bootstrap for domain in args.primary_domains):
            for iteration in range(args.bootstrap):
                draws.append({name: float(np.mean(values)) if all(value is not None for value in values) else None
                              for name in names
                              for values in [[domain_draws[domain][iteration][name] for domain in args.primary_domains]]})
        macro["bootstrap"] = intervals(draws)
    return {"thresholds": thresholds, "calibration_n": int(calibration.sum()),
            "per_domain": per_domain, "primary_macro": macro}, group_codes


def report(summary, provenance):
    lines = ["# exp_1：互补错误第一层诊断结果", "",
             "仅描述潜在互补，不证明推理时可识别救回样本，不证明融合结构有效。", "",
             f"ViT 基线类型：`{provenance['vit']['kind']}`；校准样本数：{summary['calibration_n']}。",
             "阈值仅按源域独立验证集的 balanced accuracy 确定，然后固定；分数越高越假。", ""]
    for branch, entry in summary["thresholds"].items():
        lines.append(f"- {branch} 阈值：{entry['value']:.8g}")
    lines.extend(["", "| 域 | 样本 | 双对 | 潜在救回 | 潜在误伤 | 双错 | ViT AUC | CLIP AUC | ΔAUC [95% CI] |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---|"])
    for domain, entry in summary["per_domain"].items():
        values, counts = entry["metrics"], entry["groups"]
        interval = entry["bootstrap"].get("delta_auc_clip_minus_vit", {}).get("ci95")
        cells = [f"{counts[name]['n']} ({counts[name]['fraction']:.1%})" for name in GROUPS]
        lines.append(f"| {domain} | {entry['n']} | " + " | ".join(cells)
                     + f" | {values['vit_auc']:.4f} | {values['clip_auc']:.4f} | {values['delta_auc_clip_minus_vit']:+.4f} {interval} |")
    lines.extend(["", "## 视频覆盖", "", "四组的视频数可重叠，不能相加作为总视频数。独立 group 可合并真实视频和关联伪造。", ""])
    for domain, entry in summary["per_domain"].items():
        lines.append(f"- {domain}：{entry['videos']} 视频，{entry['independent_groups']} 独立分组；"
                     + "，".join(f"{name}={entry['groups'][name]['videos']} 视频" for name in GROUPS))
    lines.extend(["", "## 解读边界", "",
                  "- 救回与误伤是两路独立分类的潜在机会，不是实际融合后的变化。",
                  "- oracle_correctness_accuracy 使用真实标签，只是不可部署的乐观正确率参考；不报告 oracle AUC。",
                  "- AUC 差比较两个独立检测器，不是融合增量；误差分组依赖固定阈值。",
                  "- bootstrap 对同一视频关联 group 的两路分数配对重采样，条件于固定模型和阈值；不含训练或阈值估计不确定性。",
                  "- 不显著不等于等价；只有一个独立 group 时不给区间。目标域不得用于调阈值或选模型。",
                  "", "## 主三域宏平均", "", "```json", json.dumps(summary["primary_macro"], ensure_ascii=False, indent=2), "```",
                  "", "## 权重与评测历史声明", "", "```json", json.dumps(provenance, ensure_ascii=False, indent=2), "```", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source", default="ffpp")
    parser.add_argument("--primary-domains", nargs="+", default=["cd2", "dfdcp", "wild"])
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261012)
    args = parser.parse_args()
    if args.bootstrap < 0 or args.seed < 0 or len(set(args.primary_domains)) != len(args.primary_domains):
        parser.error("Invalid bootstrap, seed or duplicate primary domains")
    allowed = Path(__file__).resolve().parent
    output = args.output.resolve()
    if output == allowed or allowed not in output.parents:
        parser.error("Output must be a new directory inside vit_module/exp_1")
    if output.exists():
        parser.error("Output already exists; refusing overwrite")
    data, calibration, provenance = load(args.input, args.provenance, args.source)
    summary, codes = analyze(data, calibration, args)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "samples.npz", **data, error_group=codes,
                        vit_pred=data["vit_score"] >= summary["thresholds"]["vit"]["value"],
                        clip_pred=data["clip_score"] >= summary["thresholds"]["clip"]["value"])
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "统计结果.md").write_text(report(summary, provenance), encoding="utf-8")
    config = {"arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              "input_sha256": digest(args.input), "provenance_sha256": digest(args.provenance),
              "code_sha256": digest(__file__), "numpy_version": np.__version__, "provenance": provenance}
    (output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copyfile(__file__, output / "analysis_code.py")
    hashes = {path.name: digest(path) for path in sorted(output.iterdir()) if path.is_file()}
    (output / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    (output / "COMPLETE").write_text("exp_1 paired statistics completed\n", encoding="utf-8")
    print(f"Completed statistics: {output / '统计结果.md'}")


if __name__ == "__main__":
    main()
