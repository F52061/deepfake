"""Read-only Bridge layout audit; synthetic checks are NOT detection results.

Load the actual small classes by AST, avoiding encoder downloads/construction.
Run on CPU with one thread. Existing models and checkpoints are never modified.
"""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"

import torch
from torch import nn


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_classes(source, mha):
    names = {"View", "TransformerEncoderBlock", "BridgeAdapter_Proj_ViT"}
    tree = ast.parse(source.read_text(encoding="utf-8-sig"))
    nodes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
    if {node.name for node in nodes} != names:
        raise ValueError(f"Missing audit classes in {source}")
    namespace = {"torch": torch, "nn": nn, "MHA": mha}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace


def layout_checks(source, mha):
    classes = load_classes(source, mha)
    torch.manual_seed(20261016)
    block = classes["TransformerEncoderBlock"](64, 4, 32, dropout=0).eval()
    x = torch.randn(7, 3, 64)
    changed = x.clone()
    changed[:, 1:] = torch.randn_like(changed[:, 1:]) * 4

    def block_forward(value, fixed):
        return block(value.transpose(0, 1).contiguous()).transpose(0, 1).contiguous() if fixed else block(value)

    block_metrics = {}
    with torch.no_grad():
        for fixed in (False, True):
            base = block_forward(x, fixed)
            solo = block_forward(x[:, :1].contiguous(), fixed)
            peer = block_forward(changed, fixed)
            block_metrics["fixed_layout" if fixed else "legacy_layout"] = {
                "alone_vs_batch_max_abs": float((base[:, :1] - solo).abs().max()),
                "change_companions_max_abs": float((base[:, :1] - peer[:, :1]).abs().max()),
            }
    # Per-sample scalar backward checks whether peers can influence sample 0.
    for fixed in (False, True):
        value = x.clone().requires_grad_(True)
        scalar = block_forward(value, fixed)[:, 0, 0].sum()
        grad = torch.autograd.grad(scalar, value)[0]
        block_metrics["fixed_layout" if fixed else "legacy_layout"]["peer_input_grad_norm"] = float(grad[:, 1:].norm())

    # Isolate the packing bug AFTER attention; no attention is involved here.
    projection = classes["BridgeAdapter_Proj_ViT"](64, 8, 7).eval()
    dead = copy.deepcopy(projection)
    projection.bridge_adapter_proj = nn.Sequential(*list(projection.bridge_adapter_proj.children())[:-1])

    def project(value, fixed):
        if not fixed:
            return projection(value, value.shape[1])
        scores = projection.bridge_adapter_proj(value.transpose(0, 1).contiguous())
        return projection.bridge_adapter_reduction(scores.reshape(value.shape[1], -1))

    packing_metrics = {}
    with torch.no_grad():
        for fixed in (False, True):
            base = project(x, fixed)
            solo = project(x[:, :1].contiguous(), fixed)
            peer = project(changed, fixed)
            packing_metrics["fixed_packing" if fixed else "legacy_packing"] = {
                "alone_vs_batch_max_abs": float((base[:1] - solo).abs().max()),
                "change_companions_max_abs": float((base[:1] - peer[:1]).abs().max()),
            }
        dead_difference = float((dead(x, 3) - dead(changed, 3)).abs().max())
    value = x.clone().requires_grad_(True)
    dead(value, 3)[0, 0].backward()
    dead_grad = float(value.grad.norm())

    # Three stages with the original concatenate/permute pattern, same weights
    # in all conditions. Only the three named defects are intervened on.
    blocks = nn.ModuleList([copy.deepcopy(block) for _ in range(3)]).eval()
    final = classes["BridgeAdapter_Proj_ViT"](64, 8, 21).eval()
    stages = [torch.randn(3, 7, 64) for _ in range(3)]

    def chain(inputs, remove_ln, fix_attention, fix_packing):
        previous = None
        for stage, module in zip(inputs, blocks):
            combined = stage if previous is None else torch.cat((previous.transpose(0, 1), stage), dim=1)
            seq = combined.transpose(0, 1).contiguous()
            previous = module(combined).transpose(0, 1).contiguous() if fix_attention else module(seq)
        if not remove_ln:
            return final(previous, inputs[0].shape[0])
        token_linear = final.bridge_adapter_proj[1]
        scalars = token_linear(previous)
        packed = scalars.transpose(0, 1).contiguous().reshape(inputs[0].shape[0], -1) if fix_packing else scalars.reshape(inputs[0].shape[0], -1)
        return final.bridge_adapter_reduction(packed)

    chain_metrics = {}
    with torch.no_grad():
        for name, switches in (("legacy", (False, False, False)), ("remove_ln_only", (True, False, False)),
                               ("plus_attention_layout", (True, True, False)), ("all_three_fixed", (True, True, True))):
            base = chain(stages, *switches)
            solo = chain([stage[:1].contiguous() for stage in stages], *switches)
            peers = [torch.cat((stage[:1], torch.randn_like(stage[1:]) * 4), 0) for stage in stages]
            changed_output = chain(peers, *switches)
            chain_metrics[name] = {
                "alone_vs_batch_max_abs": float((base[:1] - solo).abs().max()),
                "change_companions_max_abs": float((base[:1] - changed_output[:1]).abs().max()),
            }
    # BF16 can vary slightly between GEMM shapes; companion replacement uses
    # exactly the same shape and has the stronger independence assertion.
    assert block_metrics["legacy_layout"]["change_companions_max_abs"] > 1e-3
    assert block_metrics["legacy_layout"]["peer_input_grad_norm"] > 1e-3
    assert block_metrics["fixed_layout"]["change_companions_max_abs"] < 1e-6
    assert block_metrics["fixed_layout"]["peer_input_grad_norm"] == 0
    assert packing_metrics["legacy_packing"]["change_companions_max_abs"] > 1e-3
    assert packing_metrics["fixed_packing"]["change_companions_max_abs"] < 1e-6
    assert dead_difference == 0 and dead_grad == 0
    assert chain_metrics["remove_ln_only"]["change_companions_max_abs"] > 1e-3
    assert chain_metrics["plus_attention_layout"]["change_companions_max_abs"] > 1e-3
    assert chain_metrics["all_three_fixed"]["change_companions_max_abs"] < 1e-6
    return {"source_sha256": sha256(source), "attention": block_metrics,
            "projection_after_removing_ln": packing_metrics,
            "legacy_ln_input_gradient_norm": dead_grad, "legacy_ln_companion_difference": dead_difference,
            "three_stage_chain": chain_metrics, "assertions": "passed"}


def checkpoint_audit(native_path, fusion_path):
    native = torch.load(native_path, map_location="cpu", weights_only=True)
    fusion = torch.load(fusion_path, map_location="cpu", weights_only=True)
    for key in ("model_state_dict", "state_dict"):
        if key in native:
            native = native[key]
        if key in fusion:
            fusion = fusion[key]
    matched, unequal, missing = [], [], []
    head = {}
    for name, value in native.items():
        name = name.removeprefix("module.")
        if name.startswith("model.head."):
            head[name] = list(value.shape)
            continue
        if not name.startswith("model."):
            continue
        target = "vit." + name[len("model."):]
        if target not in fusion:
            missing.append(target)
        elif not torch.equal(value.to(fusion[target].dtype), fusion[target]):
            unequal.append(target)
        else:
            matched.append(target)
    return {"native_path": str(native_path), "fusion_path": str(fusion_path),
            "native_sha256": sha256(native_path), "fusion_sha256": sha256(fusion_path),
            "backbone_equal_keys": len(matched), "backbone_unequal_keys": unequal,
            "backbone_missing_keys": missing, "native_head_keys_not_loaded_by_fusion_loader": head,
            "note": "Equality does not establish acc.txt provenance or training/selection history."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-checkpoint", type=Path)
    parser.add_argument("--fusion-checkpoint", type=Path)
    args = parser.parse_args()
    if bool(args.native_checkpoint) != bool(args.fusion_checkpoint):
        parser.error("Provide both checkpoints or neither")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    shim = args.root / "vit_module/flash_attn_shim/mha.py"
    spec = importlib.util.spec_from_file_location("audit_mha", shim)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = {"kind": "implementation_audit_not_detection_performance", "device": "cpu",
              "threads": 1, "python": sys.version, "torch": torch.__version__,
              "audit_sha256": sha256(__file__), "shim_sha256": sha256(shim), "models": {}}
    for filename in ("vit_m2f2_detector_bridge.py", "vit_m2f2_detector_unified.py"):
        report["models"][filename] = layout_checks(args.root / "vit_module" / filename, module.MHA)
    if args.native_checkpoint:
        report["checkpoint"] = checkpoint_audit(args.native_checkpoint, args.fusion_checkpoint)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output / "architecture_audit.py").write_bytes(Path(__file__).read_bytes())
    (args.output / "COMPLETE").write_text("Implementation checks passed; no detection experiment run.\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
