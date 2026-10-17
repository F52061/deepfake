"""One-thread CPU audit on real server images, with old weights; no training.

Cache actual reduced image tokens once. Then test the identical image alone,
with different companions, and in another batch position. This measures sample
isolation only; repaired outputs are NOT detection performance evidence.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-unused-text", action="store_true", help="Bypass unused text output for incompatible transformers APIs")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.chdir(args.root)
    sys.path.insert(0, str(args.root))
    import csv
    import cv2
    import numpy as np
    import torch
    from torch import nn
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(0)
    from vit_module import vit_m2f2_detector_bridge as bridge
    from vit_module.flash_attn_shim.mha import MHA
    from bridge_layout_candidate import apply_layout_candidate

    start = time.monotonic()
    bridge.MHA = MHA  # Same non-flash implementation used by historical extraction.
    clip = str(args.root / "checkpoints/clip-vit-large-patch14-336")
    model = bridge.ViT_M2F2Det_Bridge(
        clip_text_encoder_name=clip, clip_vision_encoder_name=clip,
        hidden_size=768, pretrained=False,
    ).eval().requires_grad_(False)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    if args.skip_unused_text:
        # This detector computes text_proj(text) but never reads the result.
        # The newer deepfake_1 transformers rejects its old return_dict kwarg.
        # All audit conditions share this bypass; vision/bridge are unchanged.
        model.clip_text_encoder.forward = lambda: torch.zeros(1, model.text_proj[0].in_features)
    rows = list(csv.DictReader(args.manifest.open(encoding="utf-8-sig", newline="")))
    selected = []
    for label in ("0", "1"):
        used = set()
        for row in rows:
            video = str(Path(row["path"]).parent)
            if row["y"] == label and row["domain"] == "ffpp" and row["split"] == "test" and video not in used:
                selected.append(row)
                used.add(video)
                if len(used) == 2:
                    break
        if len(used) != 2:
            raise ValueError("Need two source-test videos for each class")
    images = []
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    for row in selected:
        image = cv2.imread(row["path"])
        if image is None:
            raise ValueError(row["path"])
        image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (336, 336))
        # Same formula as albumentations Normalize in the training pipeline.
        images.append(torch.from_numpy(((image.astype(np.float32) / 255 - mean) / std).transpose(2, 0, 1).copy()))
    vit_tokens, clip_tokens, captured = [], [], {}
    handles = [module.register_forward_hook(lambda m, i, o: vit_tokens.append(o.detach().clone())) for module in model.linear_vit_lst]
    handles += [model.clip_reduction.register_forward_hook(lambda m, i, o: clip_tokens.append(o.detach().clone())),
                model.output.register_forward_pre_hook(lambda m, i: captured.update(features=i[0].detach().clone()))]
    with torch.no_grad():
        original_logits = model(torch.stack(images))
    for handle in handles:
        handle.remove()
    stages = [torch.cat((v, c), dim=1) for v, c in zip(vit_tokens, clip_tokens)]
    if len(stages) != 3:
        raise AssertionError("Expected three stages")

    def run_branch(network, indices):
        previous = None
        for stage, block in zip(stages, network.bridge_adapter):
            current = stage[indices]
            combined = current if previous is None else torch.cat((previous.transpose(0, 1), current), dim=1)
            previous = block(combined.transpose(0, 1).contiguous())
        return network.bridge_adapter_proj(previous, len(indices)) * network.clip_text_alpha

    # Clone only small bridge modules to avoid duplicating CLIP/ViT encoders.
    small = nn.Module()
    small.bridge_adapter = copy.deepcopy(model.bridge_adapter)
    small.bridge_adapter_proj = copy.deepcopy(model.bridge_adapter_proj)
    small.clip_text_alpha = nn.Parameter(model.clip_text_alpha.detach().clone(), requires_grad=False)
    small.eval()
    results = {}
    with torch.no_grad():
        replay = run_branch(small, [0, 1, 2, 3])
        replay_error = float((replay - captured["features"][:, 768:896]).abs().max())
        if replay_error > 1e-5:
            raise AssertionError(f"Legacy replay differs: {replay_error}")
        for name in ("legacy", "remove_ln_only", "all_three_fixed"):
            current = copy.deepcopy(small)
            if name == "remove_ln_only":
                current.bridge_adapter_proj.bridge_adapter_proj = nn.Sequential(*list(current.bridge_adapter_proj.bridge_adapter_proj.children())[:2])
            elif name == "all_three_fixed":
                apply_layout_candidate(current)
            solo = run_branch(current, [0])[0]
            ab = run_branch(current, [0, 1])[0]
            ac = run_branch(current, [0, 2])[0]
            ba = run_branch(current, [1, 0])[1]
            weight = model.output.weight[:, 768:896]
            results[name] = {
                "alone_vs_batch_embed_max_abs": float((solo - ab).abs().max()),
                "different_companion_embed_max_abs": float((ab - ac).abs().max()),
                "batch_position_embed_max_abs": float((ab - ba).abs().max()),
                "different_companion_logits_max_abs": float((weight @ (ab - ac)).abs().max()),
            }
        assert results["remove_ln_only"]["different_companion_embed_max_abs"] > 1e-5
        assert results["all_three_fixed"]["different_companion_embed_max_abs"] < 1e-5
        assert results["all_three_fixed"]["batch_position_embed_max_abs"] < 1e-5

    report = {"kind": "real_image_implementation_audit_not_auc", "device": "cpu", "threads": 1,
              "python": sys.version, "torch": torch.__version__, "seconds": time.monotonic() - start,
              "image_rows": selected, "legacy_branch_replay_max_abs": replay_error,
              "conditions": results, "original_logits": original_logits.tolist(),
              "checkpoint": str(args.checkpoint), "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              "unused_text_bypassed": args.skip_unused_text,
              "limitations": ["Four source-test images; no fitting, no AUC, no generalization claim.",
                              "Cached encoder tokens isolate bridge behavior in eval mode.",
                              "Legacy trained bridge weights are not a trained repaired model."]}
    for filename in ("real_image_layout_audit.py", "bridge_layout_candidate.py"):
        source = Path(__file__).with_name(filename)
        (args.output / filename).write_bytes(source.read_bytes())
    (args.output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(args.output / "real_reduced_tokens.npz", **{f"stage_{i}": x.numpy() for i, x in enumerate(stages)})
    (args.output / "COMPLETE").write_text("Real-image sample-isolation audit passed; no training or AUC.\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
