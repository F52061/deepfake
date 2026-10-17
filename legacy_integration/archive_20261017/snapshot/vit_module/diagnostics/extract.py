"""Extract auditable, paired detector observations without changing model weights."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np


PROMPTS = [
    "a face with consistent lighting and shadows",
    "a face with inconsistent lighting and shadows",
    "a face with natural eyes and teeth",
    "a face with distorted eyes and teeth",
    "a face with consistent skin texture",
    "a face with unnatural skin texture",
    "a face with natural facial boundaries",
    "a face with blending artifacts around its boundaries",
]


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path):
    with open(path, encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"path", "y", "domain", "split", "video_id"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"CSV requires {sorted(required)}; y=1 means fake")
    for row in rows:
        if row["y"] not in {"0", "1"} or row["split"] not in {"train", "val", "test"}:
            raise ValueError("Invalid label or split")
        if not row["video_id"] or not row["domain"]:
            raise ValueError("Explicit domain and video identity are required")
        if not Path(row["path"]).is_absolute() or not Path(row["path"]).is_file():
            raise ValueError(f"Image must exist at an absolute path: {row['path']}")
    if len({row["path"] for row in rows}) != len(rows):
        raise ValueError("Duplicate image paths")
    groups = {}
    for row in rows:
        group = (row["domain"], row["video_id"])
        if group in groups and groups[group] != row["split"]:
            raise ValueError(f"Video overlaps splits: {group}")
        groups[group] = row["split"]
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--clip", required=True, help="Local pretrained CLIP directory")
    parser.add_argument("--output", required=True, help="New output directory")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--fake-logit", type=int, choices=[0, 1], default=1)
    parser.add_argument("--save-tokens", action="store_true")
    parser.add_argument("--save-regions", action="store_true")
    parser.add_argument("--region-grid", type=int, default=3)
    parser.add_argument("--variants", nargs="+", default=["clean", "jpeg70", "blur", "resize", "color"])
    args = parser.parse_args()
    if args.batch_size < 1 or "clean" not in args.variants or len(set(args.variants)) != len(args.variants):
        parser.error("Positive batch size and unique variants including clean required")
    allowed = {"clean", "jpeg70", "blur", "resize", "color"}
    if not set(args.variants) <= allowed:
        parser.error(f"Variants must be in {sorted(allowed)}")
    if not 2 <= args.region_grid <= 14:
        parser.error("region-grid must be between 2 and 14")
    rows = read_manifest(args.manifest)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    import cv2
    import torch
    from transformers import CLIPModel, CLIPTokenizer

    torch.set_num_threads(1)
    cv2.setNumThreads(0)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from vit_module import vit_m2f2_detector_bridge as bridge
    from vit_module.flash_attn_shim.mha import MHA
    from vit_module.train_bridge_phase1 import IMG_TO_TENSOR

    bridge.MHA = MHA
    model = bridge.ViT_M2F2Det_Bridge(
        clip_text_encoder_name=args.clip, clip_vision_encoder_name=args.clip,
        hidden_size=768, pretrained=False,
    )
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    for key in ("model_state_dict", "state_dict"):
        if key in state:
            state = state[key]
            break
    state = {key[7:] if key.startswith("module.") else key: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    # Move to device WITHOUT a global dtype cast. TransformerEncoderBlock builds
    # its attention in bfloat16 on purpose (vit_m2f2_detector_bridge.py:116) and
    # casts its input to match at :129; forcing torch.float32 on the whole model
    # upcasts those weights and the attention then fails with
    # "expected scalar type Float but found BFloat16". Every other submodule is
    # already float32 from the constructor defaults. This matches how the
    # project itself loads the model (vit_module/_g16/run_g16.py:303).
    model.to(device=args.device).eval().requires_grad_(False)
    native = CLIPModel.from_pretrained(args.clip, local_files_only=True).eval().requires_grad_(False)
    tower = model.clip_vision_encoder.model.state_dict()
    native_tower = native.vision_model.state_dict()
    if set(tower) != {"vision_model." + key for key in native_tower}:
        raise ValueError("Native CLIP and detector tower keys differ")
    if any(not torch.equal(tower["vision_model." + key].cpu(), value) for key, value in native_tower.items()):
        raise ValueError("Checkpoint CLIP differs from native CLIP; semantic projection is not verified")
    tokenizer = CLIPTokenizer.from_pretrained(args.clip, local_files_only=True)
    with torch.inference_mode():
        text = native.get_text_features(**tokenizer(PROMPTS, padding=True, return_tensors="pt"))
        text = torch.nn.functional.normalize(text, dim=-1).to(args.device)
    visual_projection = native.visual_projection.to(args.device)
    np.savez_compressed(output / "semantic_parameters.npz", text_embeddings=text.cpu().numpy(),
                        visual_projection=visual_projection.weight.cpu().numpy(), prompts=np.array(PROMPTS))
    del native
    metadata = {
        "schema": 1, "label": "1=fake, 0=real", "arguments": vars(args),
        "checkpoint_sha256": file_hash(args.checkpoint), "manifest_sha256": file_hash(args.manifest),
        "prompts": PROMPTS, "native_clip_tower_verified": True,
        "semantic_definition": "cosine of official pooled CLIP embedding and fixed text prompts",
        "patch_semantics": "last-layer patches projected by CLS-trained projection; exploratory only",
        "torch": torch.__version__, "numpy": np.__version__, "opencv": cv2.__version__,
        "extraction_code_sha256": file_hash(__file__),
    }
    (output / "config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    head = {"weight": model.output.weight.cpu().numpy(), "bias": model.output.bias.cpu().numpy(),
            "proj_weight": model.deepfake_proj[0].weight.cpu().numpy(),
            "proj_bias": model.deepfake_proj[0].bias.cpu().numpy(),
            "norm_weight": model.deepfake_proj[1].weight.cpu().numpy(),
            "norm_bias": model.deepfake_proj[1].bias.cpu().numpy(),
            "norm_eps": np.array(model.deepfake_proj[1].eps), "fake_logit": np.array(args.fake_logit)}
    np.savez_compressed(output / "head.npz", **head)
    captured = {}
    handles = [
        model.deepfake_proj.register_forward_pre_hook(lambda module, inputs: captured.update(V=inputs[0].detach())),
        model.output.register_forward_pre_hook(lambda module, inputs: captured.update(F=inputs[0].detach())),
        model.vit.norm.register_forward_hook(lambda module, inputs, result: captured.update(vit_tokens=result.detach())),
        model.clip_vision_encoder.model.register_forward_hook(lambda module, inputs, result: captured.update(clip=result)),
    ]
    try:
        for variant in args.variants:
            for start in range(0, len(rows), args.batch_size):
                batch_rows = rows[start:start + args.batch_size]
                images = []
                for row in batch_rows:
                    image = cv2.imread(row["path"], cv2.IMREAD_COLOR)
                    if image is None:
                        raise ValueError(f"Unreadable image: {row['path']}")
                    image = cv2.resize(image, (336, 336))
                    if variant == "jpeg70":
                        success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 70])
                        if not success:
                            raise RuntimeError("JPEG encoding failed")
                        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                    elif variant == "blur":
                        image = cv2.GaussianBlur(image, (5, 5), 1.0)
                    elif variant == "resize":
                        image = cv2.resize(cv2.resize(image, (168, 168)), (336, 336))
                    elif variant == "color":
                        image = np.clip(image.astype(np.float32) * [0.95, 1.0, 1.05], 0, 255).astype(np.uint8)
                    images.append(IMG_TO_TENSOR(image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))["image"])
                with torch.inference_mode():
                    image_batch = torch.stack(images).to(args.device)
                    logits = model(image_batch)
                    if variant == "clean" and start == 0 and len(images) > 1:
                        preserved = captured.copy()
                        single_logits = model(image_batch[:1])
                        captured.update(preserved)
                        batch_error = float((single_logits - logits[:1]).abs().max().cpu())
                        if not torch.allclose(single_logits, logits[:1], atol=1e-4, rtol=1e-4):
                            raise ValueError(f"Batch-dependent predictions: {batch_error}; use --batch-size 1")
                        metadata["first_image_batch_independence_max_abs_error"] = batch_error
                    clip = captured["clip"]
                    semantic = torch.nn.functional.normalize(visual_projection(clip.pooler_output), dim=-1) @ text.T
                    patches = clip.hidden_states[-1][:, 1:]
                    patch_semantic = torch.nn.functional.normalize(visual_projection(patches), dim=-1) @ text.T
                data = {key: np.array([row[key] for row in batch_rows]) for key in ("path", "domain", "split", "video_id")}
                if variant == "clean":
                    data["image_sha256"] = np.array([file_hash(row["path"]) for row in batch_rows])
                data.update(y=np.array([int(row["y"]) for row in batch_rows]),
                            row_id=np.arange(start, start + len(batch_rows)), variant=np.array(variant),
                            V=captured["V"].cpu().numpy(), F=captured["F"].cpu().numpy(),
                            C=clip.hidden_states[-2][:, 0].cpu().numpy(), logits=logits.cpu().numpy(),
                            S=semantic.cpu().numpy(), patch_S=patch_semantic.cpu().numpy())
                if args.save_tokens:
                    data["clip_tokens"] = patches.cpu().numpy()
                    data["vit_tokens"] = captured["vit_tokens"].cpu().numpy()
                    for name, tokens in model.vit_block_outputs.items():
                        data["vit_" + name] = tokens.cpu().numpy()
                if args.save_regions:
                    from local_features import pool_tokens
                    data["vit_regions"] = pool_tokens(
                        captured["vit_tokens"].cpu().numpy(), args.region_grid, has_cls=True)
                    data["clip_regions"] = pool_tokens(patches.cpu().numpy(), args.region_grid)
                    data["clip_last_cls"] = clip.hidden_states[-1][:, 0].cpu().numpy()
                np.savez_compressed(output / f"{variant}_{start:08d}.npz", **data)
                print(f"{variant}: {start + len(batch_rows)}/{len(rows)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    (output / "config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output / "COMPLETE").write_text("Extraction completed\n", encoding="utf-8")


if __name__ == "__main__":
    main()
