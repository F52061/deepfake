"""Server smoke: strict native load, bare-ViT parity, and actual LLaVA token API.

No training, no AUC. The tiny language model only checks wiring, not language
quality. The detector itself uses the full native ViT-B/16 and real checkpoint.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(variable, "1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "integration/configs/vit_native.json")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--images", nargs="+", type=Path)
    parser.add_argument("--device", default="cpu", help="cpu or an explicitly selected logical cuda:N")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    import torch
    from PIL import Image
    from integration.models.vit_native import NativeViTDetector, build_native_backbone, _native_state, file_sha256
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    checkpoint = args.checkpoint or Path(config["native_checkpoint"])
    images = args.images or [Path(value) for value in config["smoke_images"]]
    if not images or len(images) > 16:
        parser.error("Provide 1..16 images for this smoke test")
    if not checkpoint.is_file() or any(not path.is_file() for path in images):
        parser.error("Checkpoint and every image must exist; edit config or pass --checkpoint/--images")
    output = args.output or ROOT / "integration/runs" / f"vit_native_smoke_{time.time_ns()}"
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    detector = NativeViTDetector(config["native_vit_fake_logit_index"], config["native_vit_preprocessing"])
    detector.to(args.device)
    loaded = detector.load_checkpoint(checkpoint, config.get("native_checkpoint_sha256"))
    pil_images = []
    for path in images:
        with Image.open(path) as image:
            pil_images.append(image.convert("RGB"))
    pixels = detector.image_processor.preprocess(pil_images)["pixel_values"]
    with torch.no_grad():
        logits = detector(pixels)
        single = detector(pixels[:1])
        detector.train()
        after_train_call = detector(pixels)
    assert not detector.training and not detector.vit.training
    assert not any(p.requires_grad for p in detector.parameters())
    torch.testing.assert_close(logits, after_train_call, rtol=0, atol=0)
    torch.testing.assert_close(logits[:1], single, rtol=1e-4, atol=1e-4)
    bare = build_native_backbone().to(args.device).eval()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    bare.load_state_dict(_native_state(state, bare.state_dict()), strict=True)
    with torch.no_grad():
        reference = bare(pixels.to(args.device))
    torch.testing.assert_close(logits, reference, rtol=0, atol=0)
    expected_logits = logits.cpu()
    standalone_error = float((logits - reference).abs().max())
    single_error = float((logits[:1] - single).abs().max())
    del bare, state, detector, logits, reference, single, after_train_call

    # Exercise the real integration class with a small random language model.
    # No mm_vision_tower field: no CLIP models or LLM downloads are constructed.
    from llava.model.language_model.llava_llama import LlavaConfig, LlavaLlamaForCausalLMDeepfake
    language_config = LlavaConfig(vocab_size=64, hidden_size=64, intermediate_size=128,
                                 num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
                                 pretraining_tp=1, mm_vision_select_feature="cls_patch",
                                 deepfake_model_name="vit_native",
                                 native_vit_fake_logit_index=config["native_vit_fake_logit_index"],
                                 native_vit_preprocessing=config["native_vit_preprocessing"],
                                 native_checkpoint_sha256=config.get("native_checkpoint_sha256"))
    model = LlavaLlamaForCausalLMDeepfake(language_config).to(args.device).eval()
    model.load_deepfake_encoder(str(checkpoint), verbose=False)
    model._ensure_processors()
    assert model.get_vision_tower() is None
    assert model.processors["deepfake_processor"] is model.deepfake_encoder.image_processor
    with torch.no_grad():
        tokens = model.encode_deepfake_tokens({"images": pixels})
    integrated_logits = model._deepfake_logits_for_loss.cpu()
    torch.testing.assert_close(integrated_logits, expected_logits, rtol=0, atol=0)
    assert tuple(tokens.shape) == (len(images), 1, 64) and torch.isfinite(tokens).all()
    from llava.constants import DEEPFAKE_TOKEN_INDEX
    ids = torch.tensor([[1, DEEPFAKE_TOKEN_INDEX, 2]] * len(images), device=args.device)
    with torch.no_grad():
        language_output = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                                deepfake_inputs=pil_images, use_cache=False)
    torch.testing.assert_close(model._deepfake_logits_for_loss.cpu(), expected_logits, rtol=0, atol=0)
    assert tuple(language_output.logits.shape) == (len(images), 3, 64) and torch.isfinite(language_output.logits).all()
    assert not any(name.startswith("legacy_integration.") for name in sys.modules)
    report = {
        "kind": "native_vit_integration_smoke_not_performance", "device": args.device,
        "input_is_test_fixture": config.get("input_is_test_fixture", False),
        "checkpoint": loaded, "bare_vit_vs_detector_max_abs": standalone_error,
        "bare_vit_vs_llava_detector_max_abs": float((integrated_logits - expected_logits).abs().max()),
        "alone_vs_batch_max_abs": single_error, "token_shape": list(tokens.shape),
        "detector_has_clip_or_bridge": False,
        "language_interface": "real LLaVA class; tiny random LLM token API and language forward without CLIP",
        "language_forward_shape": list(language_output.logits.shape),
        "native_logits": integrated_logits.tolist(),
        "fake_probability_provisional_label_order": integrated_logits.softmax(-1)[:, config["native_vit_fake_logit_index"]].tolist(),
        "images": [{"path": str(path), "sha256": file_sha256(path)} for path in images],
        "preprocessing": config["native_vit_preprocessing"],
        "preprocessing_verified_against_pdi": config.get("preprocessing_verified_against_pdi", False),
        "native_label_order_verified_against_pdi": config.get("native_label_order_verified_against_pdi", False),
        "torch": torch.__version__, "seconds": time.monotonic() - start,
        "config_sha256": file_sha256(args.config), "script_sha256": file_sha256(__file__),
        "note": "Same pixels/head, not reproduction of acc.txt; no CLIP fusion, training, AUC or explanation-quality claim.",
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "input_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output / "COMPLETE").write_text("Native detector and LLaVA token interface checks passed.\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
