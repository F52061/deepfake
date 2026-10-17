"""Optional full LLaVA run with native ViT detection and visual-only CLIP.

Runtime check only: a base LLaVA has an unaligned new detection projector.
Old Bridge LoRA/merged checkpoints are not accepted as native-ViT checkpoints.
Select visible GPUs in the shell before starting Python. No training or saving
model weights. Use smoke_vit_native.py first.
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
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "integration/configs/vit_native.json")
    parser.add_argument("--llava-model", type=Path, required=True, help="Local base LLaVA, not old Bridge LoRA/merged model")
    parser.add_argument("--clip", type=Path, required=True, help="Local CLIP tower, used for language vision only")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-memory-gib", type=int, default=8, help="Allocation limit per visible GPU")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--question", default="Determine the authenticity of this image and explain your answer.")
    args = parser.parse_args()
    import torch
    from PIL import Image
    from transformers import AutoTokenizer
    from llava.model.language_model.llava_llama import LlavaConfig, LlavaLlamaForCausalLMDeepfake
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_hybrid_token
    from integration.models.vit_native import file_sha256
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        parser.error("This full-model script requires CUDA; use the CPU/native smoke first")
    native_config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    checkpoint = args.checkpoint or Path(native_config["native_checkpoint"])
    if not args.llava_model.is_dir() or not args.clip.is_dir() or not args.image.is_file() or not checkpoint.is_file():
        parser.error("All model directories, checkpoint and image must exist locally")
    if not 1 <= args.max_new_tokens <= 512 or args.max_memory_gib < 1:
        parser.error("Use max-new-tokens 1..512 and a positive GPU memory budget")
    cfg = LlavaConfig.from_pretrained(str(args.llava_model), local_files_only=True)
    if getattr(cfg, 'deepfake_model_name', None) not in (None, 'vit_native'):
        parser.error("Use the original base LLaVA, not a legacy fusion model; old adapter alignment is not transferable")
    cfg.deepfake_model_name = "vit_native"
    cfg.native_vit_fake_logit_index = native_config["native_vit_fake_logit_index"]
    cfg.native_vit_preprocessing = native_config["native_vit_preprocessing"]
    cfg.native_checkpoint_sha256 = native_config.get("native_checkpoint_sha256")
    cfg.mm_vision_tower = str(args.clip.resolve())
    cfg.mm_vision_select_feature = "cls_patch"
    cfg.mm_vision_select_layer = getattr(cfg, "mm_vision_select_layer", -2)
    output = args.output or ROOT / "integration/runs" / f"llava_vit_runtime_{time.time_ns()}"
    output.mkdir(parents=True, exist_ok=False)
    model, loading = LlavaLlamaForCausalLMDeepfake.from_pretrained(
        str(args.llava_model), config=cfg, local_files_only=True, torch_dtype=torch.float16,
        device_map="auto", low_cpu_mem_usage=True, output_loading_info=True,
        max_memory={**{i: f"{args.max_memory_gib}GiB" for i in range(torch.cuda.device_count())}, "cpu": "32GiB"},
    )
    allowed_missing = ("deepfake_encoder.", "deepfake_projector.", "model.vision_tower.")
    bad_missing = [key for key in loading.get("missing_keys", []) if not key.startswith(allowed_missing)]
    # The language vision tower is loaded explicitly from --clip, independently
    # of any delayed vision keys in the LLaVA checkpoint. Record all such keys.
    bad_unexpected = [key for key in loading.get("unexpected_keys", []) if not key.startswith("model.vision_tower.")]
    if bad_missing or bad_unexpected or loading.get("mismatched_keys") or loading.get("error_msgs"):
        raise RuntimeError(f"Unexpected base LLaVA loading mismatch: missing={bad_missing[:8]}, unexpected={bad_unexpected[:8]}")
    native_report = model.load_deepfake_encoder(str(checkpoint))
    tower = model.get_vision_tower()
    if not tower.is_loaded:
        tower.load_model(device_map="cuda:0")
    tower.to(device="cuda:0", dtype=torch.float16)
    model.eval()
    model._ensure_processors()
    tokenizer = AutoTokenizer.from_pretrained(str(args.llava_model), use_fast=False, local_files_only=True)
    with Image.open(args.image) as source:
        image = source.convert("RGB")
    pixels = model.deepfake_encoder.image_processor.preprocess(image)["pixel_values"]
    with torch.no_grad():
        direct_logits = model.deepfake_encoder(pixels).detach().float().cpu()
    conversation = conv_templates["vicuna_v1"].copy()
    conversation.append_message(conversation.roles[0], "<image>\n<deepfake>\n" + args.question)
    conversation.append_message(conversation.roles[1], None)
    prompt_ids = tokenizer_hybrid_token(conversation.get_prompt(), tokenizer, return_tensors="pt").unsqueeze(0)
    prompt_ids = prompt_ids.to(model.get_input_embeddings().weight.device)
    # Decoder forwards reset the training-loss cache; capture actual detector
    # output during generation instead of reading a possibly stale loss field.
    captured_logits = []
    handle = model.deepfake_encoder.register_forward_hook(
        lambda module, inputs, result: captured_logits.append(result.detach().float().cpu()))
    try:
        with torch.no_grad():
            generated = model.generate(inputs=prompt_ids, images=[image], deepfake_inputs=[image],
                                       do_sample=False, max_new_tokens=args.max_new_tokens, use_cache=True)
    finally:
        handle.remove()
    if len(captured_logits) != 1:
        raise RuntimeError(f"Expected exactly one detector call, observed {len(captured_logits)}")
    framework_logits = captured_logits[0]
    torch.testing.assert_close(framework_logits, direct_logits, atol=1e-4, rtol=1e-4)
    report = {
        "kind": "full_llava_native_vit_runtime_not_quality", "native_checkpoint": native_report,
        "llava_model": str(args.llava_model), "clip_visual_tower": str(args.clip),
        "clip_used_by_detector": False, "image": str(args.image), "image_sha256": file_sha256(args.image),
        "native_logits": framework_logits.tolist(),
        "direct_vs_framework_logits_max_abs": float((framework_logits - direct_logits).abs().max()),
        "text": tokenizer.decode(generated[0], skip_special_tokens=True),
        "detection_projector_newly_initialized": any(key.startswith("deepfake_projector.") for key in loading.get("missing_keys", [])),
        "alignment_status": "not validated; generation is only a runtime check",
        "preprocessing_verified_against_pdi": native_config.get("preprocessing_verified_against_pdi", False),
        "native_label_order_verified_against_pdi": native_config.get("native_label_order_verified_against_pdi", False),
        "loading_info": loading, "torch": torch.__version__, "config_sha256": file_sha256(args.config),
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (output / "COMPLETE").write_text("Runtime and detector parity passed; language quality not evaluated.\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
