"""
Dimension validation script for ViT_M2F2Det_Bridge.

Tests:
  1. Model instantiation
  2. Forward pass with random input
  3. Output shape verification
  4. Intermediate tensor dimension checks
  5. load_vit_backbone (if checkpoint available)
  6. Parameter count statistics

Usage:
    <python.exe> vit_module/test_bridge_dims.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn


def test_instantiation():
    """Test model can be instantiated.

    Uses load_vision_encoder=False to avoid HuggingFace network call
    (THis is the LLaVA integration mode where CLIP features are provided externally).
    """
    print("=" * 60)
    print("Test 1: Instantiation (load_vision_encoder=False)")
    print("=" * 60)
    try:
        from vit_module.vit_m2f2_detector_bridge import ViT_M2F2Det_Bridge
        model = ViT_M2F2Det_Bridge(
            hidden_size=768,
            load_vision_encoder=False,
        )
        print("  ViT_M2F2Det_Bridge instantiated OK")
        return model
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_forward(model):
    """Test forward pass with external CLIP features (LLaVA integration mode)."""
    print("\n" + "=" * 60)
    print("Test 2: Forward pass (external CLIP features)")
    print("=" * 60)
    model.eval()
    B = 2
    x = torch.randn(B, 3, 336, 336)

    # Provide synthetic CLIP features as 4-tuple (clip_0, clip_1, clip_2, clip_vision_features)
    fake_clip_0   = torch.randn(B, 576, 1024)   # CLIP layer 6  patches
    fake_clip_1   = torch.randn(B, 576, 1024)   # CLIP layer 10 patches
    fake_clip_2   = torch.randn(B, 576, 1024)   # CLIP layer 14 patches
    fake_clip_vis = torch.randn(B, 577, 1024)   # CLIP final features (with CLS)
    clip_features = (fake_clip_0, fake_clip_1, fake_clip_2, fake_clip_vis)

    try:
        with torch.no_grad():
            out = model(x, clip_vision_features=clip_features)
        print(f"  Input shape:        {x.shape}")
        print(f"  Output shape:       {out.shape}")
        assert out.shape == (B, 2), f"Expected ({B},2), got {out.shape}"
        print(f"  Logits:             {out}")
        print(f"  Softmax:            {torch.softmax(out, dim=-1)}")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_intermediate_dims(model):
    """Print all intermediate tensor shapes (via hook debugging)."""
    print("\n" + "=" * 60)
    print("Test 3: Intermediate dimensions")
    print("=" * 60)
    model.eval()
    x = torch.randn(1, 3, 336, 336)

    # ViT hook outputs
    with torch.no_grad():
        _ = model.vit.forward_features(model._preprocess_for_vit(x))

    for name in ["b_1", "b_2", "b_3"]:
        if name in model.vit_block_outputs:
            shape = model.vit_block_outputs[name].shape
            print(f"  ViT hook {name}:  {shape}")
            patches = model.vit_block_outputs[name][:, 1:, :]
            print(f"    → patches:     {patches.shape}")

    # CLIP encoder: skip (load_vision_encoder=False)
    if model.clip_vision_encoder is not None:
        clip_0, clip_1, clip_2, clip_feat = model.clip_vision_encoder(x)
        print(f"  CLIP layer  6:   {clip_0.shape}")
        print(f"  CLIP layer 10:   {clip_1.shape}")
        print(f"  CLIP layer 14:   {clip_2.shape}")
        print(f"  CLIP final:      {clip_feat.shape}")
    else:
        print(f"  CLIP encoder:     external (load_vision_encoder=False)")

    print(f"\n  BridgeAdapter dimensions (expected):")
    print(f"    Stage 0: vit(196) + clip(576) = 772 tokens")
    print(f"    Stage 1: 772 + 196 + 576 = 1544 tokens")
    print(f"    Stage 2: 1544 + 196 + 576 = 2316 tokens")
    print(f"    BridgeAdapter_Proj: 2316 -> 128")
    print(f"    Cat: hidden(768) + 128 + hidden(768) = 1664 -> 2")


def test_params(model):
    """Print parameter statistics."""
    print("\n" + "=" * 60)
    print("Test 4: Parameter statistics")
    print("=" * 60)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    print(f"  Total params:     {total/1e6:.2f} M")
    print(f"  Trainable:        {trainable/1e6:.2f} M")
    print(f"  Frozen:           {frozen/1e6:.2f} M")

    # Per-component breakdown
    components = [
        ("ViT backbone", model.vit),
        ("deepfake_proj", model.deepfake_proj),
        ("CLIP text enc", model.clip_text_encoder),
        ("CLIP vision enc", model.clip_vision_encoder if model.clip_vision_encoder else None),
        ("vision_proj", model.vision_proj),
        ("text_proj", model.text_proj),
        ("clip_reduction", model.clip_reduction),
        ("linear_vit_1/2/3", model.linear_vit_lst),
        ("bridge_adapter (3×)", model.bridge_adapter),
        ("bridge_adapter_proj", model.bridge_adapter_proj),
        ("output (classifier)", model.output),
    ]
    print(f"\n  Per-component:")
    for name, module in components:
        if module is None:
            print(f"    {name:25s}: N/A")
        else:
            params = sum(p.numel() for p in module.parameters())
            print(f"    {name:25s}: {params/1e6:6.2f} M")


def test_load_checkpoint(model):
    """Test load_vit_backbone if checkpoint exists."""
    print("\n" + "=" * 60)
    print("Test 5: Load ViT backbone checkpoint")
    print("=" * 60)

    candidate_paths = [
        r"E:\Cross-domain_authentication_verification\PDI\results\Ama1_aps1_1\net_050.pth",
        "./checkpoints/vit_backbone.pth",
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            print(f"  Found checkpoint: {path}")
            try:
                missing, unexpected = model.load_vit_backbone(path, verbose=True)
                print(f"  Missing keys: {len(missing)}")
                print(f"  Unexpected keys: {len(unexpected)}")
            except Exception as e:
                print(f"  Load failed: {e}")
            return
    print("  No checkpoint found (skipped).")
    print(f"  Tried: {candidate_paths}")


def test_trainability(model):
    """Verify assign_lr_dict_list returns valid optimizer groups."""
    print("\n" + "=" * 60)
    print("Test 6: assign_lr_dict_list")
    print("=" * 60)
    try:
        param_groups = model.assign_lr_dict_list(lr=1e-4)
        print(f"  Parameter groups: {len(param_groups)}")

        total_trainable = 0
        for i, group in enumerate(param_groups):
            n_params = sum(p.numel() for p in group['params'])
            lr_val = group['lr']
            total_trainable += n_params
            print(f"    Group {i}: {n_params/1e6:6.2f}M params, lr={lr_val}")

        print(f"  Total trainable in groups: {total_trainable/1e6:.2f}M")

        # Verify optimizer can be created
        opt = torch.optim.Adam(param_groups, weight_decay=1e-6)
        print(f"  Adam optimizer created OK")
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════

def main():
    model = test_instantiation()
    if model is None:
        return

    test_forward(model)
    test_intermediate_dims(model)
    test_params(model)
    test_load_checkpoint(model)
    test_trainability(model)

    print("\n" + "=" * 60)
    print("All tests completed.")


if __name__ == "__main__":
    main()
