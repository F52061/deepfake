"""Synthetic correctness checks. These are not deepfake performance results."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from integration.models import vit_native


class TinyViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = nn.Linear(3, 8)
        self.drop = nn.Dropout(0.8)
        self.head = nn.Linear(8, 2)

    def forward(self, x):
        return self.head(self.drop(self.feature(x.mean((2, 3)))))


class NativeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = TinyViT().eval()
        self.path = Path(self.temp.name) / "native.pth"
        torch.save({"model." + k: v for k, v in self.source.state_dict().items()}, self.path)
        self.detector = vit_native.NativeViTDetector(0, backbone=TinyViT())
        self.pixels = torch.randn(3, 3, 224, 224)

    def test_original_head_parity_and_frozen_eval(self):
        with self.assertRaisesRegex(RuntimeError, "not verified"):
            self.detector(self.pixels)
        result = self.detector.load_checkpoint(self.path)
        self.assertTrue(result["head_loaded"])
        self.detector.train()
        self.assertFalse(self.detector.training)
        self.assertFalse(self.detector.vit.drop.training)
        self.assertFalse(any(p.requires_grad for p in self.detector.parameters()))
        torch.testing.assert_close(self.detector(self.pixels), self.source(self.pixels), rtol=0, atol=0)
        torch.testing.assert_close(self.detector(self.pixels)[:1], self.detector(self.pixels[:1]))
        self.assertTrue(torch.equal(self.detector.vit.head.weight, self.source.head.weight))
        # Verified buffers survive an ordinary model state roundtrip.
        restored = vit_native.NativeViTDetector(0, backbone=TinyViT())
        restored.load_state_dict(self.detector.state_dict(), strict=True)
        torch.testing.assert_close(restored(self.pixels), self.source(self.pixels), rtol=0, atol=0)

    def test_missing_head_or_fusion_or_wrong_hash_refused(self):
        for name, state in (
            ("missing", {"model." + k: v for k, v in self.source.state_dict().items() if not k.startswith("head.")}),
            ("fusion", {**{"vit." + k: v for k, v in self.source.state_dict().items()}, "output.weight": torch.ones(2, 16)}),
            ("wrongshape", {**{"model." + k: v for k, v in self.source.state_dict().items()}, "model.head.weight": torch.ones(3, 8)}),
        ):
            path = Path(self.temp.name) / (name + ".pth")
            torch.save(state, path)
            with self.assertRaises(ValueError):
                self.detector.load_checkpoint(path)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.detector.load_checkpoint(self.path, "0" * 64)

    def test_native_preprocessing_and_explicit_class_order(self):
        processor = vit_native.NativeViTImageProcessor()
        image = Image.fromarray(np.full((37, 51, 3), 255, np.uint8))
        x = processor.preprocess(image)["pixel_values"]
        self.assertEqual(tuple(x.shape), (1, 3, 224, 224))
        torch.testing.assert_close(x, torch.ones_like(x))
        with self.assertRaises(TypeError):
            processor.preprocess(x)
        logits = torch.tensor([[8.0, -2.0]])
        other_order = vit_native.NativeViTDetector(1, backbone=TinyViT())
        self.assertTrue(torch.equal(other_order.token_logits(logits), torch.tensor([[-2.0, 8.0]])))

    def test_clip_input_is_rejected(self):
        self.detector.load_checkpoint(self.path)
        with self.assertRaisesRegex(ValueError, "CLIP"):
            self.detector(self.pixels, clip_vision_features=torch.ones(3, 4, 8))

    def test_actual_llava_token_path_and_train_mode(self):
        from llava.model.language_model.llava_llama import LlavaConfig, LlavaLlamaForCausalLMDeepfake
        cfg = LlavaConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                          num_attention_heads=2, num_key_value_heads=2, pretraining_tp=1,
                          mm_vision_select_feature="cls_patch", deepfake_model_name="vit_native",
                          native_vit_fake_logit_index=0)
        with patch.object(vit_native, "build_native_backbone", side_effect=TinyViT):
            model = LlavaLlamaForCausalLMDeepfake(cfg)
        self.assertIsNone(model.get_vision_tower())
        model.load_deepfake_encoder(self.path, verbose=False)
        model.train()
        self.assertFalse(model.deepfake_encoder.training)
        model._ensure_processors()
        self.assertIs(model.processors["deepfake_processor"], model.deepfake_encoder.image_processor)
        tokens = model.encode_deepfake_tokens({"images": self.pixels})
        self.assertEqual(tuple(tokens.shape), (3, 1, 16))
        expected = self.source(self.pixels)
        torch.testing.assert_close(model._deepfake_logits_for_loss, expected, rtol=0, atol=0)
        reference_tokens = model.deepfake_projector(expected.softmax(-1).unsqueeze(1))
        torch.testing.assert_close(tokens, reference_tokens, rtol=0, atol=0)
        tokens.square().sum().backward()
        self.assertIsNotNone(model.deepfake_projector[0].weight.grad)
        self.assertIsNone(model.deepfake_encoder.vit.head.weight.grad)
        self.assertFalse(any(name.startswith("legacy_integration.") for name in sys.modules))

    def test_native_token_through_actual_language_forward_without_clip(self):
        from llava.model.language_model.llava_llama import LlavaConfig, LlavaLlamaForCausalLMDeepfake
        from llava.constants import DEEPFAKE_TOKEN_INDEX
        cfg = LlavaConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                          num_attention_heads=2, num_key_value_heads=2, pretraining_tp=1,
                          mm_vision_select_feature="cls_patch", deepfake_model_name="vit_native",
                          native_vit_fake_logit_index=0, pad_token_id=0)
        with patch.object(vit_native, "build_native_backbone", side_effect=TinyViT):
            model = LlavaLlamaForCausalLMDeepfake(cfg).eval()
        model.load_deepfake_encoder(self.path, verbose=False)
        image = Image.fromarray(np.full((33, 47, 3), 127, np.uint8))
        ids = torch.tensor([[1, DEEPFAKE_TOKEN_INDEX, 2]])
        with torch.no_grad():
            result = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                           deepfake_inputs=[image], use_cache=False)
        self.assertEqual(tuple(result.logits.shape), (1, 3, 32))
        self.assertTrue(torch.isfinite(result.logits).all())
        native_pixels = model.deepfake_encoder.image_processor.preprocess(image)["pixel_values"]
        torch.testing.assert_close(model._deepfake_logits_for_loss, self.source(native_pixels), rtol=0, atol=0)

    def test_visual_clip_tokens_cannot_change_native_detection(self):
        from llava.model.language_model.llava_llama import LlavaConfig, LlavaLlamaForCausalLMDeepfake
        from llava.constants import IMAGE_TOKEN_INDEX, DEEPFAKE_TOKEN_INDEX
        cfg = LlavaConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                          num_attention_heads=2, num_key_value_heads=2, pretraining_tp=1,
                          mm_vision_select_feature="cls_patch", deepfake_model_name="vit_native",
                          native_vit_fake_logit_index=0)
        with patch.object(vit_native, "build_native_backbone", side_effect=TinyViT):
            model = LlavaLlamaForCausalLMDeepfake(cfg).eval()
        model.load_deepfake_encoder(self.path, verbose=False)
        # A deterministic stand-in for visual tokens; no pretrained CLIP is
        # needed to check the routing of the real multimodal method.
        class VisualTokens(nn.Module):
            def forward(self, images):
                return images.mean((1, 2, 3))[:, None, None].expand(-1, 5, 8)
        model.model.vision_tower = VisualTokens()
        model.model.mm_projector = nn.Sequential(nn.Linear(8, 16))
        ids = torch.tensor([[1, IMAGE_TOKEN_INDEX, DEEPFAKE_TOKEN_INDEX, 2]])
        embeddings = []
        with torch.no_grad():
            for value in (0.0, 3.0):
                result = model.prepare_inputs_labels_for_deepfake_multimodal(
                    ids, None, torch.ones_like(ids), None, None,
                    images=torch.full((1, 3, 336, 336), value), deepfake_inputs=self.pixels[:1])
                self.assertEqual(tuple(result[4].shape), (1, 7, 16))
                embeddings.append(result[4])
                torch.testing.assert_close(model._deepfake_logits_for_loss, self.source(self.pixels[:1]), rtol=0, atol=0)
        self.assertFalse(torch.equal(embeddings[0], embeddings[1]))

    def test_native_hf_checkpoint_roundtrip_keeps_original_head(self):
        from llava.model.language_model.llava_llama import LlavaConfig, LlavaLlamaForCausalLMDeepfake
        cfg = LlavaConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                          num_attention_heads=2, num_key_value_heads=2, pretraining_tp=1,
                          mm_vision_select_feature="cls_patch", deepfake_model_name="vit_native",
                          native_vit_fake_logit_index=0)
        with patch.object(vit_native, "build_native_backbone", side_effect=TinyViT):
            model = LlavaLlamaForCausalLMDeepfake(cfg).eval()
            model.load_deepfake_encoder(self.path, verbose=False)
            output = Path(self.temp.name) / "hf"
            model.save_pretrained(output, safe_serialization=True)
            restored, loading = LlavaLlamaForCausalLMDeepfake.from_pretrained(output, output_loading_info=True)
        self.assertFalse(loading["missing_keys"])
        self.assertFalse(loading["unexpected_keys"])
        self.assertTrue(bool(restored.deepfake_encoder._native_checkpoint_verified))
        torch.testing.assert_close(restored.deepfake_encoder(self.pixels), self.source(self.pixels), rtol=0, atol=0)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
