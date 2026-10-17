import torch

from transformers import CLIPImageProcessor, CLIPVisionModel
from torch import nn


class CLIPVisionEncoder(nn.Module):
    def __init__(
        self,
        pretrained_model_name_or_path: str = "openai/clip-vit-large-patch14-336",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.model = CLIPVisionModel.from_pretrained(pretrained_model_name_or_path)
        self.processor = CLIPImageProcessor.from_pretrained(pretrained_model_name_or_path)
        self.select_layer = -2    # following llava v1.5
        self.hidden_size = self.model.config.hidden_size
        for p in self.model.parameters():
            p.requires_grad = False
        print('Freezing CLIP vision encoder.')
        self.dtype = dtype
        self.model.to(dtype)
        
    def forward(
        self,
        inputs_embeds: torch.Tensor
    ):
        inputs_embeds = inputs_embeds.to(self.dtype)
        outputs = self.model(inputs_embeds, output_hidden_states=True)
        hs = outputs.hidden_states                          # tuple of [B, 577, 1024], len=25 for ViT-L/14
        # BridgeAdapter needs 3 intermediate layers + final features
        # CLIP ViT-L/14-336 has 24 transformer layers → hs[0]=embed, hs[1..24]=layers
        n_layers = len(hs) - 1                              # exclude embedding
        clip_0 = hs[max(1, n_layers // 6)]                  # early  (~layer 4)
        clip_1 = hs[max(1, n_layers // 3)]                  # mid    (~layer 8)
        clip_2 = hs[self.select_layer] if self.select_layer < 0 else hs[n_layers // 2]  # deep (~layer 22)
        clip_vision_features = hs[self.select_layer]        # final  [B, 577, 1024]
        # Strip CLS token from intermediate features (keep patches only: 576 tokens)
        clip_0 = clip_0[:, 1:, :]
        clip_1 = clip_1[:, 1:, :]
        clip_2 = clip_2[:, 1:, :]
        # Clip_vision_features keeps CLS token for classification head
        # Cast to float32 for bridge detector compatibility (its layers are float32)
        return clip_0.float(), clip_1.float(), clip_2.float(), clip_vision_features.float()