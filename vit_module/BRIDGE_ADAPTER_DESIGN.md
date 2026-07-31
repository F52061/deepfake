# ViT_M2F2Det BridgeAdapter 改造设计

## 目标

将当前 ViT_M2F2Det 的简单 cosine-similarity 融合方案，升级为与原始 EfficientNet M2F2Det 一致的 BridgeAdapter 跨模态/跨尺度递进融合方案。

## 原始 M2F2Det (EfficientNet-B4) 工作流回顾

### 三步特征提取

```
输入 [B, 3, 336, 336] → CLIP 预处理
    │
    ├── CLIP Text Encoder          → clip_text_features [B, 1024]      (冻结，未优化)
    ├── CLIP Vision Encoder (冻结)  → clip_0 [B, 576, 1024]  layer 6
    │                                  clip_1 [B, 576, 1024]  layer 10
    │                                  clip_2 [B, 576, 1024]  layer 14
    │                                  clip_vision_cls [B, 1024]  × α_v
    │
    └── EfficientNet-B4 (微调)
          ├── features[4] hook → b_1  [B, 112, H0, W0]
          ├── features[5] hook → b_2  [B, 160, H1, W1]
          ├── features[6] hook → b_3  [B, 272, H2, W2]
          └── 最终输出 → AvgPool + deepfake_proj → [B, 1792]
```

### BridgeAdapter 三阶段递进融合

每一阶段 i：融合 deepfake_feat_i + clip_i → TransformerBlock

```
Stage 0:
  deepfake_feat_0 → linear_1(112→64) → [B, H0W0, 64]
  clip_0 → clip_reduction(1024→64)   → [B, 576, 64]
  concat → [B, H0W0+576, 64] → permute(1,0,2) → TransformerBlock(0)

Stage 1:
  prev_out + deepfake_feat_1 → linear_2(160→64) + clip_1 → ...
  concat → TransformerBlock(1)

Stage 2:
  prev_out + deepfake_feat_2 → linear_3(272→64) + clip_2 → ...
  concat → TransformerBlock(2)

最终: BridgeAdapter_Proj → [B, 128] × α_t
```

### 最终分类

```python
features = torch.cat([
    clip_vision_cls,      # [B, 1024]
    clip_adapt_embed,     # [B, 128]
    deepfake_features,    # [B, 1792]  ← backbone 全局池化
], dim=-1)  # → [B, 2944]
output = Linear(2944, 2)  → [B, 2]
```

## ViT_M2F2Det 对齐改造方案

### 模型结构变化

```
ViT_M2F2Det (改造后)
├── ViT backbone (vit_adaptive_mattn_aps)
│     ├── blocks[3] → hook "b_1"  [B, 197, 768] → 去CLS → [B, 196, 768]
│     ├── blocks[6] → hook "b_2"  [B, 197, 768] → 去CLS → [B, 196, 768]
│     ├── blocks[9] → hook "b_3"  [B, 197, 768] → 去CLS → [B, 196, 768]
│     └── forward_features 输出    [B, 197, 768] → CLS token [B, 768]
│          ❌ 去掉原来的分类头 (vit.head = Identity)
│          ✅ 保留 CLS token 作为高层语义特征
│
├── CLIP Text Encoder           (不变)
├── CLIP Vision Encoder (冻结)   (不变)
│
├── BridgeAdapter (新增)
│     ├── clip_reduction   Linear(1024→64)
│     ├── linear_vit_1     Linear(768→64)     ← 替换 linear_1(112→64)
│     ├── linear_vit_2     Linear(768→64)     ← 替换 linear_2(160→64)
│     ├── linear_vit_3     Linear(768→64)     ← 替换 linear_3(272→64)
│     ├── bridge_adapter   [TransformerEncoderBlock(64,4,32) × 3]
│     └── bridge_adapter_proj  BridgeAdapter_Proj(64,128)
│
├── deepfake_proj   Linear(768 → hidden_size)  ← ViT CLS → 融合空间
├── output          Linear(1024 + 128 + hidden_size, 2)  ← 分类头
├── clip_vision_alpha  α_v
└── clip_text_alpha    α_t
```

### 改造后 forward 流程

```python
def forward(self, images, clip_vision_features=None, ...):
    B = images.shape[0]

    # 1. ViT backbone forward + 中间特征提取
    vit_input = self._preprocess_for_vit(images)  # 336→224, 反归一化
    vit_input = vit_input.to(self.vit_dtype)
    
    # ViT forward 带 hook → 自动填充 self.vit_block_outputs["b_1/b_2/b_3"]
    vit_out = self.vit.forward_features(vit_input)
    vit_cls_token = vit_out[:, 0]                 # [B, 768]
    vit_features = self.deepfake_proj(vit_cls_token)  # [B, hidden_size]

    # 2. CLIP vision features
    if clip_vision_features is None:
        clip_0, clip_1, clip_2, clip_vision_features = self.clip_vision_encoder(images)
    clip_vision_features = self.vision_proj(clip_vision_features)
    clip_vision_cls = clip_vision_features[:, 0, :]  # [B, hidden_size]

    # 3. CLIP text features
    clip_text_features = self.clip_text_encoder()
    clip_text_features = self.text_proj(clip_text_features)

    # 4. BridgeAdapter 逐层融合
    vit_feat_lst = [
        self.vit_block_outputs["b_1"][:, 1:, :],  # 去CLS [B, 196, 768]
        self.vit_block_outputs["b_2"][:, 1:, :],  # [B, 196, 768]
        self.vit_block_outputs["b_3"][:, 1:, :],  # [B, 196, 768]
    ]
    clip_feat_lst = [clip_0, clip_1, clip_2]       # [B, 576, 1024]

    bridge_adapter_output = None
    for i, (vit_feat, clip_feat) in enumerate(zip(vit_feat_lst, clip_feat_lst)):
        vit_feat = vit_feat.to(device)
        clip_feat = clip_feat.to(device)
        clip_feat = self.clip_reduction(clip_feat)           # [B, 576, 64]
        vit_feat = self.linear_vit_lst[i](vit_feat)           # [B, 196, 64]

        if bridge_adapter_output is None:
            combined = torch.cat((vit_feat, clip_feat), dim=1)
        else:
            bridge_adapter_output = bridge_adapter_output.permute(1, 0, 2)
            combined = torch.cat((bridge_adapter_output, vit_feat, clip_feat), dim=1)

        combined = combined.permute(1, 0, 2)                  # [seq, B, 64]
        bridge_adapter_output = self.bridge_adapter[i](combined)

    # 5. BridgeAdapter 池化
    clip_adapt_embed = self.bridge_adapter_proj(bridge_adapter_output, B)
    clip_adapt_embed = self.clip_text_alpha * clip_adapt_embed

    # 6. CLIP vision CLS 加权
    clip_vision_cls = self.clip_vision_alpha * clip_vision_cls

    # 7. 三特征融合 + 分类
    features = torch.cat([
        clip_vision_cls,        # [B, hidden_size]
        clip_adapt_embed,       # [B, 128]
        vit_features,           # [B, hidden_size]
    ], dim=-1)

    output = self.output(features)
    return output
```

### 需要新增/修改的组件列表

| 修改项 | 文件 | 说明 |
|--------|------|------|
| ✅ 新增 `linear_vit_1/2/3` | `vit_m2f2_detector.py` `__init__` | ViT patch→64 维（768→64） |
| ✅ 新增 `clip_reduction` | 同上 | CLIP patch→64 维（1024→64） |
| ✅ 新增 `bridge_adapter` | 同上 | 3×TransformerEncoderBlock |
| ✅ 新增 `bridge_adapter_proj` | 同上 | BridgeAdapter_Proj |
| ✅ 新增 hook 注册 | 同上 `__init__` | blocks[3][6][9] 注册 hook |
| ✅ 修改 `forward()` | 同上 | 替换 cosine_sim 为 BridgeAdapter 流程 |
| ✅ 修改 `deepfake_proj` | 同上 | 768→hidden_size (从 ViT CLS) |
| ✅ 修改 `output` | 同上 | Linear(2*hidden_size+128, 2) |
| ✅ 修改 `load_vit_backbone` | 同上 | 不改，新参数自动均随机初始化 |
| ❌ 删除 `_preprocess_for_vit` 内部的反归一化 | 可选 | CLIP输出已经归一化 |

### 维度对齐验证

```
Stage 0:
  vit_feat_0 [B, 196, 768]  → linear_vit_1(768→64)  → [B, 196, 64]
  clip_0     [B, 576, 1024] → clip_reduction(1024→64) → [B, 576, 64]
  concat → [B, 772, 64] → TransformerBlock(0) → [772, B, 64]

Stage 1:
  + vit_feat_1 [B, 196, 768] → linear_vit_2(768→64) → [B, 196, 64]
  + clip_1     [B, 576, 1024] → clip_reduction      → [B, 576, 64]
  concat → [772+196+576=1544, B, 64] → TransformerBlock(1)

Stage 2:
  + vit_feat_2 [B, 196, 768] → linear_vit_3(768→64) → [B, 196, 64]
  + clip_2     [B, 576, 1024] → clip_reduction      → [B, 576, 64]
  concat → [1544+196+576=2316, B, 64] → TransformerBlock(2)

BridgeAdapter_Proj:
  View(-1, 64) → [2316*B, 64] → Linear(64→1) → LN → View(B, 2316)
  Linear(2316, 128) → LN → [B, 128]
```

### 训练配置

| 组件 | 初始化 | 优化器 |
|------|--------|--------|
| ViT backbone | 从 `net_050.pth` 加载 | ✅ 微调 |
| CLIP Vision Encoder | vision_tower.pth | ❌ 冻结 |
| CLIP Text Encoder | HuggingFace | ❌ 冻结 |
| deepfake_proj | 随机正态 σ=0.01 | ✅ 训练 |
| vision_proj | 随机正态 σ=0.01 | ✅ 训练 |
| text_proj | 随机正态 σ=0.01 | ✅ 训练 |
| clip_reduction | 随机正态 σ=0.01 | ✅ 训练 |
| linear_vit_1/2/3 | PyTorch 默认 | ✅ 训练 |
| bridge_adapter ×3 | PyTorch 默认 | ✅ 训练 |
| bridge_adapter_proj | 随机正态 σ=0.01 | ✅ 训练 |
| output | 随机正态 σ=0.01 | ✅ 训练 |
| clip_vision_alpha | 0.5 | ✅ lr=1e-3 |
| clip_text_alpha | 4.0 | ✅ lr=3e-3 |
