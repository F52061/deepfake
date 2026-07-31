# M2F2-Det 完整项目工作流记录

> 方案选择：**方案B BridgeAdapter**（`ViT_M2F2Det_Unified(fusion_mode='bridge')`）
> 环境：M2F2_Det `C:\Users\Supor2\.conda\envs\M2F2_Det`
> 硬件：3× GTX 1080 Ti (11.8 GB/卡)

---

## 三阶段总览

```
Stage-1 (检测器训练)         Stage-2 (MLP对齐)           Stage-3 (LoRA微调)
────────────────────         ────────────────           ───────────────────
ViT_M2F2Det_Unified          将检测器接入 LLaVA            LoRA 微调 LLM
fusion_mode='bridge'         训练 deepfake_projector      输出检测+解释
FF++ 107K 二分类数据         DD-VQA judge 数据            DD-VQA 完整数据
✅ 已完成                     ❌ 待做                      ❌ 待做
```

---

## Stage-1 已完成（你的方案B）

| 项目 | 内容 |
|------|------|
| 检测器类 | `ViT_M2F2Det_Unified(fusion_mode='bridge')` |
| 训练脚本 | `vit_module/train_bridge_phase1.py` |
| 权重文件 | `./checkpoints/stage_1/bridge_v2_phase1.pth` (2.0 GB) |
| ViT backbone | `net_050.pth` (PDI 预训练) |
| 训练数据 | `./dataset/data_2023/ffpp_train_split.txt` (107,700 样本) |
| 验证数据 | FFPP_all, CD2, FFIW, dfdcp, Wild |
| 训练策略 | ViT backbone 冻结，BridgeAdapter 从头训练，早停 patience=5 |

---

## 权重下载清单

| # | 权重 | 大小 | 来源 | 状态 | 用途 |
|---|------|------|------|------|------|
| 1 | `bridge_v2_phase1.pth` | 2.0 GB | 自己的训练 | ✅ 已完成 | Stage-2/3 检测器 |
| 2 | `liuhaotian/llava-v1.5-7b` | ~14 GB | [HuggingFace](https://huggingface.co/liuhaotian/llava-v1.5-7b) | ❌ 下载中 | Stage-2/3 基座模型 |
| 3 | `openai/clip-vit-large-patch14-336` | 1.6 GB | HF 缓存 | ✅ 本地已有 | CLIP 视觉编码器 |
| 4 | `M2F2_Det_densenet121.pth` | 1.7 GB | Google Drive | ✅ 已有 | 原始项目（不使用） |
| 5 | `llava-v1.5-7b-M2F2-Det` (HF成品) | 13.6 GB | HuggingFace | ✅ 已有 | 参考/对比（不使用） |

---

## Stage-2：MLP 多模态对齐

### 目的

让 bridge_v2 检测器的二分类输出进入 LLaMA 的 token 嵌入空间。

### 数据流

```
bridge_v2 检测器 → [B, 2] logits
    │
    ▼ softmax → [B, 1, 2]
    │
    ▼ deepfake_projector (MLP mlp2x_gelu) ← ★ 唯一训练
      Linear(2 → 256) → GELU → Linear(256 → 4096)
    │
    ▼ [B, 1, 4096] → 作为额外 token 插入 LLaMA 输入序列
    │
    ▼ LLaMA (冻结) → next-token-prediction loss
```

### 训练策略

| 组件 | 状态 |
|------|------|
| LLaMA-7B (`model.args`) | ❌ 冻结 |
| CLIP→LLaMA 投影 (`mm_projector`) | ❌ 冻结 |
| deepfake_projector (`deepfake_projector`) | ✅ **训练** |
| bridge_v2 检测器 (`deepfake_encoder`) | ❌ 冻结（推理模式） |

### 数据

```
./utils/DDVQA_split/c40/train_DDVQA_format_judge_only.json  (1339 样本)
格式:
{
  "id": "057_070",
  "image": "3_057_070.jpg",
  "conversations": [
    {"from": "human", "value": "Determine the authenticity of this image"},
    {"from": "gpt", "value": "This is a computer-generated image..."}
  ]
}
./utils/DDVQA_images/c40/train/  (1339 图片)
```

### 显存估算

| 配置 | 显存 |
|------|------|
| fp16 + gradient_checkpointing + batch_size=8 | ~16 GB （单卡不够） |
| fp16 + gradient_checkpointing + batch_size=4 + accum=2 | ~12 GB （接近边界） |
| **推荐: deepspeed ZeRO-3 + 2卡** | **~8 GB/卡 ✅** |

### 完整命令

```bash
# ══════════════════════════════════════════════════════
# 需要先改的代码（见下方 "集成修改清单"）
# ══════════════════════════════════════════════════════

# ── Step 2-1: 创建随机初始化的 deepfake_projector 骨架 ─────────
python scripts/merge_lora_weights_deepfake_random.py \
    --model-path ./checkpoints/llava-v1.5-7b \
    --save-model-path ./checkpoints/llava-1.5-7b-deepfake-rand-proj-v1 \
    --deepfake-ckpt-path ./checkpoints/stage_1/bridge_v2_phase1.pth

# ── Step 2-2: 训练 deepfake_projector (MLP 对齐) ────────────────
deepspeed --include localhost:0,1 llava/train/train_deepfake.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ./checkpoints/llava-1.5-7b-deepfake-rand-proj-v1 \
    --version v1 \
    --data_path ./utils/DDVQA_split/c40/train_DDVQA_format_judge_only.json \
    --image_folder ./utils/DDVQA_images/c40/train \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --deepfake_ckpt_path ./checkpoints/stage_1/bridge_v2_phase1.pth \
    --freeze_backbone True \
    --tune_deepfake_mlp_adapter True \
    --freeze_mm_mlp_adapter True \
    --tune_mm_mlp_adapter False \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_vision_select_feature cls_patch \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.5-7b-deepfake_stage-2-proj \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 9 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True
```

### 输出产物

```
./checkpoints/llava-v1.5-7b-deepfake_stage-2-proj/
├── config.json             # ← deepfake_model_name="vit", deepfake_model_path=bridge_v2.pth
├── non_lora_trainables.bin # ← 训练好的 deepfake_projector 权重
└── tokenizer相关
```

---

## Stage-3：LoRA 微调（检测+解释）

### 目的

让 LLaMA 学会把检测信号翻译成自然语言解释。

### 与 Stage-2 关键区别

| 对比项 | Stage-2 | Stage-3 |
|--------|---------|---------|
| 训练参数 | 仅 deepfake_projector (MLP) | LoRA + mm_projector + deepfake_projector |
| LLaMA 状态 | 冻结 | LoRA 适配（可选部分） |
| 数据 | judge_only (仅"判断") | 完整 DD-VQA（判断+解释） |
| 训练参数数量 | ~1 M | ~10 M (LoRA) + 1 M (MLP) |

### 数据

```
./utils/DDVQA_split/c40/train_DDVQA_format.json  (1339 样本)
包含完整的 "why" 解释对话:
{
  "conversations": [
    {"from": "human", "value": "Determine the authenticity of this image"},
    {"from": "gpt", "value": "The image is fake because..."}  ← 解释
  ]
}
```

### 完整命令

```bash
# ── Step 3-1: 合并 Stage-2 delta 到基座 ──────────────────────────
python scripts/merge_lora_weights_deepfake.py \
    --model-base ./checkpoints/llava-v1.5-7b \
    --model-path ./checkpoints/llava-v1.5-7b-deepfake_stage-2-proj \
    --save-model-path ./checkpoints/llava-v1.5-7b-deepfake-stage-2

# ── Step 3-2: LoRA 微调 ──────────────────────────────────────────
deepspeed --include localhost:0,1 llava/train/train_deepfake.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ./checkpoints/llava-v1.5-7b-deepfake-stage-2 \
    --version v1 \
    --data_path ./utils/DDVQA_split/c40/train_DDVQA_format.json \
    --image_folder ./utils/DDVQA_images/c40/train \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --deepfake_ckpt_path ./checkpoints/stage_1/bridge_v2_phase1.pth \
    --lora_enable True \
    --lora_r 128 \
    --lora_alpha 256 \
    --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero2.json \
    --tune_mm_mlp_adapter True \
    --tune_deepfake_mlp_adapter True \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_vision_select_feature cls_patch \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.5-7b-deepfake_stage-3-delta \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1500 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True

# ── Step 3-3: 最终合并 LoRA delta → 成品 ─────────────────────────
python scripts/merge_lora_weights_deepfake.py \
    --model-base ./checkpoints/llava-v1.5-7b-deepfake-stage-2 \
    --model-path ./checkpoints/llava-v1.5-7b-deepfake_stage-3-delta/checkpoint-xxx \
    --save-model-path ./checkpoints/llava-v1.5-7b-M2F2-Det
```

### 输出产物

```
./checkpoints/llava-v1.5-7b-M2F2-Det/
├── config.json
├── model-00001-of-00003.safetensors  # 完整的推理权重
├── model-00002-of-00003.safetensors
├── model-00003-of-00003.safetensors
└── model.safetensors.index.json
```

---

## 集成修改清单

在开始 Stage-2 之前需要修改以下文件：

### 修改 1: `llava/model/language_model/llava_llama.py`

```python
# 第 33 行附近: 将 import 改为统一版本
# 原:
# from vit_module.vit_m2f2_detector import ViT_M2F2Det

# 改为:
from vit_module.vit_m2f2_detector_unified import ViT_M2F2Det_Unified as ViT_M2F2Det
```

### 修改 2: `build_deepfake_encoder()`（同一文件 第 165-186 行）

```python
def build_deepfake_encoder(
    deepfake_encoder_name: str = 'vit',     # ← 改为 'vit'（原来是默认值）
    vision_model_name: str = 'openai/clip-vit-large-patch14-336',
    text_model_name: str = 'openai/clip-vit-large-patch14-336',
    ...
):
    model = ViT_M2F2Det(                     # ← 实际是 ViT_M2F2Det_Unified
        fusion_mode='bridge',               # ★ 关键: 指定 bridge 模式
        ...
    )
    return model
```

### 修改 3: `merge_lora_weights_deepfake_random.py`（第 8 行）

`init_deepfake_branch()` 中的 `deepfake_projector` 初始化保持不变即可（MLP 初始化）。

### 修改 4: 第 1 次运行前的 config.json

随机初始化骨架的 config.json 需要包含:
```json
{
  "_name_or_path": "llava-1.5-7b-vit-deepfake",
  "deepfake_model_name": "vit",
  "deepfake_model_path": "./checkpoints/stage_1/bridge_v2_phase1.pth",
  "mm_vision_select_feature": "cls_patch",
  "mm_vision_tower": "openai/clip-vit-large-patch14-336",
}
```

（`merge_lora_weights_deepfake_random.py` 会在加载 LLaVA config 后自动更新这些字段）

### 修改 5: 所有脚本中的路径

| 脚本 | 旧路径 | 新路径 |
|------|--------|--------|
| `stage_2_train.sh` | `/user/guoxia11/.../llava-v1.5-7b` | `./checkpoints/llava-v1.5-7b` |
| `stage_3_train.sh` | `/user/guoxia11/.../llava-v1.5-7b` | `./checkpoints/llava-v1.5-7b` |
| `finetune_stage_2.sh` | `DEEPFAKE_CKPT_PATH=densenet121.pth` | `DEEPFAKE_CKPT_PATH=bridge_v2_phase1.pth` |
| `finetune_stage_3.sh` | `DEEPFAKE_CKPT_PATH=densenet121.pth` | `DEEPFAKE_CKPT_PATH=bridge_v2_phase1.pth` |

---

## 推理验证

训练完成后，用最终权重做检测+解释：

```bash
# 检测推理 (输出 JSONL 包含判断结果)
python -m llava.serve.cli_DDVQA_det \
    --model-path ./checkpoints/llava-v1.5-7b-M2F2-Det

# 解释推理 (输出 JSONL 包含解释文本)
python -m llava.serve.cli_DDVQA_exp \
    --model-path ./checkpoints/llava-v1.5-7b-M2F2-Det
```

多卡推理（代码中 device_map 分发逻辑已存在）：
```python
num_gpus = torch.cuda.device_count()
if num_gpus > 1:
    from accelerate import dispatch_model
    layers_per_gpu = (32 + num_gpus - 1) // num_gpus
    # ... 自动分发到多卡
```

---

## 显存需求汇总

| 阶段 | 配置 | 显存需求 | 你的硬件 |
|------|------|---------|---------|
| Stage-2 训练 | fp16, grad_ckpt, batch=8, 2卡 | ~8 GB/卡 | ✅ ZeRO-3 |
| Stage-3 训练 | fp16, LoRA, batch=4, 2卡 | ~9 GB/卡 | ✅ ZeRO-3 |
| 推理 (检测+解释) | fp16, 2卡分发 | ~7 GB/卡 | ✅ |
| 推理 (检测+解释) | fp16, 单卡 | ~16 GB | ❌ 不够 |

---

## 检查清单

- [ ] **Stage-1 完成** ✅
- [ ] LLaVA-1.5-7b 基座下载完成
- [x] 修改 `llava_llama.py` import (ViT_M2F2Det_Unified) ✅
- [x] 修改 `build_deepfake_encoder()` (fusion_mode='bridge', hidden_size=768) ✅
- [x] 修改 `merge_lora_weights_deepfake_random.py` (添加 --deepfake-ckpt-path) ✅
- [x] 修改 `__init__` 中 load_vision_encoder=True ✅
- [x] 修改 `prepare_inputs_labels_for_deepfake_multimodal` CLIP 传递逻辑 ✅
- [x] 修改 `load_vit_backbone()` 支持训练权重格式 ✅
- [ ] 修改 `finetune_stage_2.sh` 路径
- [ ] 修改 `finetune_stage_3.sh` 路径
- [ ] 运行 `merge_lora_weights_deepfake_random.py` (创建骨架)
- [ ] 运行 Stage-2 训练 (MLP 对齐)
- [ ] 运行 Merge Stage-2
- [ ] 运行 Stage-3 训练 (LoRA)
- [ ] 运行最终 Merge
- [ ] 运行推理测试
