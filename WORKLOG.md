---
name: M2F2-Det-worklog
description: M2F2-Det 项目完整工作日志。Stage-1 检测器训练已完成（方案B BridgeAdapter），Stage-3 LoRA 微调可启动。含环境配置、权重清单、代码修改记录、踩坑结论、恢复指引。供多智能体协作断点续接。
metadata:
  type: project
  updated: 2026-08-02
  status: stage3-ready-to-launch
  agent_note: |
    新会话读取此文件后可立即了解项目全貌。
    微调启动命令: vit_module\run_stage3.bat
    检测器权重: checkpoints/stage_1/bridge_v2_phase1.pth
    环境: M2F2_Det conda env, 4x GTX 1080 Ti
---

# M2F2-Det 项目工作日志 — 多智能体协作检查点

> **项目路径**: `E:\Cross-domain_authentication_verification\Next_work\M2F2_Det-main-hyy`
> **本文档用途**: 项目检查点、上下文恢复、新会话/新智能体接手续跑
> **最后更新**: 2026-08-02

---

## 1. 项目目标与当前阶段

### 1.1 项目简介
复现 **M2F2-Det**（CVPR 2025 Oral，多模态伪造人脸检测器），用自己的 **ViT + BridgeAdapter 检测器**（方案B）替换原始 EfficientNet-B4/DenseNet121 检测器。

### 1.2 三阶段流程与完成状态

```
┌─────────────────────────────────────────────────────┐
│ Stage-1: 检测器预训练（二分类）                     │
│ 数据: FF++ (107K 图片, dataset/data_2023/*.txt)     │
│ 模型: ViT_M2F2Det_Unified(fusion_mode='bridge')     │
│ 产物: checkpoints/stage_1/bridge_v2_phase1.pth      │
│ 状态: ✅ 完成 (2026-07-30)                           │
├─────────────────────────────────────────────────────┤
│ Stage-2: MLP 多模态对齐                             │
│ 目的: 训练 deepfake_projector (MLP) 映射检测器输出   │
│ 状态: ↩️ 跳过 — Stage-3 的 tune_deepfake_mlp_adapter │
│       会同步训练 MLP，故直接从骨架进入 Stage-3       │
├─────────────────────────────────────────────────────┤
│ Stage-3: LoRA 微调 LLaMA (检测+解释)                 │
│ 数据: DD-VQA (27,756 条对话, utils/DDVQA_split/c40) │
│ 骨架: checkpoints/llava-1.5-7b-deepfake-rand-proj   │
│ 检测器: checkpoints/stage_1/bridge_v2_phase1.pth     │
│ 状态: ⏳ 待启动（配置完整，已验证无 nan、可保存）      │
└─────────────────────────────────────────────────────┘
```

---

## 2. 环境与硬件约束

### 2.1 软件环境

| 项目 | 值 |
|------|-----|
| 操作系统 | Windows 10 Enterprise 10.0.19045 |
| conda 环境 | `M2F2_Det` (`C:\Users\Supor2\.conda\envs\M2F2_Det`) |
| Python | 3.10.14 |
| PyTorch | **2.2.2+cu118** (CUDA) — 曾误装 2.13.0+cpu，已重装 |
| torchvision | 0.17.2+cu118 — 已 patch 移除 `_meta_registrations` import |
| transformers | 4.37.0 |
| flash_attn | 已安装 ✅ |
| deepspeed | **未安装** ❌ — Windows 编译困难 |
| bitsandbytes | 0.43.0 — 前向可用，**反向传播 dtype 不兼容，不可训练** |
| peft | 0.10.0 |

### 2.2 硬件环境

| 项目 | 值 |
|------|-----|
| GPU | 4× NVIDIA GeForce GTX 1080 Ti (11.8 GB/卡) |
| 架构 | Pascal — **不支持 bf16** |
| 总显存 | 47.2 GB |
| 可用组合 | GPU 0,1,2,3（推荐用 0,2,3 三卡训练，GPU 1 留空或备用） |

### 2.3 关键约束（必须遵守，否则报错）

1. **Windows PyTorch 不读取 Python 内 `os.environ["CUDA_VISIBLE_DEVICES"]`**
   → 必须用 `.bat` 在进程启动时设系统环境变量 `set CUDA_VISIBLE_DEVICES=0,2,3`
2. **fp16 + GradScaler 与冻结 backbone 冲突**
   → 可训练参数 (LoRA/projector) 必须 fp32，冻结主模型保持 fp16；用真实 GradScaler
3. **trainer 日志显示 loss=0.0 是显示 bug**
   → 多卡 dispatch 下 `tr_loss` 聚合异常；用 `[DBG-LOSS]` 打印判断真实训练进度
4. **1080 Ti 不支持 bf16** → 不能用 `--bf16`，只能用 `--fp16` 或 fp32
5. **deepspeed 未安装** → `maybe_zero_3` 函数需 try/except ImportError fallback

---

## 3. 权重文件清单

### 3.1 核心权重（按用途分类）

| 用途 | 路径 | 大小 | 状态 | 说明 |
|------|------|------|------|------|
| **ViT backbone (PDI)** | `E:\Cross-domain_authentication_verification\PDI\results\Ama1_aps1_1\net_050.pth` | 365 MB | ✅ | Stage-1 检测器初始化 |
| **方案B 检测器 (最终)** | `checkpoints/stage_1/bridge_v2_phase1.pth` | 2.0 GB | ✅ | **Stage-3 使用的检测器** |
| 方案A 检测器 (备用) | `checkpoints/stage_1/cosine_realclip_phase1.pth` | 2.0 GB | ✅ | cosine 融合，备用对比 |
| CLIP ViT-L/14 | `checkpoints/clip-vit-large-patch14-336/` | 1.6 GB | ✅ | 离线 CLIP，所有 `from_pretrained` 重定向至此 |
| LLaVA-1.5-7b 基座 | `checkpoints/llava-1.5-7b/` | ~20 GB | ✅ | 下载自 HuggingFace liuhaotian/llava-v1.5-7b |
| Stage-3 骨架 | `checkpoints/llava-1.5-7b-deepfake-rand-proj-v1/` | ~14 GB | ✅ | LLaVA 基座 + bridge_v2 + 随机 deepfake_projector |
| M2F2-Det 成品 (HF) | `checkpoints/llava-v1.5-7b-M2F2-Det/` | 13.6 GB | ✅ | 官方成品 (DenseNet版，参考用) |
| Hybrid 合并模型 | `checkpoints/llava-v1.5-7b-bridge-hybrid/` | 14.8 GB | ✅ | HF LoRA + bridge_v2 + checkpoint-1700 projector |

### 3.2 数据集文件

| 数据集 | 路径 | 样本数 | 用途 | 阶段 |
|--------|------|--------|------|------|
| FF++ 训练 | `dataset/data_2023/ffpp_train_split.txt` | 107,700 | 二分类 (图像路径 + 标签) | Stage-1 |
| FF++ 测试 | `dataset/data_2023/ffpp_test_split.txt` | 21,000 | 验证 | Stage-1 |
| 跨域测试 | `dataset/data_2023/CD*_test.txt` 等 | 多种 | 泛化测试 | Stage-1 eval |
| **DD-VQA 微调** | `utils/DDVQA_split/c40/train_DDVQA_format.json` | **27,756** | 对话 (判断+解释) | **Stage-3** |
| DD-VQA 图片 | `utils/DDVQA_images/c40/train/` | 27,756 | JPEG 图片 | Stage-3 |
| DD-VQA judge | `utils/DDVQA_split/c40/train_DDVQA_format_judge_only.json` | — | 仅判断 | Stage-2 (跳过) |

### 3.3 备份权重（已归档，不主动使用）

| 权重 | 路径 |
|------|------|
| 早期 bridge 权��� | `checkpoints/stage_1/backup/bridge_phase1_test.pth` |
| 拼写错误 cosine | `checkpoints/stage_1/backup/cosin_realip_phase1.pth` |
| 原始 EffNet Stage-1 | `checkpoints/stage_1/backup/current_model_180.pth` |
| 旧 PDI net_050 | `vit_module/backup_weights/net_050_backup.pth` |
| 旧 cosine 权重 | `vit_module/backup_weights/vit_m2f2_phase1*.pth` |

---

## 4. Stage-3 微调完整配置

### 4.1 启动方式

**在新 CMD 窗口中执行**（推荐）：
```cmd
cd /d E:\Cross-domain_authentication_verification\Next_work\M2F2_Det-main-hyy
vit_module\run_stage3.bat
```

或直接双击 `vit_module/run_stage3.bat`。

### 4.2 启动脚本链

```
run_stage3.bat                           ← 设置 CUDA_VISIBLE_DEVICES=0,2,3
  └── run_stage3.py                      ← CLIP 离线重定向 + 无 GradScaler patch + 调用 train_deepfake.py
       └── llava/train/train_deepfake.py ← 模型加载/分发/LoRA 注入/训练
```

### 4.3 关键训练参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `--model_name_or_path` | `./checkpoints/llava-1.5-7b-deepfake-rand-proj-v1` | 骨架 |
| `--data_path` | `./utils/DDVQA_split/c40/train_DDVQA_format.json` | 微调数据 |
| `--image_folder` | `./utils/DDVQA_images/c40/train` | 图片路径 |
| `--deepfake_ckpt_path` | `./checkpoints/stage_1/bridge_v2_phase1.pth` | 方案B检测器 |
| `--lora_enable` / `--lora_r` / `--lora_alpha` | True / 128 / 256 | LoRA |
| `--tune_mm_mlp_adapter` / `--tune_deepfake_mlp_adapter` | True / True | 同时训练投影层 |
| `--fp16` | True | fp16 计算 |
| `--output_dir` | `./checkpoints/llava-v1.5-7b-deepfake_stage-3-delta` | 输出 |
| `--num_train_epochs` | 1 | 1 epoch |
| `--per_device_train_batch_size` | 1 | 微 batch（显存限制） |
| `--gradient_accumulation_steps` | 16 | 有效 batch=16 |
| `--save_steps` / `--save_total_limit` | 100 / 2 | 每 100 步 checkpoint |
| `--gradient_checkpointing` | True | 省显存 |
| `--dataloader_num_workers` | 0 | Windows spawn 兼容 |
| `--learning_rate` | 2e-5 | |
| `--lr_scheduler_type` | cosine | |

### 4.4 GPU 使用

| 参数 | 值 |
|------|-----|
| CUDA_VISIBLE_DEVICES | **0,2,3** (物理 GPU 0,2,3; GPU 1 跳过) |
| torch 逻辑索引 | 0,1,2 → 物理 0,2,3 |
| 分发方式 | `dispatch_model` / `accelerate` (自动) |
| 每卡显存 | ~4.7 GB (fp16 7B ≈ 14GB / 3 卡) |

### 4.5 微调数据说明

- **数据集**: DD-VQA (Reality Defender Deepfake Visual Question Answering)
- **内容**: 27,756 条多轮对话，每条包含图片 + 人类提问 + GPT 回答
- **对话格式**:
  ```json
  {
    "id": "057_070",
    "image": "3_057_070.jpg",
    "conversations": [
      {"from": "human", "value": "<image>\n<deepfake>\nDescribe the authenticity..."},
      {"from": "gpt",  "value": "There are stains or flaws..."},
      {"from": "human", "value": "Is the image real or fake?"},
      {"from": "gpt",  "value": "Based on the learned representation, this image is fake."}
    ]
  }
  ```
- **训练目标**: 损失 = LLM next-token-prediction loss（预测 GPT 的回答文本）

### 4.6 LoRA 微调原理简述

```
正常微调: 新权重 = 旧权重 + 大更新 (7B 全更新, 太贵)
LoRA 微调: 新权重 = 旧权重 + A×B
   ↑ 旧权重冻结不动
   A = [4096×128] (随机初始化, 可训练)
   B = [128×4096] (零初始化, 可训练)
   A×B ≈ 低秩近似增量, 仅 ~1M 参数/层

推理时: output = Wx + (A×B)x  (自动合并)
```

---

## 5. 训练进度与监控

### 5.1 当前状态

| 指标 | 值 |
|------|-----|
| 状态 | ⏳ **待启动** (配置已就绪) |
| 预计步数 | 1734 步 (27,756 / 16) |
| 预计时长 | ~34 小时 (71s/真实 step) |
| checkpoint | 每 100 步保存到 `checkpoints/llava-v1.5-7b-deepfake_stage-3-delta/checkpoint-X/` |

### 5.2 启动命令（打开新 CMD 窗口执行）

```cmd
cd /d E:\Cross-domain_authentication_verification\Next_work\M2F2_Det-main-hyy
vit_module\run_stage3.bat
```

### 5.3 监控训练进度

```bash
# 使用监控脚本
C:\Users\Supor2\.conda\envs\M2F2_Det\python.exe vit_module/watch_stage3.py

# 或手动 grep
grep DBG-LOSS <输出文件路径> | tail -20
grep -oE "[0-9]+/1734" <输出文件路径> | tail -1
```

### 5.4 预期 loss 趋势
- 起步: ~3.0-4.0 (语言建模损失起始值)
- 训练: 逐渐下降至 0.02-0.5 (模型学会输出正确文本)
- **若出现 nan → 立即停止，检查配置**

---

## 6. 已修改的代码文件（为适配本机）

### 6.1 核心修改（直接决定训练成败）

| 文件 | 修改 | 原因 |
|------|------|------|
| `llava/train/train_deepfake.py:135-146` | `maybe_zero_3` try/except ImportError | 无 deepspeed fallback |
| `llava/train/train_deepfake.py:1065,1074` | `torch_dtype` 显式 fp32 (fp16=False 时) | 统一 dtype |
| `llava/train/train_deepfake.py:1154` | `vt.to(cuda:1)` 仅 2 卡时执行 | 3 卡时不用强制移动 |
| `llava/train/train_deepfake.py:1191-1197` | LoRA 注入后转 fp32 | 防止 GradScaler 报 fp16 梯度 |
| `llava/train/train_deepfake.py:1320-1327` | trainer 创建前可训练参数统一 fp32 | 核心 nan 修复 |
| `llava/model/language_model/llava_llama.py:627-629` | deepfake_encoder dtype fp16 | 统一主模�� dtype |
| `llava/model/language_model/llava_llama.py:292-294` (及多���) | 8 处 image_tensor 硬编码 fp16→self.dtype | fp32 模式下不再 dtype 冲突 |
| `llava/model/language_model/llava_llama.py:649` | `load_deepfake_encoder` 后统一 fp16 | 同 dtype |
| `llava/train/llava_trainer.py:19-30` | `maybe_zero_3` try/except | checkpoint 保存修复 |
| `llava/train/llava_trainer.py:148` | compute_loss 加 `[DBG-LOSS]` 打印 | 真实 loss 监控 |

### 6.2 辅助修改

| 文件 | 修改 |
|------|------|
| `vit_module/run_stage3.py` | Stage-3 启动脚本 (CLIP 离线 + 完整参数) |
| `vit_module/run_stage3.bat` | 设置 CUDA_VISIBLE_DEVICES=0,2,3 |
| `vit_module/watch_stage3.py` | 训练监控脚本 |
| `vit_module/vit_m2f2_detector_unified.py` | 统一检测器 (cosine/bridge 双模式) |
| `vit_module/vit_m2f2_detector_bridge.py` | 方案B 独立版本 (BridgeAdapter) |
| `torchvision/__init__.py` (site-packages) | 移除 `_meta_registrations` import |

---

## 7. 所有踩坑记录（避免重复犯错）

### 7.1 loss=0.0 之谜
- **现象**: trainer 日志 `{'loss': 0.0}`，但 DBG-LOSS 打印显示 real loss=2.85/3.50...
- **根因**: 多卡 `dispatch_model` 下 `tr_loss` 聚合异常 + `logging_nan_inf_filter` 过滤
- **结论**: 训练正常，日志 bug。用 `[DBG-LOSS]` 判断真实进度

### 7.2 loss=nan (第一个真实 step 后)
- **现象**: 第 16 个 micro-step (第一个真实 step) 后 loss → nan
- **根因**: fp16 可训练参数 + no-op GradScaler → 梯度未缩放 → overflow
- **修复**: 可训练参数 (LoRA+projector) 转 fp32 + 真实 GradScaler
- **原理**: LLaVA 标准做法 — 冻结主模型 fp16 + 可训练参数 fp32

### 7.3 checkpoint 保存崩溃
- **现象**: 第 100 步 `ModuleNotFoundError: No module named 'deepspeed'`
- **根因**: `llava_trainer.py` 和 `train_deepfake.py` 的 `maybe_zero_3` 无条件 import deepspeed
- **修复**: 两处都加了 try/except ImportError fallback

### 7.4 Windows CUDA_VISIBLE_DEVICES 无效
- **现象**: Python 内 `os.environ["CUDA_VISIBLE_DEVICES"]="2"` 不生效
- **根因**: Windows PyTorch 在进程启动时读取系统环境变量，不读 Python 运行时设置
- **修复**: 必须用 `.bat` 的 `set CUDA_VISIBLE_DEVICES=...` 在启动前设置

### 7.5 8-bit 训练不可行
- bitsandbytes 8-bit 前向正常，但**反向传播 dtype 不兼容** (Half vs Float)
- 8-bit 仅适合推理

### 7.6 显存方案演变历史
| 方案 | 结果 |
|------|------|
| fp32 3卡 | ❌ OOM (28GB/3 > 11.8GB) |
| fp32 4卡 | ❌ dtype 不一致 (vision_tower 硬编码 fp16) |
| fp16 2卡 | ✅ 可跑 (5min/步) |
| fp16 3卡 + no-op scaler | ⚠️ 16步后 loss=nan |
| **fp16 3卡 + 真实 GradScaler + 可训练参数 fp32** | ✅ **当前方案** |

### 7.7 FP16 GradScaler 冲突
- **现象**: `ValueError: Attempting to unscale FP16 gradients`
- **原因**: freeze_backbone 后，可训练参数全 fp16 → GradScaler 无法处理
- **修复**: 可训练参数转 fp32 (见 7.2)

### 7.8 torchvision 0.17 + torch 2.2 兼容性
- **现象**: `RuntimeError: operator torchvision::nms does not exist`
- **修复**: 移除 `torchvision/__init__.py` 中 `_meta_registrations` import

### 7.9 PyTorch CPU only
- **现象**: `torch 2.13.0+cpu`, 无 CUDA
- **修复**: 重装 `torch==2.2.2+cu118 torchvision==0.17.2+cu118`

---

## 8. 已完成训练的运行记录

### 8.1 方案B Stage-1 训练

| 项目 | 值 |
|------|-----|
| 训练日期 | 2026-07-28 |
| 配置 | Frozen ViT + 训练 BridgeAdapter + projection |
| 数据集 | FF++ 107K (5 类平衡采样) |
| 验证集 | FFPP_all, CD2, FFIW, dfdcp, Wild |
| 产物 | `checkpoints/stage_1/bridge_v2_phase1.pth` |

### 8.2 Stage-3 第一次完成（2026-08-01）

| 项目 | 值 |
|------|-----|
| 步数 | 1734/1734 (100%) |
| 最终 loss | 0.207 (train_loss) |
| 最终 DBG-LOSS | 0.02~0.10 |
| 训练时长 | 34 小时 25 分 |
| checkpoint | 100~1700 步 (17个, 每个含 151MB adapter) |
| LoRA 权重 | ❌ 未保存 (最终保存时崩溃) |
| 崩溃原因 | `maybe_zero_3` 缺 deepspeed |

### 8.3 第二次重跑（待执行）

| 项目 | 值 |
|------|-----|
| 预计时长 | ~34 小时 |
| 修复内容 | `maybe_zero_3` fallback (train_deepfake.py + llava_trainer.py 两处) |
| 预期结果 | 完整保存 LoRA + adapter 权重 |

---

## 9. 已完成训练的运行记录

### 9.1 git 提交记录

```
df90502 Hybrid merge: HF trained LoRA + bridge_v2 detector + checkpoint-1700 projector
3195cde Fix final LoRA save crash: maybe_zero_3 deepspeed fallback (train_deepfake.py copy)
4936c99 Worklog: checkpoint save crash fix + monitor script updated
3e38bf1 Fix checkpoint save crash: maybe_zero_3 deepspeed fallback (no deepspeed installed)
1c21bfe Add Stage-3 training monitor script (watch_stage3.py)
468b9e6 Worklog: structured checkpoint format for Stage-3 training
1f79af7 Stage-3: nan fix verified - fine-tuning running on GPU 0,2,3
e622072 Fix Stage-3 nan: cast trainable params (LoRA+projectors) to fp32, use real GradScaler
19ef50b Stage-3: 3-GPU training running status (loss 2.4-4.3 normal)
79f8f05 Stage-3: 3-GPU dispatch config (GPU 1,2,3), vision_tower move only for 2-GPU
1faa074 Backup: project code + config (exclude dataset nested repo, large weights)
```

### 9.2 git 配置

- 仓库根目录: `E:\Cross-domain_authentication_verification\Next_work\M2F2_Det-main-hyy`
- `dataset/` 是独立嵌套 git 仓库，已通过 `.gitignore` 排除
- 大权重已 gitignore (`*.pth`, `*.safetensors`, `*.bin`, `checkpoints/`, `utils/weights/` 等)

---

## 10. 待办清单

- [ ] **启动微调** — 执行 `vit_module\run_stage3.bat`
- [ ] 训练中监控 — `python vit_module/watch_stage3.py`
- [ ] 等待 1734 步完成 (~34h)
- [ ] 训练完成后 **合并 LoRA delta**:
  ```bash
  python scripts/merge_lora_weights_deepfake.py \
      --model-base ./checkpoints/llava-v1.5-7b-deepfake-stage-2 \
      --model-path ./checkpoints/llava-v1.5-7b-deepfake_stage-3-delta/checkpoint-xxxx \
      --save-model-path ./checkpoints/llava-v1.5-7b-M2F2-Det
  ```
- [ ] **推理验证**:
  ```bash
  python -m llava.serve.cli_DDVQA_det --model-path <合并后权重>
  python -m llava.serve.cli_DDVQA_exp --model-path <合并后权重>
  ```
- [ ] 清理 `checkpoints/llava-v1.5-7b-deepfake_stage-3-delta/` 中 0 字节日志文件
- [ ] 可选: 修复 trainer 日志显示 bug（低优先级）

---

## 11. 恢复检查点指引

**若新会话/新智能体接手，请按以下步骤恢复上下文**：

1. **读本文件** §2 (环境约束)、§4 (配置)、§5 (进度)
2. **检查训练是否活着**: `nvidia-smi | grep python`
3. **检查 checkpoint 目录**: `ls checkpoints/llava-v1.5-7b-deepfake_stage-3-delta/checkpoint-*`
4. **若训练已完成**:
   - 确认 `train_loss` < 0.3（已收敛）
   - 确认 checkpoint 保存正常（含 `mm_projector.bin` 和 `config.json`）
   - 执行 §10 的 LoRA 合并 + 推理验证
5. **若训练中断**:
   - 从 `checkpoint-N` 目录恢复 adapter 权重
   - 重新启动: `vit_module\run_stage3.bat`

---

## 12. 快速参考

### 文件路径速查

| 用途 | 路径 |
|------|------|
| 项目根目录 | `E:\Cross-domain_authentication_verification\Next_work\M2F2_Det-main-hyy` |
| 检测器权重 (最优) | `checkpoints/stage_1/bridge_v2_phase1.pth` |
| Stage-3 启动 | `vit_module/run_stage3.bat` (双击或 CMD 运行) |
| 监控脚本 | `vit_module/watch_stage3.py` |
| 工作日志 | `WORKLOG.md` (本文件) |
| git 仓库 | 项目根目录 `.git` |

### 常用命令

```bash
# 环境
conda activate M2F2_Det

# 启动微调 (需要新 CMD 窗口，不能用 Git Bash)
vit_module\run_stage3.bat

# 监控训练
C:\Users\Supor2\.conda\envs\M2F2_Det\python.exe vit_module/watch_stage3.py

# 检查 GPU
nvidia-smi

# 杀死训练 (如需要)
taskkill /F /PID <训练进程PID> /T
```
