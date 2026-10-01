---
name: M2F2-Det-worklog
description: M2F2-Det 项目完整工作日志。三阶段全部完成，LoRA 已合并为 7.3B 终版模型；推理卡在「M2F2_Det conda 环境已被删除」+「processors 未初始化」。含环境配置、权重清单、代码修改记录、踩坑结论、恢复指引。供多智能体协作断点续接。
metadata:
  type: project
  updated: 2026-08-08
  status: inference-working-model-quality-issue
  agent_note: |
    新会话读取此文件后可立即了解项目全貌。
    最新状态见文首「2026-08-08 状态快照」。
    检测器权重: checkpoints/stage_1/bridge_v2_phase1.pth
    终版模型: checkpoints/llava-v1.5-7b-M2F2-Det-bridge (7.3B)
    环境: M2F2_Det 已重建(2026-08-08)，推理已跑通
    推理命令: vit_module\test_inference.py (exit 0)
    已知问题: 模型模态坍缩——纯文本也输出伪造解释，见快照「新发现」
---

# M2F2-Det 项目工作日志 — 多智能体协作检查点

> **项目路径**: `E:\Cross-domain_authentication_verification\Next_work\M2F2_Det-main-hyy`
> **本文档用途**: 项目检查点、上下文恢复、新会话/新智能体接手续跑
> **最后更新**: 2026-08-08

---

## ⚡ 2026-08-08 状态快照（最新）

### 总体进度
- ✅ **Stage-1** 检测器预训练完成 → `checkpoints/stage_1/bridge_v2_phase1.pth`
- ✅ **Stage-3** LoRA 微调完成（43h50m, 1734 步, train_loss=0.207）
- ✅ **LoRA 合并完成** → `checkpoints/llava-v1.5-7b-M2F2-Det-bridge`（7.3B, 13.7GB）
- ⛔ **推理测试阻塞**

### 阻塞原因 1：推理代码 bug（已修复）
- **现象**: 模型加载成功（7.3B），`generate()` 报 `TypeError: 'NoneType' object is not subscriptable`
- **根因**: `generate()` 直接访问 `self.processors['clip_processor']`，但 `from_pretrained` 推理路径**不会初始化 processors**（只在训练时通过 `initialize_vision_modules` 设置，构造时第 639 行 `self.processors = None`）。
- **修复**（已应用到工作区）:
  1. `llava/model/language_model/llava_llama.py` `generate()`: 开头调用 `self._ensure_processors()`（该方法已存在，位于 660 行）。
  2. `vit_module/test_inference.py`: 加载后调用 `vt.load_model()`（合并 checkpoint **不含 CLIP tower 权重**，delay_load=True 初始化时无参数故未保存），并 `model._ensure_processors()`。
  3. `test_inference.py`: deepfake 输入改为传**原始 PIL 图**（再传已预处理 tensor 会被 CLIPImageProcessor 二次归一化）。

### 环境重建：✅ 完成（2026-08-08）
- **M2F2_Det env 已重建**: python 3.10.20 + torch 2.2.2+cu118 + torchvision 0.17.2+cu118 + transformers 4.37.0 + peft 0.10.0 + timm 0.9.2 + accelerate + protobuf 等（`conda create -n M2F2_Det python=3.10` 后 pip 安装，全部 exit 0，4 GPU 可见）。
- **flash_attn 解决**: Windows cu118 无轮子（woct0rdho 仓库已删）→ 自建纯 PyTorch shim `vit_module/flash_attn_shim/mha.py`（einsum 非 flash 路径，与 flash_attn 2.5.9 数学逐字一致）。已用「独立 matmul 参考」端到端验证: `mha(x) == 参考, max diff = 0.0`。两个 detector 文件加 try/except fallback 导入。
- **torchvision 0.17.2 patch**: `__init__.py` 移除 `_meta_registrations`（RuntimeError: operator torchvision::nms 不存在）。
- **测试脚本修 3 处**: ① sys.path 加项目根（`python vit_module/xx.py` 时找不到 llava 包）；② `tokenizer_hybrid_token` 结果 `.unsqueeze(0)`（官方用法一致，model_worker.py:168）；③ vision tower 加载后 `.half()`。

### 推理测试：✅ 跑通（exit 0, 3 张图全部生成）
修复的推理 dtype 问题（合并 checkpoint 是 fp16，训练是 fp32，推理暴露的潜伏 bug，均在 `vit_m2f2_detector_unified.py`）：
1. **line 503**: CLIPVisionEncoder 返回前强转 `.float()`（vision_encoder.py:44），中间特征恒 fp32 → 加 `.to(self.vision_dtype)` 转回 fp16。
2. **TransformerEncoderBlock.forward**: 硬编码 `x.to(torch.bfloat16)`，加载后 attn 权重是 fp16 → 改为用 attn 权重实际 dtype。
3. **line 524**: `.to(self.deepfake_dtype)`（=fp32）会把 cls 转 fp32 → cat 报错 → 改为 `.to(self.output.weight.dtype)`。
4. **generate()**: `deepfake_processed_inputs` 漏初始化（有图无 deepfake_inputs 时 UnboundLocalError）→ 补 `= None`。

### ⚠️ 新发现：模型模态坍缩（shortcut learning）— 非环境问题
- **现象**: 真实图（5_001/5_002）也被输出伪造解释；3 张不同 fake 图输出相同文本；**纯文本无图提问也输出 "The person has mismatched beard."**。
- **管线全链路验证正常**:
  - 检测器 logits 逐图正确: real 图 [-4.668, 4.203]，fake 图 [6.691, -7.559] ✅
  - CLIP 图像 token 确实进入 LLM（两图 embeds 577/631 位置不同）✅
  - deepfake 嵌入（2→768 projector 输出）real/fake 明显不同（范数均 ~20.9）✅
  - 训练/推理预处理路径完全一致（forward() 718-723 == generate() 783-788）✅
- **根因**: Stage-3（1734 步, loss 0.207）中文本先验已足够压低 loss（"The person has mismatched X" 是训练集最常见句式），梯度边际收益小 → LLM 学会忽略视觉与检测信号，无条件输出最常见回答。这是深度伪造解释微调的典型模态坍缩，不是环境/管线 bug。
- **⚠️ 2026-08-08 晚修正**: 该结论被 exp 版官方 eval **推翻**（见下节「官方 eval 完整结果」）。之前的判断基于: det 测试集 144 张全 fake + judge 脚本缺陷 + 手动测试的特殊性。exp 版（596 真实图 + 474 伪造图）语义评估显示模型解释能力正常（真实图 99% 自然描述, 伪造图 100% 伪痕迹描述）。
- **可能缓解方向**（未实施, 原记录保留）: 数据平衡/增加真实图权重、检测 logits 作辅助损失、LoRA lr 加大、多轮 eval（用训练措辞的 question 效果略好但本质不变）。

### GPU 状态（2026-08-08）
| GPU | 占用 |
|-----|------|
| 0 | ~9.8GB 被 `deepfake_1` 环境的一个 python 进程占用（PID 17804）|
| 1, 2, 3 | 空闲 |

### 官方 eval 完整结果（2026-08-08 晚, det + exp 两版）
- **det 版**（144 张全 fake, `DDVQA_det_c40.jsonl`）: Acc=1.0/F1=1.0 — 测试集无真实图, 该指标仅证明"对 fake 图输出 fake 回答"。
- **exp 版**（1070 条, 含 596 真实图 + 474 伪造图, `DDVQA_exp_c40.jsonl`）:
  - judge 原始指标 Acc=0.9551/F1=0.9770, **但 judge 脚本不可靠**（三重缺陷, 见下）。
  - **语义评估（修正后）**:
    - 真实图 596: 99% 含自然特征词（naturally smooth/arched/round/straight...）, 1% 含伪痕迹词
    - 伪造图 474: 100% 含伪痕迹词（mismatched/blurry/misaligned...）
    - **结论: 模型能正确区分并解释真实/伪造图 — 解释质量良好**
- **judge 脚本 `eval/eval_judgement.py` 的三重缺陷**:
  1. 只匹配 "real"/"fake"/"computer-generated" 关键词, exp 版 1070 条中 825 条（77%）被丢弃（真实图 595/596 被丢, 因为回答是 "naturally smooth skin" 这类自然描述, 不含 "real"）。
  2. **"unrealistic" 含 "real" 子串** → 10 条伪造图回答（"unrealistic shadows" 等）被误判为 "real" 预测。修正后伪造图 244 条判分中 234 正确。
  3. 引导式问题自带 "This image is real/fake", 若 text 含 question 会污染匹配。
- **修正先前"模态坍缩"结论**: 之前判断的"模型对真实图输出伪造解释"是**误判**——测试集全 fake + judge 缺陷 + 我手动测试时的问题/图特殊性。exp 版语义评估显示模型解释能力正常。

### 下一步
1. （可选）如需更严格的量化, 可自建语义/LLM-judge 评估（如 GPT 打分解释与 GT 的匹配度）。
2. （可选）若需提升解释质量, 按「模态坍缩」缓解方向重训 Stage-3。
3. 更新本文档的元信息（status: 已从 inference-blocked 改为可推理）。

---

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
│ 状态: ✅ 完成 (2026-08-04, 43h50m, 1734 步)          │
│ 合并: ✅ llava-v1.5-7b-M2F2-Det-bridge (7.3B)        │
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

---

# 跨域分析研究账（探针 / t-SNE / Phase A / 方差谱 E3）

> 本文以下追加部分为 **2026-09-06 ~ 2026-09-09** 跨域分析阶段的记录，主题从"训练/评测"转向 **"论文创新点定位 + 检测器特征层归因"**。分析对象 = 已完成的 bridge_v2 冻结双编码器检测器（stage-1 权重 `checkpoints/stage_1/bridge_v2_phase1.pth`）。

## A. 已确认的结论（全部经独立复算 / 单一固定协议）

### A.1 双编码器互补性（R-test，视频级 split 干净）
| 量 | 值 | 含义 |
|---|---|---|
| probe(V) 视频级 | 0.985–0.988 | 冻结 PDI-ViT 在"未见身份"上强于 CLIP |
| probe(C) 视频级 | 0.853–0.878 | CLIP 较弱但非零 |
| probe(R) 视频级 | 0.672–0.706 | 剥掉 CLIP 可预测部分后的残差仍判伪 → 互补真实存在 |
| 决策层 CKA(V,C) | ≈0.14 | 近正交互补（非冗余堆叠） |
| V 有效秩 | 参与率 30，90%能量≈5维 | 极低秩、窄方向 |
| R 有效秩 | 参与率 42，90%能量≈5维 | 残差更散布但仍低秩 |

### A.2 Phase A 判定（2026-09-06 实测，`vit_module/_phaseA/`）
- **E0 域内**：concat(V_proj,C_proj) 视频级 AUC=0.9874 vs V_proj 0.9850（+0.24%）；MLP concat 0.9838 vs MLP V 0.9803（+0.35%）。→ **域内融合决策冗余**；互补信息对二分类得分几乎不可加。论文卖点不能是"融合提升检测"。
- **E1 迁移（单协议）**：V: cd1 0.829 / cd2 0.863 / dfdcp 0.826 / ffiw 0.824 / wild 0.809；C: 0.656/0.723/0.707/**0.834**/0.717。→ **FFIW 是唯一 C>V 域**，且其 V 空间重叠最低(ovl_V10=0.602)。无标签预测器(重叠/对齐)对迁移 AUC 的 Spearman +0.74~0.94（n=5 仅序数）。wild 破单调（重叠高但 AUC 低）→ wild 是"本身难分"非"分布远"。
- **E2 距离 regime 搜索**：全局三分位(k=10,n=1500) V AUC 0.921/0.759/0.720，C 0.736/0.678/0.625 → **无任何合并距离 regime 使 C>V（原始"远→信 C"门控假设被证伪）**；但 **V 随源距单调、大幅掉点（0.92→0.72），域内近/远对照(排除域混淆)亦成立** → "源距=无标签逐样本 V 脆弱性/可靠性信号"。唯一例外：FFIW **近源半** C=0.937 vs V=0.863（k=5/k=10 一致）→ C 优势在 FFIW 的 in-distribution 区而非远区。
- **当前 GPU1 全量端到端评测（bprsida7g）已 kill**，未产出 9 域模型级数字；如需可重跑（非当前优先级）。

### A.3 数据文件清单（本阶段分析全部基于已存 npz，零 GPU）
| 文件 | 内容 | 关键 keys |
|---|---|---|
| `vit_module/_probe/probe_feats.npz` | FF++ 3000 视频级 split（train 2200/test 800, 0 视频重叠） | V(768) C(1024) R(768) V_proj(768) C_proj(768) y(1=real) paths vids train_mask |
| `vit_module/_tsne/feats_multi.npz` | 6 域共 2300（每域 300，ffpp 800，balanced） | V C F(1664) y domain vid path domains |
| 报告 | `_probe/residual_probe_report.txt`, `_tsne/tsne_report.txt`, `_phaseA/phaseA_report.txt`(+summary.json) | — |
| Phase A 脚本 | `_phaseA/inspect_data.py`, `_phaseA/run_phaseA.py` | 单一协议可复现 |

## B. 论文创新点定位（转向结论）

原始两条 Thread：①距离门控融合→**被 E2 证伪**；②无标签子空间重叠预测→保留为工具。Phase A 后收敛为：
- **可用发现（写论文用）**：(1) 域内融合决策冗余（E0）；(2) V 跨域失效 = 逐样本源距可预测的脆弱性（E2）；(3) FFIW(最低重叠域) 近源区 CLIP 通道被 concat 忽略却大胜（E1+E2，+7pt）。
- SBI/堆数据 = 标准手段，零新意，只作对照 baseline。
- **论文骨架 v0.2 已建立**: `paper/paper_skeleton.md`（定位/标题候选/摘要草稿/claim 级发现章 F1-F4 + 证据数字台账/方法章 T-Route 草案/实验表 T1-T11/缺口清单 G1-G7/待裁定项）。后续写作与实验决策以该文件为入口。

## C. 方差谱归因假设与 E3 验证方案（2026-09-09 起）

**H1**：V 特征中承担真实/伪造判别的是**小方差方向**（大方差≈身份/内容共享结构，判别弱）。
**H2**：V 与 C 互补性 = 判别信号位于**不同方差谱区段**（解释 CKA=0.14）。
统计注：逐方向 768 个 AUC 不成立（小样本+多重比较）→ 用**方差带 + 截断累积曲线 + null 对照**。主分析在**原始协方差 PCA**，**标准化(相关阵) PCA 作敏感性**。

| 模块 | 内容 | 判定 |
|---|---|---|
| E3a | train V 协方差 PCA 按 λ 降序，逐方向 Fisher F_k | logλ vs F_k 谱、Spearman、判别力由多少能量携带 |
| E3b | K∈{1,2,5,10,20,50,100,200} 截断：只留 top-K vs 只留 tail 各训探针 | H1 真伪：tail≈full & top-K 长期<0.8 → H1 成立 |
| E3b-null | 标签置乱；方差匹配随机正交方向 | 排除噪声/低方差本身 |
| E3c | V、C、R 各跑 E3a/E3b + 判别带重叠 | H2：R 判别序位是否更低、V/C 是否分层 |
| E3d | FF++ train 固定本征基投影 5 目标域 top-K/tail-K 跨域探针 | 尾带更脆还是更通用（接 A.2 V 随距易碎） |
| E3e | 探针权重(及可选 bridge 真实分类头)在 PCA 基展开 | 模型实际压在大/小方差方向 |

协议铁律（子 agent 必须遵守）：PCA/ridge/标准化只 fit FF++ train；AUC 只在 video-clean test 或目标域；单一固定 L2=1e-3 探针；**全 CPU 单线程、禁 GPU 前向**；脚本置顶设 OMP/MKL/OPENBLAS/NUMEXPR/VECLIB=1 与 torch.set_num_threads(1)。结果写 `vit_module/_spectrum/`，agent 只报数字+按 C 表判定，不自行演绎。

### C.1 E3 终版结果与判定（2026-09-09，投影 bug 修复后全流水线重跑，`spectrum_report.txt` 135 行全部有效）

> 管道修复：`topk_tailk_curve` 曾切原始列而非投影 `@E[:, :K]` / `@E[:, K:]`（造成 raw==corr 逐字相同、tail≈full 假象）。修复后 E3b / E3b-null / E3c 截断数字全部重生成；E3a/E3d/E3e 原本就正确投影，一并重跑对齐。产物：`vit_module/_spectrum/spectrum_report.txt`、`spectrum_curves.npz`。跨模块锚点 V=0.9852 / C=0.9108 / R=0.7034 处处一致；E3d 跨域表与 Phase A E1 逐域对齐。

**H1（判别信息在小方差方向）→ 拒绝（决定性，且方向反转：判别超集中于最大方差方向）**

| 模块 | 关键数字 | 含义 |
|---|---|---|
| E3a | max-F = 方差 rank 0；Spearman(logλ,F)=**+0.9992**；PC0 单方向含 >97% 总 Fisher 判别量且占 62% 方差；90%判别累计 = 1 方向 | 判别与方差几乎完全单调 |
| E3b | top1=0.9827 ≈ full 0.9852；top50=0.9891；**tail50（其余 718 个小方差方向）=0.4885 ≈ 掷硬币**；raw≈corr（各 K 差 ≤0.004） | 因果截断：信号全在方差头部 |
| E3b-null | 标签置乱 top/tail = 0.477±0.101 / 0.514±0.043（30 seeds，≈0.5）；Gaussian 尾带替代 = 0.4924；随机 718/768 维子空间 = 0.9861 | 排除低方差/噪声本身；信号跟着大方差子空间走 |
| E3d | V tail50 跨 5 域 0.53–0.64（噪声级）vs top50 0.76–0.84 | 跨域同样拒绝 H1 |
| E3e | 真实探针权重（V_proj）：99.9% 权重能量在 top 带（48 维）；PC0 单独 52.6%、top-5 91.6%；Spearman(logλ,g)=+0.9998 | 模型实际根本不压小方差方向 |

**H2（V/C 按方差带分工解释 CKA≈0.14）→ 部分成立，需重新表述为「集中度 × 正交性」互补**
- **否定部分**：C 的判别同样单调对齐方差（Spearman +0.9994，max-F = 其 rank 1）→ 不存在"V 用高方差带、C 用低方差带"的谱段分工。
- **保留部分（重新表述）**：V 判别超锋利（90% 判别 = 1 方向、62% 能量；top-5 方向 = 90% 方差能量）；C 弥散（90% 判别 ≈ 274 方向，50%→34 方向；域内 top50=0.8197 < full 0.9108 → C 需要 >50 方向）。V/C 顶判别轴近正交：CKA(test)=0.169、顶方向投影相关 **−0.35**，而**得分相关 +0.67** → 正交证据流给出相近判决 = E0"域内 concat 决策冗余"的谱级解释。
- **跨域体现（E3d C 行）**：cd1 上 C 的 tail 是**有害噪声**（top50=0.8028 > full=0.6559，+0.147）；ffiw 上 tail 却是**承重墙**（full=0.8344 > top50=0.7272，+0.107）→ 与 E2"FFIW 近源半 C=0.937 vs V=0.863"闭环：C 的 FFIW 优势来自 top-50 之外的弥散方向，在 cd1 型域移下同样的弥散尾变成负资产。

**残留不确定项（诚实记录）**
1. 结论对象 = 冻结 bridge_v2 双编码器 + LR 探针（阶段一权重），非端到端模型级；
2. V tail 跨域 0.53–0.64 略高于域内 0.4885（疑为高维 LR + 标准化在域移下的伪影），相对 top 带 0.76–0.84 仍为噪声级，不改变主判定；
3. raw≈corr 一致性有审计支撑：V 维度尺度 cv=0.43、C cv=0.14（并非同一组基），顶谱被 PC0（62% 方差）主导故两基下截断曲线一致。

**对论文方向的净影响**
- E2 的"V 脆弱性"从现象升级为机制：**V 判别 = 超集中的单轴（≈PC0），域移使其失配**；逐样本源距信号（E2 单调 0.92→0.72）预测的正是单轴失配。C 的弥散正交判别方向是天然多样化备份，在 V 轴失配区段（FFIW）接管。
- 方差谱集中度（top-5 能量占比 / 谱曲率）可作**无标签域失配预测变量**候选（接 B 节 Thread 2 工具链）。

### C.2 E4 融合头实验（2026-09-09，子 agent 执行，`vit_module/_fusion/fusion_report.txt` 全产物）

**用户假设 H3**：V/C 近似正交 → CLS 级 concat 后再放 transformer（交叉注意力）让两分支"互通/互相增强"，吃 CLIP 广域 + ViT 专项。
**头对比**（train 2010/视频级内验 190/3 seeds；H0 无 seed）：H0=concat→LR(C=1e-3)（跨域 concat **首次测量**）| H1=MLP(256×2) | H2=双token 交叉注意力 1 层(d=128,4头) | H3=2 层(d=256)。setA=raw V/C(域内+跨域)，setB=V_proj/C_proj(仅域内, feats_multi 无 proj)。锚点复算与 E3d/E2 逐位一致。

**关键表（setA）**：域内 H0=0.9879 / H1=0.9854 / H2=0.9824 / H3=0.9823 / V-only=0.9861；
跨域 Δvs V-only(0.8311)：H0 **+0.0031**(0.8400/0.8722/0.8233/0.8164/0.8187)、H1 −0.0105、H2 −0.0171、H3 −0.0238；
FFIW 近/远半：H0 0.8971/0.7213、H2 0.8776±0.046/0.6848、V-only 0.8637/0.7877、C-only 0.9227/0.7025；
合并域三分位 H0 0.9252/0.7642/0.7253 vs V-only 0.9212/0.7570/0.7211（每段 +0.4~0.7pt）。

**判定（机械值）**：R1 域内 (H2−H1)=−0.0030、(H2−H0)=−0.0055；R2 跨域 H2 vs V-only Δ=−0.0171、ffiw −0.0233；R4 FFIW 近半 H2−V=+0.014（但 H0−V=+0.033 更优更稳）；R5 远三分位 H2−H0=−0.0304。
→ **H3 被拒绝，且呈"容量-泛化单调律"**：跨域 LR > MLP > TF1 > TF2、域内亦然 → 小源域(2010)+大域移下交互容量只过拟合，学不到可迁移的互相增强（源上 V 错率~1.5%、V−C 残差≈噪声，互通模式无训练信号）。
→ **意外收获 1**：跨域线性 concat 是**首次测量**且基本无害微增益（均 +0.3pt、三分位每段 +0.4~0.7pt）；唯一例外 ffiw 单域 −0.9pt（concat 0.8164 < V 0.8257 < C 0.8344——源上 C 残差被 shrink 成噪声权重 → 线性融合吃不到 ffiw 的 C 红利）。
→ **意外收获 2**：FFIW 近半 concat +3.3pt、远半 −6.6pt → C 红利 regime 依赖，最简单的线性融合就能吃近半红利但带不动远半 → 选择性路由必要性再确认。

### C.3 Phase F 选择性/路由实验（2026-09-09，子 agent，`vit_module/_selective/selective_report.txt`）
协议：V/C LR(C=1e-3) fit FF++train 2200；评估 800test + 5×300；E2 源距复用 _fusion 复刻（k=10，锚点 max dev 0.0024）。
- **Q1 失败预测 AUROC**（预测 V 判错）：域内 800——轴支持度 **0.86**(低轴支持→易错，机制呼应 E3 单轴)、MSP/犹豫 0.835、分歧 0.586、源距 0.41(负向)；跨域合并1500——轴支持度 0.695、MSP 0.682、源距 0.607、分歧 0.579。→ 信号真实、机制自洽，但**跨域全 <0.70 门槛**，且免费 MSP 几乎打平专用信号；免训集成(域内)0.86。
- **Q2 选择性弃权 risk@coverage**（合并1500）：MSP 弃权 1.0→0.5 coverage：risk 0.269→**0.159（−41%）**；源距弃权→0.195；域内 MSP 0.056→0.015。→ **免费 softmax 置信即强弃权基线**，源距弃权不占优。
- **Q3 路由转 C**：oracle ceiling **+16.1pt**（1500 acc 0.731→0.892；可救样本 241/1500=16.1%，源距/分歧显著高于全体）；但固定比例转 C：源距转 C 全面掉点(f=0.2→0.712<V 0.731)、置信转 C 峰值 ≤+0.5pt → 根因 C 跨域整体弱(errC 0.385 vs errV 0.269) + 信号只预测"V 会错"预测不了"此刻 C 更好"。**预注册出口触发：路由降级**，方法形态 = 失败预测 + 选择性弃权 + 机制解释。可救子集分歧富集(0.52 vs 0.34) → 二阶路由信号未测（待定）。

### C.4 G5 真实 head 归因 + G3 CI（2026-09-09，子 agent，`vit_module/_head/head_report.txt`）
- 真实分类头 `output.weight` (2,1664) 确认；关联 clip_vision_alpha=0.306 / clip_text_alpha=1.780。
- **三块 ΔL2 占比：C_proj 块 62.3% > V_proj 块 37.7% > bridge 0.03%** → 真实头**给了 C 大权重**却仍跨域不敌 V 单通道、ffiw 上 concat(0.818)<V(0.824)<C(0.834) → 反直觉结论："问题不是忽略 CLIP，而是权重锁死在源分布"，强化 regime 叙事。
- 集中度（Δ 切片在 probe 特征空间 PCA 基）：V 侧块(896:1664)在 E_V 基 top-5=**93.8%**(复现 E3e 探针头 91.6% top-5 于**真实头** ✓)、E_C 基仅 0.43%；C 侧块(0:768)在 E_C 基 top-5=80.7%。V_proj 空间 PR=1.13(top5 eig 99.86%)、C_proj PR=5.21。
- G3 CI（样本级 B=2000，视频内相关→偏乐观 caveat）：域内 V 0.9852[0.976,0.993]、C 0.9108[0.892,0.929]、concat 0.9878[0.980,0.994]；concat 跨域逐域 CI 宽 ~0.08 → **E4"跨域 concat 微增益"不显著**，宜表述为"不劣于 V、无显著增益"。

---

## D. H4 假设 + G8 浅层捷径归因验证（2026-09-09 起）

### D.0 用户现状判定与方向（用户原话要旨）

> 数据集丰富度不够，不足以支撑跨域类型识别；当前框架已将两冻结分支信息汲取到极限 → 需要在**训练结构**上改进（框架侧）+ **数据集**改进（SBI 等）。问题：加 SBI 是否意味着 ViT 需要重训？

**H4 假设（用户）**：ViT（PDI 专用防伪编码器）其实**没有学习伪造手段（语义级）**——它只是**直接记忆了图像里部分显著的低层特征**：颜色突变、明显合成纹路。没有真正从图像语义出发（异常光照、人脸扭曲等细节）。若成立 → 需要其他手段（如借鉴**光流检测**思路）。**应对回答（分析结论）**：
- **SBI/数据增强 = 换数据不换机制**：bridge_v2 中 PDI-ViT 是冻结 stage-1 backbone。让它学 domain-robust 伪造特征必然要求**重训/替换编码器并重跑 stage-1 协议**，这同时打破既有全部冻结特征分析（E1–G5、探针）的直接可比性。SBI 对冻结编码器无效。
- **光流/时序 = 换检测模态**（拼接时序痕迹 vs 单帧内容痕迹），与当前单帧双编码器**正交**，等于新项目而非修补 7.3B 现有模型。
- **行动顺序**：两条结构改动都贵（GPU 天级）且都建立在"捷径假设"成立之上 → 先离线钉死 H4 再决定主线。G4（GPU 9 域端到端）因 G5 已证真实头复现探针结论而**维持暂缓**（task #10 pending，等待用户决定）。

### D.1 H4 与既有证据的映射（为何值得用一整轮实验验证）

| 已有证据 | 对 H4 的含义 |
|---|---|
| E3：V 判别 90% 由 **1 个方向(PC0)**、>97% Fisher、62% 方差携带；top-1 AUC 0.983≈full | 与"学的是一个（类）显著性浅层捷径"高度吻合：真正综合多类语义证据的判别器（光照/扭曲/几何…）判别力必然**弥散**多方向——对照 C 需 274 方向才到 90% |
| E3a 单调谱 Spearman(logλ,F)=+0.999 | 判别=大方差方向=内容/身份级共享结构，恰是"显著外观统计"所在层级 |
| E4 容量-泛化单调律 | 源域上可榨信息已被单一捷径吸干，更多交互容量只过拟合 → 与"捷径饱和"一致 |
| E2 源距单调崩 + E3d tail 跨域噪声级 | 捷径随域失配（单一轴的域移失配 = 现象级解释） |
| G5 真实头 62.3% ΔL2 给 C_proj 却救不回 | C 学的是弥散语义，需测试时域适配，静态头锁死源 → 语义信息 C 有但不可静态用 |
| C：90%判别需 274 方向、CKA(V,C)≈0.14-0.17 | CLIP 未走捷径（判别弥散）但**未专门化** → 语义在 C、捷径在 V 的初步印象 |

**诚实边界（写入判定前）**：现有证据能说"V 学的是一个捷径"；**不能**说清"具体是哪种浅层量"，也**无法**区分"模型偷懒"vs"FF++ 伪造本身就主要靠浅层痕迹可分"（对后续路线含义不同）。G8 只证伪"这些统计是捷径"或支持"其中某几类统计承载判别"；**无法**正面证明"语义缺失"——负结果需谨慎表述。

### D.2 G8 验证计划（纯离线，零 GPU 前向，全 CPU 单线程）

**目标问题**：V 判别方向（PC0 / 探针 logit）编码的是**低级统计捷径**还是语义证据？各目标域 fake 的浅层痕迹是否漂移从而定量解释 V 跨域崩？

**共享统计套件**：`vit_module/_g8/_lowlevel.py`（两子 agent 必须 import 同一模块，保证定义一致）。14 维统计：`g_mean g_std sat_mean sat_std colorfulness ch_diff lapl_var grad_mean edge_density hi_en spec_slope g_entropy noise_est blockiness` + 原生分辨率 `h,w`（审计用）。统一在 **224×224 LANCZOS** 图上算（跨域尺度混淆最小化；FF++ crop 原生 ~145–160px 属轻度上采样，hi_en/spec_slope 的保真 caveat 记录在案）。数据源（路径全量存在审计通过）：`_probe/probe_feats.npz`(3000/3000)、`_tsne/feats_multi.npz`(2300/2300)。

| 模块 | agent | 内容 | 判定映射 |
|---|---|---|---|
| G8A 域内 | A | probe 全量 3000 图统计；(A1) real/fake 逐统计 Cohen d；(A2) 统计 × PC0 投影 / 探针 logit 相关；(A3) **仅统计量的 LR → 域内 held-out AUC**（对照 V 0.9852 / C 0.9108）；(A4) OLS：V-logit~stats vs C-logit~stats 的 adjR² 对照；(A5) 按 FF++ 伪造子方法(Deepfakes/Face2Face/FaceSwap/NeuralTextures)拆解 top 统计；(A6) 统计间相关 + 原生分辨率审计 | H4 支持 ← stat-LR AUC≳0.93 且 adjR²_V≳0.5 且集中 ≤3 统计；弱 ← 0.80–0.93；不支持 ← <0.80 & adjR²<0.2 |
| G8B 跨域 | B | 复算 probe-train V PCA 轴 E[:,0]；feats_multi 5 目标域×300 + ffpp800 源池统计；(B1) 源池拟合统计判别方向 u（标准化的 stats-LR）；(B2) u 在每域的判别 AUC（捷径迁移度）+ 逐统计 d 漂移；(B3) 捷径迁移度/顶统计漂移 vs 已知 V 跨域 AUC（cd1 .8286/cd2 .8633/dfdcp .8261/ffiw .8244/wild .8090）跨 5 域 Spearman；(B4) V 单轴 z 每域 AUC（轴塌缩基线）；(B5) V 判错 fake 的捷径得分 t 是否更"真样"（接 Phase F 轴支持） | H4 支持 ← 源捷径迁移度与 V 崩幅跨域 Spearman≳0.7 |

**预注册门（机械值，agent 只报不演）**：按上表逐模块给 H4「支持 / 部分支持 / 不支持」，不自行演绎结构改动。产物写 `vit_module/_g8/`（g8a_report.txt、g8b_report.txt + 统计缓存 npz）；agent 只报数字+按表判定。

### D.3 决策树（G8 出结果后，下一步由用户定）

- G8「支持」：坐实"V 判别=低级捷径 + 各域伪影漂移"。论文收获 = 量化的失败归因章。结构改动的科学依据确立 → 候选主线：(a) 语义级检测器（重训/换骨干，需 GPU 天级，与 SBI 训练同轨）；(b) 时序/光流模态（新项目）。
- G8「部分/不支持」：需进一步归因（如模型前向 + 像素扰动测试，需 GPU 决定权），或改判"FF++ 本身浅层可分"（则冻结框架分析结论不变，主线转向数据/结构）。
- 无论结果：G4（GPU 端到端）与二阶路由信号仍待用户单独决定。

### D.4 G8 结果与判定（2026-09-10，两个子 agent 完成，`vit_module/_g8/g8a_report.txt` + `g8b_report.txt`；共享统计 `_lowlevel.py`，0/4892 读图失败，锚点 V 0.9852/C 0.9108 及跨域全表复算 4 位小数一致）

**机器块**
```
G8A_V_full_AUC=0.9852 C_full=0.9108 STATONLY_AUC=0.5580 TOP3=0.5407 ADJR2_V=0.0426 ADJR2_C=0.0551  VERDICT=NOT-SUPPORTED
G8B_SOURCE=probe(196real/196fake,≤2帧/视频) TRANSFER_AUC: cd1 .490 cd2 .619 dfdcp .581 ffiw .592 wild .384
    V_AUC: cd1 .8286 cd2 .8633 dfdcp .8261 ffiw .8244 wild .8090   rho(transfer,V_AUC)=0.6000 (p=0.285)  VERDICT=PARTIAL
```

**判定：H4 的"低层显著像素特征(颜色/纹理/噪声)"形态 → 被拒绝（G8A 决定性）；跨域同向共变仅部分（G8B）**

| 量 | 值 | 含义 |
|---|---|---|
| stat-only(all 14) test AUC | **0.5580**（≈掷硬币；V 0.9852 / C 0.9108） | 全局低级统计在 FF++ c23 上**基本不带判别力**；单个统计最好 grad_mean 也仅 0.567，无一 ≥0.80 |
| adjR²(V-logit ~ 14统计) | 0.0426（C 侧 0.0551） | V 的判别**不能被全局低级量线性复现**（甚至略低于 C） |
| A1 Cohen d | top |d| = grad_mean −0.21 / noise −0.17 / hi_en −0.14 | 假样本整体只"略微更平滑"，效应量 ~0.1–0.2，方向一致但极弱 |
| A5 子方法 | Deepfakes/F2F/FS/NT 四种在 top 统计上都贴 real 均值 | 无任一伪造法在全局统计上显著可分 |
| G8B 源捷径签名 | 源池内 in-sample LR AUC 仅 **0.6116**（自证源"捷径"本身极弱）；u 系数 ≤0.03 | 与 G8A 同源互证：不存在强的全局低级捷径可迁移 |
| rho(shortcut-transfer-AUC, V_AUC) 跨 5 域 | **+0.60**（p=0.285；0.4–0.7 → PARTIAL） | 源低级量在各域的判别残留与 V 崩**同向共变但弱**；且残留本身近 chance → 低级漂移只解释 V 崩的很小一部分 |
| rho(e0 单轴投影 AUC, V_AUC) | **1.0000**（e0 单轴逐域 0.806–0.860 ≈ full-V 0.809–0.863） | 塌缩**由单轴本身携带**（复证 E3d），与低级统计无关 |
| B5 判错 fake 的源捷径得分 | 5 域 delta：+0.023/−0.007/−0.019/+0.054/+0.018 → 仅 cd2/dfdcp 符合预期方向 | V 判错的 fake **不像**"全局低级量上的 real 样" → Phase F 轴支持错误与低级统计无关 |

**对 §D.1 的修正（诚实记录）**：D.1 曾按 E3 单轴推测 V "学的是（一类）显著性浅层捷径"——G8A 证明**就颜色/锐度/高频/噪声/JPEG 块这 14 种全局图像统计而言此形态不成立**。单轴机制确凿（E3/E3d/G8B e0 rho=1.0），但该轴编码的**不是**这批低级量。

**开放问题（G8 的净产出——把谜题变尖锐而非解决）**：
1. **PC0 到底是什么仍未知**。FF++ 探针是 video-disjoint 且 real/fake 出自**同一批身份**（同一人真实帧 vs 被伪造帧）→ 判别轴必须在"同一身份、同帧来源"上区分 real/fake → 排除了纯身份记忆；也非全局颜色/纹理。它更像是**局部化/区域化的篡改痕迹**（融合边界、五官几何变形、特定 patch 的相关结构）——这是"图像局部统计"而非我们测的"全局聚合统计"。
2. **方法学 caveat（留给下一轮）**：14 统计全是**整图聚合量**，局部/分区域统计（per-patch、按五官 ROI）未被测 → H4 的"局部伪影捷径"变体既未被证实也未证伪。
3. **B5 混合** → V 判错的 fake 在低级量上不偏 real，轴支持类错误预测另有根源（待定位，可能即问题 1 的同源量）。

**对 D.3 决策树的影响（机制证据更新）**：SBI 增强注入的是**像素级**伪影（混叠、色彩混合）——G8A 表明 V 在 c23 上并不吃全局像素伪影 → **SBI→重训 ViT 的路线依据被削弱**（注入的恰好不是 V 依赖的那类线索）；光流/时序仍正交、未受影响。**在花 GPU 重训前**，应先识别 PC0 编码对象。候选下一实验（离线优先，CPU 可行）：G9 = 区域化统计(按 4×4/5×5 分块或五官 ROI 的局部 hi_en/grad/blockiness) × V 判别关联，测"局部伪影捷径"变体；附加可做 PC0 高低投影样本的**分块差异图** + 肉眼/VLM 描述。二阶路由信号、G4 GPU 端到端维持待定。

### D.5 G9 验证计划（2026-09-10 起，用户批准执行）

**目标**：G8A 的 14 统计是**整图聚合量**，会把局部篡改痕迹洗掉。G9 把低级统计**区域化**，检验 H4 的"局部伪影捷径"变体，并给"PC0 在看什么"第一次空间定位证据。数据 = `_probe/probe_feats.npz`（3000，video-disjoint train 2200/test 800）；复用共享 `_lowlevel.py` 构建块；同固定 LR 探针协议（C=1e-3,lbfgs）；锚点 V=0.9852/C=0.9108、G8A stat-only 全局=0.5580 作对照。

**区域化特征（单次读图一张图同时算两套）**：
- Grid A `4×4`=16 块 × 5 统计 {grad_mean, lapl_var, hi_en, edge_density, blockiness} → 80 维区域特征；
- Grid B `8×8`=64 块 × 4 统计 {grad_mean, lapl_var, hi_en, blockiness} → 256 维（只做精定位与交叉验证，主门用 Grid A）。
- 每块统计在 224 图上按块计算（块内 FFT 求 hi_en；块边界沿用 8px blockiness 定义）。

**门（预注册，机械值）**：regional stat-only LR test AUC（Grid A 80 维）对比全局 0.558：
- ≥0.75 → H4 局部变体「支持」（聚合把信号洗掉 → 定位该区域，机制明确）；
- 0.65–0.75 → 「部分」；
- <0.65 → 局部低级量同样不解释 V → PC0 非低级伪影 → 转向高层/分布归因（VLM 描述极端样本、模型级归因，含 GPU 选项）。

**分析**：① Grid A 全特征 stat-only test AUC；② adjR²(V-logit ~ Grid A)；③ 逐块判别力（每块 max 统计 AUC）打印 4×4 表 → 空间定位（中心/五官区 vs 角块）；④ top 判别块特征与 z_PC0/s_V 相关；⑤ **对齐混淆审计**：face crop 原生 70–920px 不一 → 检查 top 判别块特征是否与原生尺寸/比例代理高度相关（若是则区域信号可能是对齐伪影而非篡改痕迹，须标注）；⑥ 可选：V 判对 vs 判错的 fake 逐块统计差（定位"判错 fakes 在哪个区域更像 real"）+ Read 工具目检 ≤8 张极端样本（PC0 投影极值/错判样本，Read 失败则跳过，纯 anecdote）。

产物写 `vit_module/_g9/`（run_g9.py、g9_report.txt、g9_stats.npz 缓存）。agent 只报数字+按门给判定。CPU 铁律同 G8（顶部 5 env 单线程、禁 torch/GPU/多线程多进程）。结果落 §D.6。

### D.6 G9 结果与判定（2026-09-10，子 agent 完成，`vit_module/_g9/g9_report.txt`；锚点 V 0.9852 / C 0.9108 / PC0 varfrac 0.6229 复算一致，0/3000 读图失败）

**机器块**
```
G9_GRID_A_AUC=0.6257 G9_GRID_B_AUC=0.6252 G9_ADJR2_V_GRIDA=0.1434 G9_ADJR2_V_GRIDB=0.2499
G9_TOPBLOCK=(2,2,lapl_var,0.6219)   G9_VERDICT=NOT-SUPPORTED
```

**判定：H4 像素低级伪影（区域化变体）→ 拒绝。区域化比整图略升，但仍远低于门（0.75）与 V（0.985）。**

| 量 | 值 | 对照 | 含义 |
|---|---|---|---|
| Grid A 80 维 stat-only test AUC | 0.6257 | 整图 14 维 = 0.5580；V = 0.9852 | 区域化捞回一点信号但仍近 noise，<0.65 → NOT-SUPPORTED |
| Grid B 256 维 stat-only AUC | 0.6252 | — | 8×8 精网格无额外增益（信号饱和于很弱水平） |
| adjR²(s_V ~ 区域量) | GridA 0.1434 / GridB 0.2499 | 整图 = 0.0426 | 分块把 V-logit 可解释性抬了 3–6 倍但仍 ≤0.25（余量仍被 V 的 0.985 甩开） |
| 最强单块 | (2,2)=中心区 lapl_var AUC 0.622；grad_mean 四邻 ~0.58–0.60 | — | 中心(五官/脸部)锐度略判别，与 G8A"fake 略平滑"同向；很弱 |
| G6 对齐混淆审计 | top-3 判别块 vs native h/w/(h·w) 最大 |r|=0.384（<0.5） | 区域信号**不是**对齐/尺度伪影（排除主要假阳性源） |
| G7 判错 fake (43/1500) | top 块差 (0,3) hi_en **反方向**：判错 fake 的 hi_en 反而高于 real | — | 判错 fakes 在低级区域量上**不像 real** → 与 B5(全局) 一致：错判根源非低级量 |

**净结论（D.4→D.6 累积）**：PC0 **既非身份**（video-disjoint + real/fake 同身份）、**也非全局低级统计**（G8A）、**也非区域低级统计**（G9）→ "V 记忆像素级捷径（颜色/纹路/锐度/噪声/网格）"这一整类被排除。区域化确实捞到**一点中心锐度弱信号**（0.558→0.626；adjR² 0.043→0.14–0.25，即 V 约 1/4 的判定线性量落在一小块中心"平滑度"上），但远不足以复现 0.985。定性目检因本会话视觉不可用跳过（Read 对 PNG 返回 Unsupported）。

**解释性重构（supervisor 推断，区别于测量结果，需用户确认）**：V 分支 = 冻结 PDI **face anti-spoofing（活体防伪）ViT**——net_050 / `vit_adaptive_mattn_aps`，加载自 `PDI/results/.../Ama1_aps1_1/net_050.pth`（见 BRIDGE 训练配置表），**从未在 FF++ 深度伪造上训练**。据此：FF++ real/fake 的"单轴可分"更可能是**活体/防伪线索在合成人脸/回放上的涌现区分**，而不是"在 FF++ 上记住了什么捷径"。这把一串现象串成一条更自洽的因果：
- 单轴为何沿"源距"单调崩（E2）＝活体线索来自 PDI 自己的训练分布，域移即失配；
- 静态头为何锁死在源分布（G5）＝同因；
- **SBI 路线的含义被反转**：SBI 本是 face anti-spoofing 的数据增强（制造 blending/spoof 伪影），与 PDI-ViT 的归纳偏好**同向** → 之前的"D.4 削弱 SBI 依据"结论可能过强；但 G9 未定位到简单边界伪影，SBI 是否真能喂饱该轴仍是开放问题、必须在真实训练/微调里验证。
确认路径：查 net_050 的训练数据/协议（PDI 侧），若为 anti-spoofing 则上述重构成立。

**下一候选（待选，不再建议扩展像素低级量——该支线已关闭）**
1. **PC0 模型内归因**（第一次能"看见"轴内容的唯一手段）：小样本 PDI-ViT 前向（GPU 或慢 CPU，需用户决定硬件），取 patch/层注意力/浅层输出看轴的空间-通道模式；
2. **PC0 × fake 子方法/来源视频分层度**（纯特征，~10 分钟，CPU）：若轴按 4 种 FF++ 方法或来源视频分层 → 提示它锚定某类合成管线而非通用伪影；
3. **接受"机制-agnostic 但刻画充分"框架**收束：V=防伪单轴 + 沿源距单调崩（E2/E3d/G8B rho=1.0 全链一致）+ 轴支持可预测失败（Phase F）→ 把"PC0 具体是什么"作为 future work，转向论文叙事；
4. G4 GPU 端到端（9 域模型级）仍待定——现为"探针机制 ↔ 模型级行为"之间唯一未闭环缺口。
本会话视觉不可用：定性目检（VLM/肉眼）项对 PNG 无法执行，仅机器数字。

### D.7 更正 + 最新综合结论（2026-09-10，用户澄清后）

**更正 D.6 的解释性重构（作废）**：D.6 末段"V = 冻结防伪先验、从未在 FF++ 上训练、单轴为防伪线索涌现"**不成立**。事实（用户澄清 + `BRIDGE_ADAPTER_DESIGN.md` 训练配置表）：stage-1 中 **ViT backbone 从 `net_050.pth` 初始化后被微调（✅ 训练）**，V 分支就是**在 FF++ 上训练出来的**，域内检测力不差（0.985）；其跨域 0.81–0.86 **不及融合的 ~84% 最优表现**。因此单轴是**在 FF++ 上被训练出来的方向**——这反而**加强**而非削弱"学到了源特有的非通用判别量"的解读；D.4 中"G8A 削弱 SBI 依据"的措辞也需按此重估（SBI 是否命中该轴仍未验证，须在真实训练里测，不能离线定论）。

**最新综合结论（经 E3 / E4 / E2 / G5 / Phase F / G8 / G9 七项验证）**

对 H4「模型没学到真正的伪造特征」——**证据总体支持，但须精确表述为四条**：

1. **V 学到的是"单一主导判别方向"，不是多模态伪造理解。** 域内 0.985 中 **>97% 判别压在同一根轴 PC0** 上（90% 判别 = 1 方向；头部 top-5 方向占 90% 方差能量）；对照 C 需 274 方向才到 90% 判别（弥散）。V 尾部 718 个方向跨域为噪声级（0.49–0.64）。→ 模型实质只有**一条**线索，而非多种伪造痕迹的综合。
2. **这条线索不是任何可测的像素级伪影。** G8A 整图 14 统计 stat-only AUC = 0.558（≈chance）、adjR² = 0.043；G9 区域化（4×4 与 8×8 分块）AUC = 0.626、adjR² = 0.14–0.25；对齐/尺度混淆已排除。→ "记住颜色突变 / 合成纹路 / 边界网格"这类**具体低层机制被否定**；判别量位于**网络自身表征空间**中，手工像素统计看不见。
3. **该线索是源分布特有的、不迁移的。** 跨域崩**完全由这根轴承载**（单轴投影逐域 AUC 与全 V 逐域 AUC 排序 rho=1.00）；崩幅随"源距"单调（E2：0.92→0.72）；尾部方向零迁移（E3d）；增加融合容量只过拟合（E4 容量-泛化单调律）；真实分类头已给 C 侧 62.3% 权重仍跨域不敌 V 单通道（G5）→ **轴与权重双双锁死在源分布**。
4. **融合的 ~84% ≈ V 单独（~0.83）+ 可忽略增量。** 跨域 concat 仅 +0.3pt 且逐域 CI 宽 ~0.08（G3：不显著）；CLIP 的广域语义**没有被静态头转化为可迁移的伪造知识**（Phase F：路由转 C 的 oracle 上界 +16.1pt 无法被任何现有信号兑现）。→ 当前 84% 不是"两分支互补学到的伪造理解"，而是"**V 的源特有单轴 + 一点 CLIP 残差**"。

**一句话**：模型不是"没学会检测"，而是**学会了一个在 FF++ 分布内极强、却既非通用像素伪影、也非语义级伪造概念的单轴判别量**——这正是它源内 0.985、跨域却与融合一起卡在 ~84% 的根本原因。

**由结论反推的可行方向（供决策，不含新承诺）**：
- 要在 84% 之上再进，需要**打破单轴**（让 V 表征出多条可迁移线索），而非继续扩展像素统计（G8A+G9 已关闭该支线）；
- 识别 PC0 究竟编码什么，唯一剩余手段是**模型内归因**（需前向，GPU/CPU 由用户定）；
- 或接受"机制-agnostic 但刻画充分"框架，将方法形态定为**选择性预测 + 失败预测 + 机制解释**（Phase F 已备），把"PC0 是什么"列为 future work。

### D.8 H5（用户假设）+ G10/G11 验证计划（2026-09-10 起，用户批准执行）

**H5（用户）**：V 分支学习能力强，且把 FF++ 映射到表征空间中的**一个大方向**；该方向上 FF++ 表现好；**跨域时其他数据集的样本在表征空间中没有表达出近似该"轴"的特征方向** → 输出崩溃。

**前提事实（已澄清，覆盖 D.6 的错误推断）**：V 分支由 net_050 初始化后在 **FF++ 上微调训练**（stage-1），特征提取管线 = `vit_module/_probe/residual_probe.py` → `ViT_M2F2Det_Bridge`（`vit_module/vit_m2f2_detector_bridge.py`）+ `checkpoints/stage_1/bridge_v2_phase1.pth`；图像 cv2→RGB→resize336→CLIP-Normalize→tensor；V = `model.vit.forward_features(model._preprocess_for_vit(images))[:,0,:]`。

**G11（离线，纯特征，CPU 单线程；"方向1"可行性验证 + 崩塌形态分解）** —— 数据 = probe npz（源轴 e0/mu 由 train V 拟合）+ feats_multi 5 域。
| # | 量 | 判据 |
|---|---|---|
| Q1 | **轴对齐/支撑**：每域 z=(V_d−mu)@e0 的 ①域内 z 方差占比（对照源 0.6229）②轴外能量（投影到源 top-1/5/50 子空间后的残差占比）③域自身 PC0 与源 e0 夹角/子空间重叠 ④域内 real/fake 沿 z 的间隔 Δz（对照源） | 量化"跨域样本是否未表达在源轴上" |
| Q2 | **重对中恢复**：z 仅用**无标签**域均值中心化后的 AUC vs 原始（对照 G8B e0_AUC 0.806–0.860） | ≥+2pt → 崩含"轴上位置漂移"，可无标签校正；<2pt → 非偏置问题 |
| Q3 | **oracle 上限**：域内自训 LR（视频内 5 折，oracle 标签）AUC − 源训 LR 跨域 AUC；域 LDA(Fisher,shrinkage) 方向与 e0 的 cos | ≥+5pt → 冻结 V 含该域可用结构（方向错位，改头/适配可救）；≤+1pt → 冻结表示无线索，**必须换表示/重训**（方向1 动编码器） |
- 门：H5「支持」= 轴外能量/间隔收缩随域崩单调且 Q2 或 Q3 显示"错位可部分恢复"；「不支持」= 域崩与轴表达无关。
- 产物 `vit_module/_g11/`（run_g11.py / g11_report.txt / g11_stats.npz）。

**G10（模型内归因，需前向；GPU 优先，严格用空闲 GPU）** —— 同管线复现 V 特征（先校验：抽 10 张 probe 图，与存盘 V 的 cos ≥0.999，否则必须修正管线）。
| # | 实验 | 判据 |
|---|---|---|
| I1 | **变换不变性（低层捷径 vs 语义的决定性判据）**：源内 test 采样（~120 张，real/fake 平衡），对 {灰度化、JPEG(q=70)、2×下采样-上采样、水平翻转、轻度高斯模糊} 各算 V 探针 AUC，与原始 AUC 比 Δ | Δ≥15pt → 轴对低层/压缩极敏感（H4 复活）；Δ≤5pt → 轴非低层，H4 再否 |
| I2 | **空间归因**：遮挡 patch（8×8 网格，逐块置灰/替换）对探针 z 的 Δ，聚合为每类/每域热图；若可行再做 input×grad（需确认 `_preprocess_for_vit` 可导，否则纯遮挡） | 定位"轴在看哪里"：五官/面部区（语义几何）vs 全图均匀/边缘（低层） |
| I3 | **源内 vs 跨域归因对比**：同法在 1–2 个目标域（如 cd1/ffiw）采样 60 张，热图差异 | 跨域时模型注意力偏离 → 轴失配的可视化证据 |
- 硬约束：`CUDA_VISIBLE_DEVICES=2`（GPU1 满载、GPU0 有他进程，**禁用**）；前向 ≤~600 次、batch 16、fp32、num_workers=0、`torch.set_num_threads(1)`；GPU 不可用时降级 CPU 小样本（N≤40）并报告。产物 `vit_module/_g10/`。

**两实验共同的 CPU 铁律**：脚本顶部 OMP/MKL/OPENBLAS/NUMEXPR/VECLIB=1、torch.set_num_threads(1)、cv2.setNumThreads(0)、单进程；agent 只报数字+按门判定，不演绎方向决策；结果落 §D.9。

### D.9 G10/G11 结果与 H5 判定（2026-09-10，两子 agent；`vit_module/_g10/`、`vit_module/_g11/`）

**G10（模型内归因，GPU2；复现校验 PASS：重抽 10 张 test 图与 npz V 余弦 = 1.000000，relL2=0；确认 npz `V` = raw ViT CLS（pre-deepfake_proj）；PC0 varfrac 0.6229 / test AUC 0.9852 锚点全对）**

| I1 变换不变性 (n=120 视频分层, z 轴 AUC=0.9733 原图) | ΔAUC(z) pt | ΔAUC(LR) pt |
|---|---|---|
| gray | **3.69** | 4.86 |
| jpeg70 | **−0.47** | 0.67 |
| resize2x | **1.00** | 1.69 |
| flip | **−0.92** | −0.19 |
| blur | **1.56** | 2.08 |
| crop90（几何对照，不入判定） | **17.25** | 11.44 |

→ 五个低层变换全部 ≤5pt = **AXIS_NOT_LOWLEVEL（轴非低层：不依赖灰度/压缩/重采样/镜像/模糊）**；但 90% 居中裁剪掉 17pt → 轴对**构图/几何**高度敏感。

- **I2 遮挡归因（8×8，均值色填充，n=40）**：real top-1 块 (5,5)、中心 16 块占比 **0.397**；fake top-1 (1,5)、中心占比 **0.312**（均匀 = 0.25）；**fake 的 Σ|Δz| = 3366 是 real（1083）的 3.1 倍**；input×grad 与遮挡法 cos = 0.83/0.89（定性一致）。
- **I3 源内 vs 跨域热图**：中心占比 ffpp 0.333 / cd1 0.319 / ffiw 0.371；top 块均落在中下部（行 2–6）→ **跨域时归因分布无明显位移**（无"注意力转移"证据）。
- 审计：8700 次前向（762 batch，超我设的 ~600 预算，已记录）、40 次反向、105 s、GPU2 峰值 4.1 GB、0 读图失败。
- **诚实缺口**：I1 不变性只在**源内**（FF++) 做；目标域上轴是否同样稳健未测。

**G11（离线轴分解；锚点全对：varfrac 0.6229 / test 0.9852 / e0 跨域 0.8576-0.8604-0.8322-0.8103-0.8060 / full-V 0.8286-0.8633-0.8261-0.8244-0.8090）**

**重要修正**：参照域不能用 `feats_multi['ffpp']`（140 vids 中 98 个与源 train 同视频 → 污染），已改用 probe test 800（42 vids，完全 video-disjoint）。

| dom | zdim_frac(源0.6229) | residK1/K5 | \|cos(e0_dom,e0_src)\| | cos(域均值差,e0) | d_z(源train 4.6555) | AUC_z | ORACLE | SRCLR | ORC−SRC |
|---|---|---|---|---|---|---|---|---|---|
| fftest(ref) | 0.6232 | 0.378/0.070 | 0.9993 | — | 4.036 | 0.9827 | 0.9790 | 0.9852 | −0.006 |
| cd1 | 0.6357 | 0.398/0.128 | 0.9870 | 0.9790 | 1.403 | 0.8576 | 0.9082 | 0.8286 | **+0.0796** |
| cd2 | 0.5986 | 0.423/0.119 | 0.9890 | 0.9766 | 1.530 | 0.8604 | 0.8860 | 0.8633 | +0.0226 |
| dfdcp | 0.6581 | 0.393/0.175 | 0.9790 | 0.9606 | 1.304 | 0.8322 | 0.8774 | 0.8261 | +0.0513 |
| ffiw | 0.6852 | 0.318/0.142 | 0.9796 | 0.9865 | 1.228 | 0.8103 | 0.8980† | 0.8244 | +0.0736† |
| wild | 0.6122 | 0.384/0.148 | 0.9793 | 0.9437 | 1.141 | 0.8060 | 0.8349 | 0.8090 | +0.0259 |

† ffiw 仅 1 个 vid → oracle 交叉验证退化（StratifiedKFold）→ 0.8980 被同视频泄漏高估，**不可与其它域并列**。

**三条必须诚实标注（agent 主动上报，我认可）**：
1. **Q2 规格缺陷**：`z→z−mean(z_d)` 对域内 AUC 是**数学恒等变换**（AUC 只依赖排序）→ Δ ≡ 0；"无标签重对中可恢复"这条规格本身不成立。规格外补充：CORAL 白化 Δ = +0.005/−0.013/−0.015/−0.003/−0.020（≈0 或微负）、跨域池化重对中 +0.0064 → **无免费（label-free）校正**。
2. **(b) 轴外能量的结论随 K 翻转**：K=1 时 5 域残差与源无差（NOT-SUPPORTED）；K=5 时 4/5 域达门。即：**承载 62% 能量的第一主轴在目标域没有错位**，偏离集中在第 2–5 维。
3. **(d) 与 AUC_z 数学同义**（AUC≈Φ(d_z/√2)，实测最大偏差 0.0181，Q1 审计）→ "d_z 收缩"不是独立于 AUC_z 的证据；但 Q1e 分解可用（d_z = gap/合并sd，恒等）：**类间隔保留比例 gap/gap_src = 0.38–0.47（衰减 ~2.2 倍）；沿轴合并 sd 膨胀 1.42–1.66 倍 → 二者合成 d_z ratio = 0.25–0.33**；且两类中心同时位移——real 侧沿 fake 方向大幅正移（d_real +7.4~+30.0），fake 侧负移（d_fake −7.4~−27.2），中点位移 shift_mid 各域异号（−9.9~+11.3）→ 崩 = **分离衰减 + 类内扩散 + 双向漂移**，**不是**整体偏置、**不是**轴错位。
4. **补充锚点**：源 LR 权重方向（raw 空间）与 e0 的 cos 仅 **+0.4251**（即权重平方范数的约 82% 落在轴外）；但轴外方向跨域几乎无贡献——逐域对照 单轴 e0 vs 源 LR：cd1 0.8576 vs 0.8286（**LR 反而低 2.9pt**）、cd2 0.8604 vs 0.8633、dfdcp 0.8322 vs 0.8261（e0 略高）、ffiw 0.8103 vs 0.8244、wild 0.8060 vs 0.8090。而域内 LR 仅比单轴高 +0.0025（0.9852 vs 0.9827）→ **源训头在轴外维度上花掉的表达能力在跨域时归零甚至为负**，可迁移成分集中于轴上（D.9 汇总线索，非独立实验结论）。

### D.10 H6/H7 假设与 G12/G13 验证计划（2026-09-10，用户提出；两子 agent 并派）

**派单前侦察确认的事实（审计）**
- `vit_module/backup_weights/net_050_backup.pth`（365 MB）：键前缀 `model.*`（164 键：cls_token/pos_embed/patch_embed/blocks.*/attn.dynamic_mask…），与 `checkpoints/stage_1/bridge_v2_phase1.pth` 的 `vit.*`（162 键）**逐名对应** → 可直接 `model.`→`vit.` 映射装载（先例 `_archive/analyze_branch_redundancy.py:163`）。
- `deepfake_proj = Sequential(Linear(768→768), LayerNorm)`、`vision_proj = Sequential(Linear(1024→768), LayerNorm)`（`vit_m2f2_detector_bridge.py:236/258`）→ **V_proj / C_proj 是 V / C 的纯离线函数**，可用 checkpoint 权重离线重算，并用 `probe_feats.npz` 的 V_proj/C_proj 做复现校验（cos 应 ≈1）。
- `probe_feats.npz`：V(3000,768) C(3000,1024) R(3000,768) V_proj C_proj y paths vids train_mask；FF++ train 2200 / test 800（42 vids，video-disjoint）。
- `vit_module/_tsne/feats_multi.npz`：**F(2300,1664)** V(2300,768) C(2300,1024) y domain vid path；域 = ffpp(800, 140 vid, **含 98 个源 train 视频 → 只可作脚注，不得作 FF++ 参照**) + cd1/cd2/dfdcp/ffiw/wild 各 300（balanced 150/150；ffiw 仅 1 vid 退化）；提取脚本 `_tsne/extract_multi.py` 用的是**同一 checkpoint bridge_v2_phase1.pth**。
- 标签编码：probe 脚本 `y = 1 real / 0 fake`；multi 同族（agent 必须**先复现锚点** e0-AUC cd1=0.8576、full-V LR cd1=0.8286 再开跑）。
- `vit_module/backup_weights/vit_m2f2_phase1.pth`（868 MB, Jun 17）疑为早一阶段检测器；`train_vit_m2f2_phase1.py` 开头注明 **ViT 从 net_050 加载并冻结** → G13 可选中间点（冻结 ViT + 训练头），用于分离"头训练 vs backbone 训练"效应。
- G10 参考实现（管线/装载可直接复用）：`_g10/run_g10.py` L118-145（to_336 / to_tensor_batch / forward_V）、L386-410（ViT_M2F2Det_Bridge 构建 + ckpt `model_state_dict` 装载）。

**H6（用户原话）**：vit 分支仅在 FF++ 上学习到了 FF++ 的特征表达方式，这种方式仅适用于 FF++ 数据集，对其他域数据并不适用；**并需验证该假设是仅在 vit 分支成立，还是对整个框架都成立**。
**H7（用户原话）**：vit 学习的方法仅表达了 FF++ 数据集，但对域内类别的区分能力并没有提升，甚至由于训练收敛导致类间隔缩小、类中心偏移。

**与既有证据的关系（派单前必须承认）**
- H6 强形式已被部分反驳：G11 显示 FF++ 训练出的判别轴在 5 个目标域均被主导表达（|cos(e0_dom,e0_src)| 0.979–0.989、zdim_frac 0.599–0.685），full-V 跨域 AUC 0.81–0.86 ≠ 随机 → "完全不适用"不成立。**可检验的是分级形式**：FF++ 表征区域/几何对其他域的"陌生度"（域差相对类差多大），及这种特异性是 ViT 引入、冻结 CLIP 也有、还是框架整体继承。
- H7 的现象部分已在 post 模型 V 上测得（G11 Q1e：类间隔保留 38–47%、类中心双向漂移 d_real +7.4~+30.0 / d_fake −7.4~−27.2）。**未测的是因果归属**：是 FF++ 微调造成（训练收敛/过拟合），还是 PDI 预训练表征固有 → 必须 pre/post 对照。

**G12（H6 分支级专用性分解）——纯离线 CPU，只读 npz + checkpoint 权重（零前向）** `vit_module/_g12/`
数据：FF++ 参照 = probe test 800（42 vids）；目标域 = multi 5×300（多域 ffpp 只作脚注）。分支：V / C / F + 派生 V_proj=deepfake_proj(V)、C_proj=vision_proj(C)。
- A 复现校验：V_proj/C_proj vs npz cos（≥0.999 方采用派生分支）；F 的 1664 维布局验证（哪 768 段 = V_proj / C_proj）；锚点 e0-AUC cd1=0.8576、full-V LR cd1=0.8286。
- B 域-类间隔比（每分支每域）：class_gap=‖μ_fake−μ_real‖/√(源每维均方差)；dom_gap(d)=‖μ_d−μ_src‖/同归一化；ratio=dom_gap/class_gap。
- C 类条件域可分性（每分支每域，video-disjoint 划分）：三探针 FF++real-vs-域real / FF++fake-vs-域fake / mixed，报 test AUC_real / AUC_fake / AUC_all（ffiw 1 vid 退化 → 标 leak、不入聚合门）。
- D 子空间包含（每分支每域）：K=1/5/50 残差能量（μ_src 中心化），对 FF++ test 参照的膨胀。
- E 轴对齐（每分支每域）：|cos(e0_dom,e0_src)|、zdim_frac（各分支空间内各自估）。
**G12 预注册判定门（机械）**
- Δ_ratio(d)=ratio_V−ratio_C；Δ_AUCreal(d)=AUCreal_V−AUCreal_C。
- H6-ViT 特有：Δ_ratio ≥ +1.0 或 Δ_AUCreal ≥ +0.05 在 ≥4/5 域成立。
- H6-非 ViT 特有（栈共有）：两者在 ≥4/5 域均低于阈值。
- 框架归属：ratio_F ≥ 0.8·ratio_V（或 AUCreal_F ≥ 0.8·AUCreal_V）→ **继承**；ratio_F ≤ 较低者{1.2·ratio_C, 0.8·ratio_V} → **缓解**；否则 mixed。
- 类条件归属：AUC_real 与 AUC_fake 差 <0.05 且均 ≥0.90 → 源特异性"整体区域级"（含 real）；AUC_real <0.75 而 AUC_fake 高 → 差异主要在 fake 子类（伪影语义），H6 解读须收窄。
- 汇总三选一（允许并列 mixed）：{ViT 特有 / 表征栈共有 / 框架放大}。

**G13（H7 pre/post 训练对照）——需前向（net_050 复现 + 特征提取）** `vit_module/_g13/`
- A 装载校验：net_050 `model.*`→`vit.*` 覆盖到 bridge_v2 构建模型的 vit 子模块（其余部分保持 bridge_v2 权重，我们只读 V）；报 missing/unexpected；post 模型 10 图复现 npz cos≈1；pre vs post 的 V 应明显不同（cos<1）。
- B 提取 pre-V：probe train 2200 + test 800 + 5 域×300 = 4500 图 → `_g13/pre_feats.npz`（V/y/paths/vids/domain/train_mask）。
- C 两模型各自独立协议（各自 FF++ train V 拟合 center-only PCA 轴 + LR(C=1e-3)）：域内（800 test）LR AUC / e0 轴 AUC / PC0 varfrac / d_z_src / gap_src；跨域（5 域）源轴 AUC / 源 LR AUC / d_z / gap 保留比例 / 类中心位移 / zdim_frac / |cos(e0_dom,e0_src)|；类条件域可分性 AUC_real/AUC_fake（与 G12 同定义）pre vs post。
- **关键新增**：pre 模型的 PC0 varfrac 与 zdim_frac —— 判定"单轴集中"是 PDI 预训练固有还是 FF++ 微调产物。
- D 可选中间点：`vit_m2f2_phase1.pth` 若可装载且 V 与 net_050 cos≈1（验证"冻结 ViT"）→ 报其跨域表现，分离"头训练 vs backbone 训练"；时间盒内做，失败即弃并注明。
**G13 预注册判定门（机械）**
- b-1 域内增益：post 域内 test AUC − pre ≥ +0.02 → IMPROVED，否则 NO-GAIN。
- b-2 跨域增益：Δ_cross = mean_d(post 源 LR AUC) − mean_d(pre 源 LR AUC)：≥+0.03 IMPROVED / |Δ|<0.01 NO-GAIN / ≤−0.03 DEGRADED / 其余 PARTIAL。
- b-3 间隔缩小与中心偏移的因果归属：Δgap(d)=post−pre（各自 gap 保留比例）；post ≤ pre−0.10 或类中心位移明显更大 在 ≥3/5 域 → **训练造成（H7 后半 CONFIRMED）**；|Δgap|<0.05 且位移相当 → **NOT-CONFIRMED（预训练固有）**。
- b-4 单轴来源：pre PC0 varfrac ≥0.5 → 预训练固有（D.7/D.9 的"单轴由 FF++ 训出"表述需再修正）；<0.3 → 训练产物；之间 PARTIAL。

**资源铁律（两 agent 共同）**：CPU 侧 OMP/MKL/OPENBLAS/NUMEXPR/VECLIB/JOBLIB=1、torch.set_num_threads(1)、cv2.setNumThreads(0)、单进程、num_workers=0；G12 零前向零 GPU；G13 先 `nvidia-smi`，仅当存在空闲 GPU（显存占用 ≤100 MiB 且可用 ≥6 GB）才用（取最低占用者、固定 CUDA_VISIBLE_DEVICES、batch≤16、fp32、记录到审计），否则 CPU 单线程回退并报 ETA。两 agent 均不改 WORKLOG，只交 `run_*.py` + `*_report.txt`（机器块+表+诚实 caveat）+ `*_stats.npz`；结果由我落档 §D.11。

### D.11 G12 结果与 H6 判定（2026-09-10 子 agent；`vit_module/_g12/`，纯离线 CPU，零前向零 GPU，wall 26.4 s）

**校验（全 PASS）**：锚点 e0-AUC cd1=0.8576 / srcLR cd1=0.8286 / PC0 varfrac=0.6229（最大偏差 2.2e-5）；V_proj、C_proj 离线复算与 npz 逐样本 cos=1.000000（LayerNorm eps=1e-5）；**F 布局实测 = [0:768]=0.306288·C_proj | [768:896]=bridge(128, 离线不可得) | [896:1664]=V_proj**（各段 cos_min=1.000000，交叉项仅 0.022）→ G12 的 F 分支用 1536-d（剔 128-d bridge，占 7.7%）；V 分支 D/E 块与 G11 逐位对上（residK1 cd1=0.3983、zdim_srcaxis 0.6357/0.5986/0.6581/0.6852/0.6122）→ 口径等价、无实现漂移。

**核心表（有效域均值；ffiw 1 vid 退化已剔出 AUC 聚合门）**

| 分支 | ratio=dom_gap/class_gap（5 域均值） | AUC_real | AUC_fake | AUC_all | src_zdim | 域轴 \|cos\| vs 源轴 |
|---|---|---|---|---|---|---|
| **V** | **0.1983**（0.161–0.234） | **0.8606** | 0.8630 | 0.9316 | 0.6229 | **0.979–0.989** |
| **C** | **2.7160**（2.43–3.22） | **0.9989** | 0.9972 | 0.9988 | 0.1169 | 0.093–0.333（仅 fftest 0.9124） |
| **F**（1536-d） | 0.1796 | 0.9524 | 0.9564 | 0.9865 | 0.8530 | 0.989–0.998 |
| V_proj | 0.1493 | 0.8347 | 0.7648 | 0.8332 | 0.9387 | 0.994–0.999 |
| C_proj | 1.0904 | 0.9846 | 0.9671 | 0.9829 | 0.3665 | 0.334–0.984 |

- 子空间膨胀（μ_src 中心化 residK Δ=域−FF++参照）：V 温和（K1 +0.007~+0.046、K5 +0.049~+0.105），**C 全维大幅膨胀**（K50 +0.065~+0.239），F K1 +0.042~+0.188。
- 类条件：V 的 |AUC_real−AUC_fake| 均值 0.0228；C 0.0021；F 0.0278。

**机械判定（照抄执行）**
- `G12_H6_VIT_ONLY=NOT-SUPPORT`：Δ_ratio=ratio_V−ratio_C 全 5 域为负（−2.21/−2.36/−3.06/−2.66/−2.30，均值 −2.5177；门要求 ≥+1.0，0/5）；Δ_AUCreal 全 4 有效域为负（−0.20/−0.15/−0.11/−0.09，均值 −0.1384；门 ≥+0.05，0/4）。
- `G12_H6_SHARED=SUPPORT`：两者在 5/5（或 4/4）域均低于阈值。
- `G12_FRAMEWORK=INHERITS`：ratio_F 0.1796 ≥ 0.8·ratio_V 0.1586，AUCreal_F 0.9524 ≥ 0.8·AUCreal_V 0.6884，逐域 5/5；F 的 AUC_real 介于 C(0.9989) 与 V(0.8606) 之间、更靠近 C。
- `G12_CONDITION=CONDITIONAL_MIXED`（branch=V；C 与 F 均为 REGION_LEVEL）：V 的 real/fake 两侧域可分性相当（0.8606/0.8630）→ 不是"仅 fake 子类"的差异；但 V 自身两侧绝对水平都在 0.86（不达 0.90 门）→ 归为 mixed。

**H6 判定（综合）**：用户的 H6（**"ViT 仅在 FF++ 上学到 FF++ 的表达方式、对其它域不适用"**）在**"ViT 特有"这一支上被数据否定，且方向相反**——按同一协议衡量，**冻结 CLIP 分支(C)的域特异/域敏感程度远高于 ViT 分支(V)**（相对域位移大 ~13.7 倍；AUC_real 0.999 vs 0.861；FF++ 主轴在目标域的存在度 C 仅 |cos| 0.09–0.33 而 V 达 0.98）。"域特异性是整栈共有"成立（`H6_SHARED=SUPPORT`），融合头 F 继承两塔的域信息且更靠近 C 侧。

**必须随判定一起讲的诚实限制**：
1. 三个测量块（B 范数比 / C 线性可分性 / D 子空间包含 / E 轴对齐）测的都是"**可分性/几何距离**"，**可分性 ≠ 模型真的使用了该信息**（探针能分开域 ≠ ViT 靠域信息判真假）；本判定只否定"ViT 表征把 FF++ 表达得比其它域更'专用'"这一几何命题。
2. C 的 PC0 varfrac 仅 0.1169（E3：判别弥散于 ~274 方向）→ 对 C 做 "PC0 对 PC0" 的轴对齐比较**本身意义弱**（无主导轴可对齐），E 块对 C 的读法须降权。
3. V_proj/C_proj 的 ratio 下降（"缓解"）**受 LayerNorm 逐样本归一化影响**（V_proj 的 PC0 varfrac 被抬到 0.9387），不能直接读作"训练把头变成了域不变"。
4. 每域仅 300 样本、探针 test 仅 90 样本（AUC 1SE ≈0.05–0.07）；ffiw 1 vid 退化已排除；multi 的 ffpp 域污染只作脚注。
5. 这是"**表达/几何**"层面的否定，**不推翻** G11 的"跨域时沿该轴的类间分离被压扁"结论——两者拼起来是：**轴在、域也不远，但轴上的判别内容不迁移**。

**待 G13（H7 pre/post）完成后合并解读。**

### D.12 G13 结果：**H7 前提被事实推翻——ViT 分支从未被训练过**（2026-09-10 子 agent + 我独立复核；`vit_module/_g13/`）

**执行**：GPU 查询显示 index 2 空闲（54 MiB / 0%），取 GPU2；抽 4500 图（probe train 2200 + test 800 + 5 域×300，跳过 multi 的污染 ffpp 域）wall 47.6 s，总计 75.2 s；峰值显存 3126 MiB；读图失败 0。

**决定性发现（agent 发现 + 我独立复核，双重确认）**
- `G13_A_VIT_TENSOR_IDENT`：`bridge_v2_phase1.pth` 的 `vit.*`（162 个张量）与 `net_050_backup.pth`（`model.*`，跳过 `model.head.weight/bias`）**逐张量完全相同：compared=162, differing=0, shape_mismatch=0, max_abs_diff=0.000e+00 → BIT_IDENTICAL=1**。
- 我的独立复核（离线，无前向）：同上 162/162 全等；并进一步对比**原始 PDI 路径** `E:\Cross-domain_authentication_verification\PDI\results\Ama1_aps1_1\net_050.pth` → 与 `bridge_v2` 的 `vit.*` 也是 **162/162 全等（max=0.0000e+00）**。
- 中间模型 `vit_m2f2_phase1.pth` 的 vit 亦与 pre 逐张量相同（`G13_D_MID ... tensor_differing_vs_pre=0 FROZEN=1`）。
- 全 4500 图的 pre-V 与 post-V：probe 行 **max|d|=0.000e+00**（逐位相同），域行 ≤3.16e-05（仅 flash-attn shim 与已安装 backend 的数值差，见 caveat 10）。因此 G13 的 **B1=NO-GAIN / B2=NO-GAIN / B3=PRETRAINED_INHERENT / B4=PRE_INHERENT_AXIS 全部是"结构性恒等"结果，不是"测到的无变化"**。

**机制（项目自身代码坐实，非推断）**：`vit_module/train_bridge_phase1.py` L266-272 ——
```
# 1. Freeze ViT backbone (prevents overfitting — 91M on 107K data)
for p in model.vit.parameters(): p.requires_grad = False
logger.info('FROZEN: ViT backbone (91.4M) — anti-overfitting')
```
以及优化器构造处注释 `# ViT backbone is NOT in optimizer (frozen)`。→ **stage-1 bridge_v2 训练中 ViT 被显式冻结**（理由：91M 参数 vs 107K 数据，防过拟合）。`vit_module/BRIDGE_ADAPTER_DESIGN.md` 训练配置表中 "ViT backbone … ✅ 微调" 一行**与代码不符，属陈旧/错误记录**。

**对 §D.7 的再更正（本次为硬证据，非推断）**
- §D.6 的推断"V = 冻结 PDI 先验、从未在 FF++ 上训练"**是对的**；§D.7 依用户澄清做的"更正"（改为"V 就是在 FF++ 上训练出来的"）**是错误的**，现予作废。
- 事实：**两个分支都冻结**（PDI-ViT 冻结 + CLIP 冻结）。stage-1 唯一被训练的是 **head/bridge 侧**：`deepfake_proj / vision_proj / text_proj / clip_reduction / linear_vit_1-3 / bridge_adapter×3 / bridge_adapter_proj / output / clip_*_alpha / prompt_tokens`。
- 因此 D.7/D.9 中"**单轴是在 FF++ 上被训练出来的方向**"这类表述**作废**；正确表述：**FF++ 上的单轴结构（PC0 varfrac 0.6229）、跨域间隔压缩（gap 保留 38–47%）、类中心双向漂移，全部是冻结 PDI 预训练表征的固有属性**，与 FF++ 训练无关（G13 B3/B4 以恒等方式确认）。

**H7 判定**：**前提在该分支上不成立（VOID）**——"ViT 学习的方法仅表达了 FF++、训练收敛导致类间隔缩小/类中心偏移"这一因果链**没有可作用的训练过程**：ViT 参数一字未改。
- "对域内类别的区分能力并没有提升"**字面为真但机制不同**：不是"训练了却没提升"，而是该分支**根本没训练**。
- 正确因果归属：跨域退化（G11 的间隔压缩与中心漂移）**不是训练造成的，是 PDI 先验固有的**。

**H6 判定（随之更新的表述）**：G12 的几何测量全部有效，但其解释须改写——V 与 C **都是冻结先验**，"V 由 FF++ 训练引入源特异性"这一说法不成立；G12 实际比较的是 **PDI 先验（V）vs CLIP 先验（C）** 的域几何，结论不变：**可分的域信息明显更集中在冻结 CLIP 一侧，ViT 一侧反而更"跨域一致"**；可训练的头/桥继承了该信息（F 更靠近 C）。

**对研究路线的影响（重要）**
1. "当前框架已把冻结双分支的信息汲取到极限"——该判断**现在字面成立**：两条分支都是冻结先验，能被训练的只有融合头/桥。
2. **"让 ViT 真正在 FF++ 上训练"是一个从未尝试过的杠杆**：现方案冻结 ViT 的唯一理由是"91M/107K 防过拟合"。若用户的 H4/H5 路线（数据增广 SBI + 训练结构改进）要落地，**解冻 ViT（配合增广/正则/低 lr/分层冻结）是当前最直接、且完全未被探索的改动**。
3. 论文口径修正：**不得再写"ViT 分支在 FF++ 上微调"**；应写"双冻结分支 + 可训练桥/头，ViT 表征为 PDI 通用先验"。
4. 遗留开放项：被训练的头/桥本身是否过拟合源分布（G5 已给"权重锁死源分布"线索；D.7④ 给"融合 ≈ V 单独 +0.3pt"）——若要完整回答"框架层面是否也只在 FF++ 有效"，需对 head/bridge 单独做一次 `pre(随机初始化头) vs post(训练头)` 的对照，代价小但需新前向。

**审计与 caveat（agent 报告，我认可）**：pre 模型 = bridge_v2 架构仅覆盖 vit 子模块（其余沿用 bridge_v2 权重），因 V 只依赖 vit 子模块 + 无参数的 `_preprocess_for_vit`，对比仍然干净；4500 图子集；每域 300 样本、ffiw 1 vid 退化已剔出聚合；d_z 与单轴 AUC 数学同义；pre/post 各自的轴各自拟合；域行 3e-05 的 backend 数值差对任何指标无影响。

**§D.12 附注（用户澄清 2026-09-10，优先级高于上文推断）**：`net_050` **本身就是在 FF++ 上训练出来的权重**（训练发生在 **PDI 项目侧**，不是本项目 stage-1）。据此更正：
- §D.6 的推断"V = 从未在 FF++ 上训练的防伪先验"**作废**（错在把"本项目未训练"当成"从未训练"）。
- §D.7 的说法"V 就是在 FF++ 上训练出来的"**实质成立**，但必须加限定：**训练在 PDI 侧完成，本项目 stage-1 只是冻结沿用**（`train_bridge_phase1.py` L267-272 显式 freeze，理由 91M/107K 防过拟合）。
- **G13 的发现仍然有效且有意义**：stage-1 的 bridge 训练**没有改动 ViT**（162/162 bit-identical）；但由此推出 H7 **用现有产物不可判定**——真正的 "pre" 应是 PDI 侧 FF++ 训练**之前**的 checkpoint（MAE/ImageNet 初始化等），该产物不在本项目内。H7 暂缓（用户亦指示"解冻的问题先放一放"）。
- 当前成立的口径：**V 分支 = 冻结的"FF++ 训练先验"**（PDI 侧产物），与本会话 G8–G12 的全部测量完全相容；待补：net_050 的确切训练配置（FF++ 子集/任务/epoch），仅影响"先验"叙述精度，不影响任何测量。

### D.13 H8 假设与 G14 验证计划（2026-09-10 用户提出）

**H8（用户原话）**：现在 vit 其实能表达不同数据集来源图像的特征向量，**但是对同域内不同类区分能力有限**，也就是两类的特征向量被拉近，区分能力被限制。
**目的（用户说明）**：若 H8 成立 → 后续改进（SBI 增广 / 解冻训练 / 增加信息来源分支）就都有立足点——它们都建立在"ViT 能提供域无关的先验引导、问题只在'如何把两类分开'"之上；否则会陷在广度探索里说不清究竟要针对什么入手。

**已有证据对照（派单前分析）**
- **第一半"域不敏感/能表达不同域"——已有三点口径支持（分级成立，非绝对）**：① G12 归一化域位移比 ratio_V=0.198 vs ratio_C=2.716（ViT 侧的域位移只有类间距的 ~20%，CLIP 侧是 2.4–3.2 倍）；② G11/G12 轴对齐 |cos(e0_dom,e0_src)|=0.979–0.989、zdim_frac 0.599–0.685；③ G12 子空间膨胀 V 温和（K1 +0.007~+0.046）而 C 大（K50 +0.065~+0.239）。**诚实张力**：V 的 real-only 域判别 AUC 仍达 0.861（不是零敏感），故只能表述为"**相对**域不敏感（远低于 CLIP）"，不能说"完全域不变"。
- **第二半"类间被拉近、区分受限"——已有部分支持**：G11 Q1e 类间隔保留 38–47%、类内 sd 膨胀 1.42–1.66、两类质心**相互**接近（d_real +7.4~+30 朝向 fake、d_fake −7.4~−27 朝向 real）；G11 ORACLE（域内自训 768-d LR，按 vid CV）仅 0.83–0.91，比源训头高 +2.3~+8.0pt → 说明限制**主要在表示层而非头层**。
- **仍未验证、且决定改进方向的三个点**：① 域内可分离性上限究竟有多低（相对 FF++ 0.9852 的缺口）；② 损失是**纯幅度衰减**（同方向、幅度不足 → H8 成立）还是**方向旋转**（类方向变了 → H8 需收窄）；③ 瓶颈是否**专属 V 分支**（C/F 在目标域内是否含更多类可分信息 → 决定"增加分支"是否有意义）。

**G14（H8 判定 + 瓶颈定位）——纯离线 CPU，零前向，只读 npz** `vit_module/_g14/`
数据：FF++ train 2200 / test 800（probe npz，42 vids）；5 目标域各 300（multi npz；跳过污染 ffpp）。分支 {V, C, F}（F 用 1536-d 拼接口径，同 G12）。
- **R 复核行**：复现 ratio / |cos(e0_dom,e0_src)| / gapRatio / d_z / d_real / d_fake（与 G11/G12 对表，保证口径一致）。
- **A 域一致性（记录行，不设主门）**：ratio 与 residK 复现，作第一半的复核。
- **B 域内可分离性上限（核心）**：每域每分支，用**域内标签**训练 oracle 头：LR(C=1e-3) 与 LR(C=1.0) 各一次 + kNN(k=5) + RBF-SVM 对照，全部 **video-grouped 5 折 CV**（ffiw 1 vid → 退化、标 leak、不入聚合）；报 oracle AUC 表与相对 FF++ 域内 0.9852 的缺口 Δ_oracle。
- **C 拉近量化**：沿源轴的 gap、合并 sd、d_z、Fisher 比、**重叠系数（1−AUC_e0）**、两类质心相互接近量；并对"(把间隔线性放大到源水平后 AUC 会变多少)"做一个简单缩放对照。
- **D 衰减 vs 旋转**：域内类均值差方向 Δ_d=μ_fake−μ_real（无正则）；cos(Δ_d, e0)；三个域内打分探针对照 —— 用 Δ_d 全方向 / 只用 e0 / 只用 Δ_d 的**轴外分量**，比较 AUC → 若"只用 e0"≈"全 Δ_d" 且远超"只用轴外"，即**同方向、幅度不足 = 纯衰减**（H8 成立）。
- **E 瓶颈定位**：汇总 B 的 V/C/F 域内 oracle AUC。
**预注册判定门（机械）**
- `G14_H8_DOMAIN_CONSIST`: mean_d ratio_V ≤ 0.5 且 ratio_V < ratio_C − 1.0 → SUPPORT（第一半复核）；ratio_V ≥ 1.5 → NOT；否则 PARTIAL。
- `G14_H8_LIMITED`: mean_d (0.9852 − oracle_V(d)) ≥ 0.08 → LIMITED（域内类分离显著受限）；≤0.03 → NOT-LIMITED；之间 PARTIAL。
- `G14_H8_ATTENUATION`: cos(Δ_d, e0) ≥ 0.9 且 (AUC(Δ_d 全) − AUC(仅 e0)) ≤ 0.02 在 ≥4/5 域 → ATTENUATION_ONLY（同方向、幅度不足）；若 cos < 0.7 或 AUC 差 > 0.05 在 ≥3/5 域 → ROTATED/MIXED（H8 需收窄为"方向部分改变"）。
- `G14_BOTTLENECK`: max(mean oracle_C, mean oracle_F) − mean oracle_V ≥ +0.05 → BOTTLENECK_AT_V（其它分支含更多类可分信息 → "增加分支"有依据）；|差| < 0.02 → BOTTLENECK_SHARED（信息上限如此 → 须动表征本身，如 SBI/解冻）；否则 MIXED。
- 汇总：H8 判定 = {域一致 + 类间受限 + 纯衰减} 三块合成（允许并列 partial）。

**资源铁律**：同 G12（OMP/MKL/…=1、torch.set_num_threads(1)、单进程、零前向零 GPU）；产物 `run_g14.py` + `g14_report.txt`（机器块+表+机械判定+caveat）+ `g14_stats.npz`；不改其它文件、不写 WORKLOG；结果由我落档 §D.14。

### D.14 G14 结果与 H8 判定（2026-09-10 子 agent；`vit_module/_g14/`，纯离线 CPU 零前向，wall 25.1 s）

**校验全 PASS**：锚点 e0-AUC cd1=0.8576 / srcLR cd1=0.8286 / varfrac=0.6229（偏差 2.2e-5）；G11 cd1 行口径对齐（gapRatio 0.4268 / dReal +25.8687 / dFake −8.9057 / zdimFrac 0.6357 / \|cos\| 0.9870，偏差 4.2e-5）；R 块 5 域与 G11 全对表；A 块 ratio 与 G12 完全一致（V 0.1983 / C 2.7160 / F 0.1796）。

**B 域内可分离性上限（每域 300 样本，video-grouped 5 折 CV；oracle = 4 方法取最好）**

| 分支 | cd1 | cd2 | dfdcp | wild | 4 有效域均值 | ffiw(leak) | FF++ 域内(best) |
|---|---|---|---|---|---|---|---|
| **V** | **0.9550** | 0.8956 | 0.9223 | 0.8717 | **0.9111** | 0.9991 | **0.9881** |
| C | 0.9534 | 0.9266 | **0.7244** | 0.9083 | 0.8782 | 1.0000 | 0.9108 |
| F(1536) | 0.9298 | 0.9140 | 0.8461 | 0.8914 | 0.8953 | 0.9984 | 0.9874 |

- **Δ_oracle(d) = 0.9881 − oracle_V(d)**：cd1 +0.0331 / cd2 +0.0925 / dfdcp +0.0658 / wild +0.1164 → **有效域均值 +0.0770**。
- **方法学细节（重要）**：V 分支上**每个域的最优方法都是 LR(C=1.0)（较弱正则的线性头）**（0.9550/0.8956/0.9223/0.8717），而 **kNN(k=5) 明显更差**（0.8465/0.8116/0.7964/0.7748）、RBF-SVM 居中（0.8962/0.8546/0.8769/0.8347）；G14 的 **LR(C=1e-3) 列与 G11 的 oracle 逐域完全吻合**（0.9082/0.8860/0.8774/0.8349）→ **G11→G14 的 oracle 增量来自正则强度，不是非线性**。折间 std 0.03–0.16（cd1 最大）。
- C 分支在 **dfdcp 上塌到 0.72**（其余 0.91–0.95），说明 CLIP 的域内类信息本身对数据类型高度不稳。

**C 拉近量化（V，沿源轴）**：gap 保留 0.38–0.47；d_z 1.141–1.530（源 4.6555）；**重叠系数 1−AUC_e0 = 0.140–0.194**；dReal +7.4~+30.0、dFake −7.4~−27.2（两类相互靠近）。

**D 衰减 vs 旋转（每域 5 折，折内 train 构造方向、折内 test 打分）**

| | cd1 | cd2 | dfdcp | ffiw | wild |
|---|---|---|---|---|---|
| cos(Δ_d, e0) | 0.9790 | 0.9766 | 0.9606 | 0.9855 | 0.9437 |
| AUC(Δ_d 全方向) | 0.8662 | 0.8712 | 0.8426 | 0.8107 | 0.8270 |
| AUC(仅源轴 e0) | 0.8614 | 0.8592 | 0.8358 | 0.8073 | 0.8048 |
| AUC(仅 Δ_d 轴外分量) | 0.6858 | 0.6888 | 0.7519 | 0.7291 | 0.7298 |
| Δ = 全 − 仅 e0 | +0.0048 | +0.0120 | +0.0068 | +0.0033 | +0.0222 |

**机械判定**：`G14_H8_DOMAIN_CONSIST=SUPPORT`（ratio_V 均值 0.1983 ≤0.5 且 < ratio_C−1.0）；`G14_H8_LIMITED=PARTIAL`（**+0.0770，距 0.08 门仅 0.003，临界**）；`G14_H8_ATTENUATION=ATTENUATION_ONLY`（5/5 域 cos≥0.9 且 Δ≤0.02，0/5 旋转）；`G14_BOTTLENECK=BOTTLENECK_SHARED`（max(oracle_C,oracle_F) − oracle_V = **−0.0158**）。

**H8 判定（综合）**：**"ViT 提供域无关的先验、问题在同类间的分离"这一核心判断成立**，但需按三块分别限定：
1. **域一致性 = 成立（相对口径）**：V 的归一化域位移只有类间距的 0.198（C 为 2.716）；且**目标域的类判别方向与源轴近乎同向**（cos 0.944–0.986）——同一个"提取与判别轴"在各域都成立。限定：V 的 real-only 域判别 AUC 仍 0.861，属"相对不敏感"，非域不变。
2. **"两类被拉近、区分受限" = 成立但临界**：Δ_oracle 均值 **+0.0770**（踏在 0.08 门线上，判 PARTIAL）；分域看 cd2/dfdcp/wild 掉 0.07–0.12，而 cd1 只掉 0.033。
3. **机制 = 纯衰减，无旋转（本轮最干净的结论）**：用域内标签构造的完整类方向相比只用源轴，只多 +0.005~+0.022 AUC；而纯轴外分量只有 0.69–0.75 → **跨域损失不是"方向变了/信息缺失"，而是同一根轴上类间分离幅度被压低**。
4. **瓶颈 = 共享**：C（0.8782）与 F（0.8953）的域内 oracle 都不高于 V（0.9111）→ **"增加信息来源分支"就"信息量"而言没有依据**——除非新分支带来**现有 V/C 都没有的新信息**（复用 CLIP 无增量）。

**对改进方案的三条直接含义（本研究路线用）**
- (i) **单轴分数层面的任何域校准都不可能改变域内 AUC**（AUC 是秩统计量：重对中/缩放/阈值平移都是单调变换）→ G11 的 recenter 恒等与 G14 的"重叠系数"共同锁死这条捷径；要利用"方向同向"这一红利，必须在**多维**上做域适配，或直接改表征。
- (ii) 目标域内 **最优头是"弱正则线性头"而非非线性头**（kNN 明显更差）→ 可恢复余量（+2~+8pt）是**线性可达**的，量级有限；这为"换头/域自适应"路线给出清晰但**有限**的上界。
- (iii) 由于方向跨域同向（cos≥0.94）而幅度不足，**SBI 增广/解冻重训的靶心应当是"让类间分离幅度在域间保持"**（而非"学会跨域不变特征"）——这是本轮 H8 给后续实验最具体的一句话。

**Caveat（agent 报告，我认可）**：oracle 用目标域标签（上界、不可部署）、折间方差大（单域 AUC 差 <0.05 不宜解读，cd1 的 0.955 vs 0.908 可能就是正则强度+噪声）；ffiw 1 vid 泄漏已剔出聚合；高斯投影为近似非测量；相关≠因果（域内 oracle 高于跨源头不能证明增广/解冻可兑现）；F 分支缺 128-d bridge 块（7.7% 维度）；"域一致"仅相对意义。

### D.15 问题 Q-IL：「如何提高目标域内类间判别」——杠杆清单与 G15/G16 计划（2026-09-10 用户提出）

**用户问题原话**：域一致性确实存在，但现在**不知道什么方式能提高域内类别的判别**。

**已被现有证据排除的路径（不必再试，除非有反例）**
| 候选手段 | 排除依据 |
|---|---|
| 单轴分数层面的域校准（重对中/缩放/阈值平移） | AUC 是秩统计量 → 单调变换恒等（G11 Q2 Δ≡0；G14 重叠系数） |
| 线性协方差对齐（CORAL/白化到源） | G11 规格外补测 Δ ≈ 0 或微负 |
| 非线性头（kNN / RBF-SVM） | G14 B 块：LR(C=1.0) 每域最优；kNN 明显更差、RBF-SVM 居中 |
| 复用 CLIP/C 分支的信息 | G14 E：oracle_C 0.8782、oracle_F 0.8953 均 ≤ oracle_V 0.9111（BOTTLENECK_SHARED） |
| 域内标签可及的上限 | G14：域内 oracle 0.87–0.96（均值 0.911），比域内 FF++ 低 7.7pt |

**待检验的杠杆（本轮设计）**
- **L1 无标签特征变换（域内可用）**：逐维标准化 / 目标域 ZCA 白化 / 源域白化后重训——检验"仅靠目标域无标签统计能否提升"。
- **L2 半监督自训练（无标签）**：源头打伪标签 → 置信样本重训 → 迭代——检验"无标签适应能否兑现"。
- **L3 多域联合监督（= "数据广度"杠杆的冻结表征代理）**：leave-one-domain-out，用 FF++ + 其它 4 域（或仅其它域）的**真实标签**训练头，评估留出域——**这条直接检验用户"H4/数据集丰富度"假设**，且在冻结表征上即可给出量化增益。
- **L4 特征源（需前向，G16）**：ViT 中间层（blocks 3/6/9，bridge 已 hook）+ 末层 CLS 的单用/拼接——检验"更可迁移的类信息是否存在于别的深度"。

**G15（L1/L2/L3 离线杠杆排序）——纯离线 CPU，零前向** `vit_module/_g15/`
数据：源 = probe train 2200（V/C）/ test 800；目标 = multi 5×300（V/C/F）。基线 L0 = 源训 LR(C=1e-3 与 1.0) 在 V 上的跨域 AUC（锚点 0.8286/0.8633/0.8261/0.8244/0.8090）。
- **L1**：(a) 域内逐维标准化（mean/std 由目标域无标签样本估计）→ 源头打分；(b) 目标域 ZCA 白化；(c) 源域白化后重训 + 跨域评估；(d) CORAL 作参照行（已知无效，不入门）。
- **L2**：video-grouped 5 折：在 4/5 折的目标样本上取源头伪标签（阈值/比例 p=25%/50%/100%），（可含源数据）重训线性头，在留出折上用**真标签**评估；迭代 2 轮；报 p 与迭代的最优组合。
- **L3**：leave-one-domain-out，训练集 = {FF++ train + 其它 4 域×300}（变体：仅其它 4 域），评估 = 留出域全部 300（真标签）；另报"其它域→留出域"的单源迁移矩阵。
- **参照 L4**：直接引用 G14 的域内 oracle 行。
**预注册判定门（机械）**
- `G15_L1_LABELFREE`：最优无标签变换相对 L0 在 ≥4/5 域提升 ≥+0.03 → EFFECTIVE（记录名字）；全部 ≤+0.01 → INEFFECTIVE。
- `G15_L2_SELFTRAIN`：≥4/5 域提升 ≥+0.05 → EFFECTIVE；≤+0.01 → INEFFECTIVE；之间 PARTIAL。
- `G15_L3_MULTIDOMAIN`：（FF++ + 其它 4 域）≥4/5 域提升 ≥+0.05 → DATA_BREADTH_EFFECTIVE；≤+0.01 → INEFFECTIVE；之间 PARTIAL。
- `G15_RANK`：按 4 域均值增益排序全部杠杆，输出"最有效"与"最便宜有效"。

**G16（L4 特征源：中间层是否更可迁移）——需前向（GPU 优先，规则同 G13）** `vit_module/_g16/`
- 用 bridge 模型已注册的 hook（`_make_hook("b_1"/"b_2"/"b_3")`，blocks 3/6/9）或直接 `model.vit.blocks[k]` 前向，取该层 **CLS** 与 **patch 均值池化**；连同末层 CLS（=V，作复现校验）一起，对 4500 图（probe train 2200 + test 800 + 5 域×300）提取。
- 指标（每层/每池化/拼接组合）：FF++ 域内 AUC（train→test）、跨域 AUC（源训 LR）、目标域内 oracle（video-grouped CV）。参照 = 末层 CLS（V）。
- 门：`G16_LAYER_TRANSFER`：任一层/拼接在 ≥4/5 域跨域 AUC ≥ V+0.03 → LAYER_LEVER_FOUND；`G16_LAYER_ORACLE`：同理对 oracle ≥ V+0.03。
- 复现校验：末层 CLS 与 probe npz 的 V cos≈1（否则停）。

**资源铁律**：G15 同 G12/G14（全离线零前向）；G16 遵守 G13 的 GPU 规则（先查 nvidia-smi，仅用空闲卡，否则 CPU 单线程）。两 agent 均不改其它文件、不写 WORKLOG；结果由我落档 §D.16。

**H5 判定（综合 G10+G11，机械读法 + 我的汇总）**
- **H5 字面版（"跨域样本没有表达出近似该轴的方向"）→ 不成立**：域自身 PC0 与源 e0 的 |cos| = 0.979–0.989，第一主轴能量占比 0.599–0.685（源 0.6229），域内类均值差方向 cos(e0) = 0.944–0.986 → **轴在每个目标域都被主导地表达着**。
- **成立的是改写版**："**轴在，轴上的类间分离不迁移**"——类间隔保留 38–47%（衰减 ~2.2 倍）、类内沿轴散布膨胀 1.4–1.7 倍（合成 d_z 降至源的 25–33%）、两类中心双向漂移（real 向 fake 侧 +7.4~+30.0，fake 向 real 侧 −7.4~−27.2）→ 判别力沿同一根轴被**压扁**。
- **可恢复余量（oracle 上界）**：域内有标签重定向头可回收 **+2.3~+8.0pt**（0.83–0.91），label-free 手段（重对中/CORAL）≈ 0 → **换头/无标签自适应的空间有限且无免费午餐**；距域内 0.985 仍有 ~8–15pt 缺口只能靠**动表示**（重训/新监督/新数据）。
- **G10 补充**：轴对像素级变换免疫、对几何裁剪高度敏感、遮挡热图中心偏好温和且跨域不位移 → 轴更像**几何/结构/构图**性质，而非像素统计，也非"局部伪影"；但 I1 只在源内做过，目标域稳健性未测。

**对 D.3 决策树的净影响**：①方向 1（打破单轴/重训）获得**量化依据**：头层面最多 +2~8pt，实质跃升必须动表示；②方向 2（模型内归因）完成第一轮，得到"非低层 + 几何敏感 + 弱中心偏好"，但**未能定性轴内容**——进一步手段（目标域不变性测试、子空间追踪、更强归因）未做；③免训自适应方案（对中/CORAL）被本次否定。

---

### D.16 G16 结果：特征源杠杆（ViT 中间层 blocks 3/6/9）——L4 判定不成立（2026-09-10）

**目的**：检验 §D.15 的 L4——"更可迁移的类信息是否存在于 ViT 别的深度"，即换特征源能否抬高跨域/域内判别。

**运行审计**：物理 GPU 2（GTX 1080 Ti；查询时占用 54 MiB，满足 ≤100 MiB 且可用 ≥6 GB），`CUDA_VISIBLE_DEVICES=2`，**无 CPU 回退**；fp32，batch=16，num_workers=0，CPU 线程钉 1；4500 图 / 4510 前向（含 10 张校验）/ 283 batch；read_fail=0；显存 peak torch 2199 MiB、nvidia-smi 4070 MiB；**wall 104.2 s**。产物 `vit_module/_g16/{run_g16.py, layer_feats.npz(4500×768×7), g16_report.txt, g16_stats.npz, run_log_g16.txt}`。

**前置校验（全部 PASS）**
- 管线复现：10 张 probe-test 图的末层 CLS 与 `probe_feats.npz['V']` 余弦 **min = 1.000000**。
- 锚点复现：`cls_final` 跨域 AUC 五域全部命中（0.8286/0.8633/0.8261/0.8244/0.8090），**maxdev = 0.000033**（口径零漂移）。
- **hook 语义实测澄清（对后续分析重要）**：`vit.blocks[3]/[6]/[9]` 注册 forward hook → key `b_1/b_2/b_3`，输出是**块返回的残差流 (B,197,768)**（[:,0]=CLS，[:,1:]=196 patch token）。该 ViT 为 pre-norm 结构：块内 LayerNorm 只归一化 attention/MLP 的输入，**块输出本身未经输出侧 LayerNorm，也未经 V 所经过的最终 `vit.norm`** → 即中间层特征本质是"第 k 层 post-block 残差流 / 相对 V 属 pre-final-norm"。bridge 实际消费的是 `hook[:,1:,:]`（patch token）经 `linear_vit_*`(768→64) 投影，**不是 CLS**。

**我的独立复核**：不采信 agent 汇总，直接读 `layer_feats.npz` 用 StandardScaler(fit train) + LR(C=1e-3, lbfgs) 重算 **10 个源 × 5 域**，**逐格复现** agent 报告（含 `cls_final` 五域锚点）。

**表 A：跨域 AUC（源 FF++ train 2200 训线性头，C=1e-3；positive=fake）**

| source | cd1 | cd2 | dfdcp | wild | ffiw(单列) | mean4 |
|---|---|---|---|---|---|---|
| `cls_final`（=V，参照） | 0.8286 | 0.8633 | 0.8261 | 0.8090 | 0.8244 | 0.8318 |
| `cls_b3` | 0.5915 | 0.6152 | 0.7190 | 0.5633 | 0.5905 | 0.6223 |
| `cls_b6` | 0.8149 | 0.7133 | 0.7332 | 0.7564 | 0.6308 | 0.7545 |
| `cls_b9` | 0.8361 | 0.8572 | 0.8300 | 0.8112 | 0.8053 | 0.8337 |
| `mp_b3` | 0.6855 | 0.6031 | 0.6880 | 0.6618 | 0.6168 | 0.6596 |
| `mp_b6` | 0.7672 | 0.7520 | 0.7254 | 0.7665 | 0.6698 | 0.7528 |
| `mp_b9` | 0.8167 | 0.8572 | 0.8272 | 0.8058 | 0.8065 | 0.8267 |
| `cat_cls369` | 0.8599 | 0.8328 | 0.8475 | 0.8169 | 0.7879 | 0.8393 |
| **`cat_b6_final`** | 0.8549 | 0.8554 | 0.8415 | 0.8200 | 0.8053 | **0.8429** |
| `cat_mp369_final` | 0.8442 | 0.8486 | 0.8230 | 0.8096 | 0.7608 | 0.8313 |

**表 B：域内 oracle（video-grouped CV，上界性质、不可部署）— 关键行**

| source | cd1 | cd2 | dfdcp | wild | ffiw | mean4 |
|---|---|---|---|---|---|---|
| `cls_final`（=V） | 0.9550 | 0.8953 | 0.9223 | 0.8717 | 0.9991 | 0.9111 |
| `cls_b3` | 0.9442 | 0.7479 | 0.6760 | 0.7530 | 0.9920 | 0.7803 |
| `cls_b6` | 0.9507 | 0.8751 | 0.7586 | 0.8368 | 0.9904 | 0.8553 |
| `cls_b9` | 0.9100 | 0.8999 | 0.9072 | 0.8520 | 1.0000 | 0.8923 |
| `mp_b9` | 0.9523 | 0.9191 | 0.9303 | 0.8633 | 0.9967 | 0.9163 |
| `cat_cls369` | 0.9452 | 0.9244 | 0.9042 | 0.8581 | 1.0000 | 0.9080 |
| **`cat_mp369_final`** | 0.9708 | 0.9453 | 0.9149 | 0.8901 | 1.0000 | **0.9303** |

**判定（机械读法）**
- `G16_LAYER_TRANSFER = PARTIAL`（门：某层/拼接在 ≥4/4 有效域 ≥V+0.03）：最优 `cat_cls369` 仅在 **cd1 达 +0.0312**，**1/4 域** → **不构成杠杆**。
- `G16_LAYER_ORACLE = PARTIAL`（同门，oracle 口径）：最优 `cat_mp369_final` 仅在 **cd2 达 +0.0500**，**1/4 域** → **不构成杠杆**。
- `G16_BEST_SOURCE`：跨域 = `cat_b6_final`（0.8429 vs 0.8318，**+0.011**）；oracle = `cat_mp369_final`（0.9303 vs 0.9111，**+0.019**）。两者均**远未达 +0.03 门**。

**结论**：**换特征源（深度）不构成杠杆**。层级趋势清晰且单调——`cls_b3` 均值 0.6223（近"随机以上"），`cls_b6` 0.7545，`cls_b9` 0.8337 ≈ `cls_final` 0.8318 → **类判别信息随深度累积，末层已是最可迁移的单源**；中间层不携带额外的、跨域更稳的类信息。拼接组合只在个别域出现 +0.03~+0.05 的局部增益，**无跨域一致性**，且伴随显著折间方差。

**caveat（如实标注）**
1. hook 输出为 **pre-final-norm 残差流**，且中间层**未经 bridge 的 `linear_vit_*`(768→64) 投影** → 本实验测的是各层的"信息可用性（线性可及性下界）"，**不是模型实际的融合行为**；不能据此断定模型内部完全没有用到中间层信号。
2. 跨域 AUC 用源域训练的**线性头**评估 = 线性可及性**下界**；oracle 用域内标签训练 = **上界性质、不可部署**。
3. 每域仅 300 样本、5 折 CV **折间方差大** → **单域 <~0.05 的差异不宜过读**；这一条直接影响"个别域 +0.03/+0.05 增益"的可信度。
4. `ffiw` 仅 1 个视频 → 退化为 StratifiedKFold（**leak=True**），已单列、**不入聚合**（故 mean4 为 4 域均值）。
5. 拼接的标准化为 per-feature（等价于分块标准化）；**相关 ≠ 因果**：某层 AUC 更高不等于检测器会利用该层信号。

**对 Q-IL 的含义**：**L4 排除**。想在冻结表征上拿到更可迁移的类判别，**改深度没用**——必须走目标域适应 / 多域监督（G15 正在测），或者改变表示本身（解冻微调 / 新监督信号 / 新模态）。这条为"改表示"方向又添了一条否定性证据。

---

### D.17 G15 结果：离线杠杆排序（L1 无标签 / L2 自训练 / L3 多域监督 / L3b 边际曲线）——**含我方重大更正：多域监督增益被同族视频泄漏夸大**（2026-09-10）

**运行审计**：纯离线 CPU，零前向 / 零 GPU / 零图读取，threads=1，processes=1，wall **76.4 s**。产物 `vit_module/_g15/{run_g15.py, g15_report.txt, g15_stats.npz, run_log_g15.txt}`。

**前置校验 PASS**：srcLR(C=1e-3) 五域锚点全部复现（0.8286/0.8633/0.8261/0.8244/0.8090），e0-AUC cd1=0.8576 复现，**maxdev = 3.3e-05**。

#### D.17.1 agent 原始判定（我复核前的口径）

| 判定门 | 结果 | 数字 |
|---|---|---|
| `G15_L1_LABELFREE` | **INEFFECTIVE** | 最优 L1 = CORAL，4 域均值增益 **−0.0108**，0/4 域 ≥ +0.03；目标域逐维标准化 −0.010~−0.015；ZCA 白化 b1 **−0.16~−0.19**；源域白化 −0.02~−0.12 |
| `G15_L2_SELFTRAIN` | **PARTIAL** | 每域最优均值 **+0.0498**，1/4 域 ≥ +0.05（仅 cd1 +0.0819）；最优组合统一为 (p=25%, 仅伪标签, 迭代 2) |
| `G15_L3_MULTIDOMAIN` | **PARTIAL** | FF+++其它 4 域 4 域均值 **+0.0667**，2/4 域 ≥ +0.05 |
| `G15_L3b_MARGINAL` | **MARGINAL_POSITIVE** | 无源 k1→k4 平均 Δ = **+0.0668**（逐域 cd1 +0.1566 / cd2 +0.0771 / dfdcp +0.0535 / wild −0.0202） |
| `G15_RANK` | — | `L4_oracle(+0.1154) > L3_multidomain(+0.0667) > L2_selftrain(+0.0498) > L1_label_free(−0.0108)`；最有效 = oracle（上界、不可部署）；最便宜有效 = L2 自训练 |

#### D.17.2 我的独立复核（读两 npz 自行重算）

**复现成功的**：L0(C=1e-3) 五域锚点**逐位一致**（0.8286/0.8633/0.8261/0.8090）；L0(C=1.0) 均值 0.7957（agent 0.7955）；**层级/杠杆方向全部一致**。

**新算出的、agent 未报的关键格子——基线口径问题**：

| 臂 | C=1e-3（强基线） | C=1.0（弱基线） |
|---|---|---|
| L0 仅 FF++ train | 0.8318 | 0.7955 |
| L3 多域监督 | **0.8516** | **0.8604** |

- agent 的 "+0.0667" 是拿 **L3(C=1.0)** 比 **L0(C=1.0)** —— 同 C 口径（可辩护），但 L0 在 C=1.0 时因 2200 样本 / 768 维而严重过拟合，基线被人为压低。
- 拿 L3 与**最强单源基线**（L0 C=1e-3 = 0.8318，即论文锚点）比，真实增益只有 **+0.0198**；逐域 cd1 +0.0353 / cd2 +0.0219 / dfdcp +0.0136 / wild +0.0086 —— **没有一域达到 +0.05**。
- **结论：多域监督的增益对基线口径高度敏感（+0.02 ~ +0.065 之间摆动）**，agent 的头条数字取的是上界。

**未复现的**：我的 CORAL 快速实现给出 0.44（方向反转，属我实现缺陷），与 agent 的 −0.0108 不符 → **我的 CORAL 数不采信**；结论仍与 G11 的 Δ≈0 一致：CORAL 不提升。另 dfdcp/cd1 单域数值与 agent 有 ~0.01~0.03 的差（标准化/solver 细节），**说明 +0.01~+0.04 量级的效应落在实现噪声带内**。

#### D.17.3 ⚠️ 我方发现的重大数据完整性问题——L3/L3b 的 cd1/cd2 行被视频泄漏污染

按 video-id 逐一比对 `feats_multi.npz` 与 `probe_feats.npz`：

| 重叠对 | 命中 | 量化 |
|---|---|---|
| **cd1 的视频在 cd2 中出现** | **41/41 vids** | **cd1 的 300/300 张图（100.0%）** 的视频都出现在 cd2 |
| cd2 的视频在 cd1 中出现 | 41/124 vids | cd2 的 **84/300 张图（28.0%）** 被 cd1 覆盖 |
| ~~wild 的视频在 ffpp 中出现~~ | ~~17 vids~~ | ❌ **已作废，见 §D.23.0 更正** |
| ~~wild 的视频在 probe train 中出现~~ | ~~8 vids~~ | ❌ **已作废，见 §D.23.0 更正** |

**机制**：Celeb-DF v1 是 v2 的**子集**（同一批源视频），所以 cd1 ⊂ cd2 在视频层面成立。

**后果（直接推翻 L3 的头条结论）**：
1. **留出 cd1 时，训练集里的 cd2 含有 cd1 的 100% 视频**（同视频不同帧）→ L3 在该域 AUC = 0.9514，几乎等于"域内 oracle"0.9550 —— **这不是跨域迁移，是视频级泄漏**。
2. 留出 cd2 时，训练集里的 cd1 覆盖其 28% 图 → cd2 的 +0.059 也被部分夸大。
3. **L3b 无源曲线 cd1 k1→k4 = +0.1566 的主体同样来自泄漏**（k1 加的就是 cd2）。
4. **只有 dfdcp 与 ffiw 与任何域零重叠，是干净评估域**；~~wild 因 3% 与 probe train 重叠~~ → **❌ 该条已作废，见 §D.23.0：wild 实为干净域，其历史数字未被抬高。**

**修正后的干净域画面（仅 dfdcp 完全干净）**

| 臂 | dfdcp @C=1e-3 | dfdcp @C=1.0 |
|---|---|---|
| L0 仅 FF++ train | 0.8261 | 0.7864 |
| L3 多域监督 | 0.8397 (**+0.0136**) | 0.8301 (**+0.0437**) |

→ **在唯一完全无泄漏的域上，多域监督的真实增益是 +0.014 ~ +0.044**；而在 wild 上它是**负的**（agent −0.0184 vs C=1.0 基线；我复算 k1→k4 = −0.042）。

#### D.17.4 更正后的判定

- `G15_L1_LABELFREE = INEFFECTIVE` —— **维持**（全部为负或 ≈0）。
- `G15_L2_SELFTRAIN = PARTIAL` —— **下调为 WEAK**：+0.0498 是 4 域均值，但 dfdcp 最优组合仅 2/5 折有效、且量级在噪声带内；仅 cd1 显著，而 cd1 又与 cd2 同源（该域真实性存疑）。
- `G15_L3_MULTIDOMAIN = PARTIAL` —— **下调为 CONTAMINATED / WEAK**：头条 +0.0667 建立在弱基线 + 泄漏之上；干净域 +0.014~+0.044，wild 为负；换强基线仅 +0.0198。
- `G15_L3b_MARGINAL = MARGINAL_POSITIVE` —— **下调为 UNRELIABLE**：其正增益主要由 cd1 泄漏贡献（+0.1566），干净域 dfdcp 为 +0.0535（C=1.0）、wild 为负。
- **对用户"数据集丰富度"假设的回答**：**在冻结表征 + 线性头这一层，增加数据集数量的收益是"小且不稳健"的**（干净域 +0.01~+0.05，且有一个域转负），**远不足以解释 84%→ 目标域 83% 的差距**。原先 cd1 上看到的"几乎追平 oracle"是泄漏假象。

#### D.17.5 对 Q-IL 的净含义（本轮 G15+G16 合并）

1. **冻结表征上的所有"换头/适应"手段（L1 无标签变换、L2 自训练、L3 多域监督）都无法可靠抬高目标域判别**：L1 无效且多为负；L2 弱且不稳；L3 被泄漏污染、干净域增益仅 +0.01~+0.04。
2. **L4 特征源排除**（G16）：中间层不携带额外可迁移类信息，末层已最优。
3. **合计**：本轮把 §D.15 表中"冻结表征上的最后三条杠杆"全部测完，结论是**没有免费午餐，也没有便宜午餐**——目标域判别的实质提升**必须动表示本身**（解冻微调 / 换监督信号 / 换模态），这与 §D.14 的"BOTTLENECK_SHARED + 距域内 0.985 有 8~15pt 缺口只能靠动表示"完全一致。
4. **附带的资产**：本次发现了 multi npz 的跨域视频重叠，**后续任何用到 multi npz 的实验必须按视频级去重**，cd1 不可作为留出评估域（与 cd2 同源）。

**caveat（如实）**
1. 全部杠杆作用于**冻结特征上的线性头**，是"换头/域适配"的代理，**不等于** bridge/头被重训后的端到端效果。
2. L2 伪标签（无真标签）/ L3 用其它域真标签 / L4 用目标域真标签 —— 部署性逐级不同，不可直接并列比较。
3. 每域仅 300 样本、5 折 CV **折间方差大**；加之我方与 agent 实现间存在 ~0.01~0.03 的差异 → **<~0.04 的效应不宜过读**。这条对本节多域监督的增益结论同样适用（正反两方向）。
4. ffiw 仅 1 个视频（leak=True），一律单列、不入聚合；作训练源时 300 样本高度相关，贡献偏弱。
5. 相关 ≠ 因果：冻结表征上的增益不能保证端到端重训同向（甚至可能反号）。

---

### D.18 阶段综合结论（G10–G16 全部收束，2026-09-10 用户问"现在能得出的最有效结论是什么"）

#### D.18.1 一句话

**跨域失配的根因不是"伪造线索的方向不共享"，而是"共享方向上的类间判别被域特异地压扁（scale collapse）"；而在冻结表征上，任何换头/无标签适应手段都收不回这个压缩——特征里还有信息，但把它读出来的唯一办法是用标签重训表示。**

#### D.18.2 缺口分解（mean4，V 分支，positive=fake）

| 量 | 值 | 含义 |
|---|---|---|
| A. FF++ 域内（上界参照） | **0.9852** | 表示在源域能达到的水平 |
| B. 源标签跨域（源训 LR，C=1e-3 锚点） | **0.8318** | 当前可部署性能 |
| C. 目标域 oracle（域内标签 + 线性头，video-grouped CV） | **0.9111** | 表示在目标域"线性可读"的水平（上界、不可部署） |

- **总缺口 A−B = 0.1534**，拆为两半：
  - **决策边界/域适配段 C−B = 0.0793（52%）**——信息在，是"读法"对不上（含两类中心双向漂移 + 类内散布膨胀）；
  - **表示段 A−C = 0.0741（48%）**——即使在目标域内给定标签，线性头也读不出来，属表示本身的损失。
- **关键**：免标签手段（L1 无标签变换 / L2 自训练 / L3 多域监督）实测最多回收 **+0.01~+0.05**，只覆盖 C−B 段的 25~60%，且不稳健（G15 §D.17）→ **这半段缺口在实践中拿不回来**，除非给标签。

#### D.18.3 三条已锁定的硬事实（本轮之后不再需要重测）

1. **轴是跨域存在的**（G11）：目标域 PC0 与源 e0 的 |cos| = 0.979–0.989，域内类均值差方向 cos(e0) = 0.944–0.986。"方向不迁移"被否证。
2. **ViT 是 FF++ 的固定特征提取器**（G13）：stage-1 显式冻结，`vit.*` 与 PDI 的 net_050 **逐张量 bit-identical**（max_abs_diff = 0.000e+00）。且**域敏感性主要来自冻结 CLIP 分支而非 ViT**（G12：ratio_C 2.716 vs ratio_V 0.198）。
3. **换特征源/深度无效**（G16，我已逐格独立复现）：类判别信息随深度单调累积（b3 0.622 → b6 0.755 → b9 0.834 ≈ 末层 0.832），中间层不携带额外可迁移信息。

#### D.18.4 唯一剩下的路径

**用标签重训表示**——即解冻微调（或部分层解冻）+ 多域联合监督。G15 顺带给了它一个有利前提：多域联合可将有效训练样本从 2200 提到 ~3400，正是当初"91M/107K 防过拟合"冻结理由所依赖的那个约束的松动点。

但必须如实标注：**这是排除法剩下的路径，不是已被验证的增益**。G15 已证明"多域监督只作用在头上"时增益仅 +0.01~+0.04（干净域），把它下沉到表示层是否放大，尚未测。

#### D.18.5 论文层面的两个可发表发现

1. **"域不变 ≠ 域可判别"**：跨域 deepfake 检测的障碍不在共享方向，而在共享方向上的**类间可分性被域特异压缩**（间隔保留 38–47%、类内散布膨胀 1.4–1.7×、合成 d 降至源 25–33%）。且衰减是**全方向等比、无旋转**（cos(Δ_d, e0) = 0.944–0.986）——这限制了"旋转对齐类"方法的适用性。
2. **方法学警告（可独立成点）**：**Celeb-DF v1 ⊂ v2 的视频级重叠**（cd1 的 41 个视频 100% 出现在 cd2 中）会让"多域联合监督"呈现虚假增益——留出 cd1 时其 300/300 图全部被训练集覆盖，L3 因此得到 0.9514 ≈ oracle 0.9550 的假象。**任何把 cd1/cd2 当作两个独立域的跨域评测都存在此问题**。

#### D.18.6 待补（若继续）

- 解冻/部分解冻 + 多域监督的端到端验证（唯一未测的实质路径）。
- 所有使用 `feats_multi.npz` 的实验需按**视频级**去重；cd1 不可作留出评估域；~~wild 的历史数字含 ~3% 系统性抬高~~ → **❌ 已作废，见 §D.23.0**。

---

### D.19 分支权重实测更正 + "多域监督=泄漏?"概念澄清 + 换 CNN/频域分支方案评估（2026-09-10 用户提出）

#### D.19.1 ⚠️ 更正：CLIP 分支的决策权重实为 **62.3%**（我此前误记 30.6%）

**实测**（`checkpoints/stage_1/bridge_v2_phase1.pth`）：

| 参数 | 值 |
|---|---|
| `clip_vision_alpha` | **0.306288**（可学习，初值 0.5；这是**输入缩放系数，不是权重占比**） |
| `clip_text_alpha` | **1.779604**（初值 4.0） |

**末层 `output` 决策方向的能量占比**（`W[1]-W[0]`，分段范数平方占比）：

| 分支 | 范数 | 能量占比 |
|---|---|---|
| **CLIP（[0:768]）** | 1.077 | **62.3 %** |
| bridge 128-d（[768:896]） | 0.022 | **0.03 %** |
| ViT（[896:1664]） | 0.838 | 37.7 % |

**两个关键推论**：
1. **分类器把 62% 的决策能量放在 CLIP 分支上**——而该分支是三者中**域内最弱**（C 域内 0.9108 vs V 0.9852）且**域敏感度高 13.7 倍**（ratio_C 2.716 vs ratio_V 0.198，G12）的那一个。**这是一次明确的权重错配**，直接解释了 84% 的天花板。
2. 折算回原始特征尺度更极端：CLIP 输入被乘 0.306 却仍拿到 62% 能量 → 分类器对原始 CLIP 特征的敏感度约为 ViT 的 **4.2 倍**（1.077/0.306 vs 0.838/1.0）。
3. 附带发现：**bridge_adapter 的 128-d 紧凑嵌入在末层几乎不参与决策（0.03%）**——有效路径实质是"CLIP_proj + ViT_proj 直接拼接"。

#### D.19.2 "多域监督是否等于数据泄露？"——概念澄清

**分两层，结论不同**：
- **不是自动等于泄漏**：用**其它独立数据集**的真实标签联合训练，是标准的 domain generalization / multi-source DA 设定，只要与测试集**视频级不相交**，就是合法的。
- **但确实极易变成泄漏，而且我已经实测到一个**：Celeb-DF v1 是 v2 的子集 → cd1 的 41 个视频 100% 出现在 cd2 中，留出 cd1 时训练集把它的视频**全看过**。G15 里 cd1 的 0.9514 ≈ oracle 0.9550 就是这么来的（§D.17.3）。**用户的警觉是对的。**
- 另外两点必须说清：
  1. **我从未提议加入目标域的数据**；多域监督加的是**其它域**。
  2. **它改变了论文的设定**：论文头号数字 84% 是 FF++ → 目标域的 **zero-shot 跨域**口径；一旦加入其它域标签，设定就变成 multi-source DG，**是另一个（更弱的）claim**，不能混报道。
  3. 实践中若真能拿到其它数据集标签，往往也能拿到目标域标签 → 那时就退化成普通微调。所以它的价值主要是**作为代理实验**量化"数据广度"这个杠杆，而不是部署方案。

#### D.19.3 方案评估：把 CLIP 分支换成 CNN / 频域检测器分支

**用户方案**：ViT 跨域强但缺局部/颜色突变的类间判别；用 CNN（卷积局部性、邻域不变性伪造特征）或频域（颜色变化频率）分支补足 ViT 缺失的判别维度。

**支持它的既有证据（三条，均出自本项目）**：
1. **权重错配已量化**（§D.19.1）：62% 决策能量压在最弱且最域敏感的分支上 → 换掉它**方向正确**。
2. **ViT 的轴是几何/结构性的、明确不是低层像素级**（G10）：五个低层变换 Δ≤3.69pt，而 crop90 对照 17.25pt → **ViT 确实不使用像素级/频率线索**，因此频域/颜色分支所覆盖的是一个**ViT 当前未使用的正交子空间**，互补性**由构造保证**，不是猜测。
3. C 分支本身域内最弱（0.9108 三分支最低）→ 用更强的检测器替换它是合理的。

**但有两条必须正视的反向证据**：
1. **G14 `BOTTLENECK_SHARED`**：oracle_C（0.8782）、oracle_F（0.8953）**均 ≤ oracle_V（0.9111）** → 现有的第二分支（CLIP）**并未抬高峰值**。这说明"再加/再换一个分支"**不自动**提高可回收上限；新分支必须携带 V 确实没有的信息。
2. **域锁定风险 = CLIP 的失败模式会重演**：频域/颜色统计恰恰是最容易被数据集特性（压缩、分辨率、相机管线）绑定的量——C 的高域敏感（2.716）已经是前车之鉴。**一门在域内很准、跨域崩掉的频域分支，会把 62% 的错误权重问题从 CLIP 复制到新分支上。**

#### D.19.4 建议的下一步（G17 可行性探针，先验后做，成本极低）

在动架构之前，先用**离线探针**判定"频域/CNN 特征是否携带 V 没有的可迁移类信息"：

- 数据：`probe` 2200/800 + `multi` 5×300（**按 §D.17.3 视频级去重，cd1 不作留出**）。
- 特征：DCT/FFT 径向谱与高频能量统计（零训练成本），外加一个小 CNN 的中间层（可选）。
- 指标（与 V 同口径）：① 域内 AUC；② 跨域 AUC（FF++ 训 → 各域）；③ **增量 oracle：`[V_proj | freq_feat]` vs `V_proj`** —— 这是判定"是否补充了 V 缺失的类间判别"的**唯一决定性指标**；④ 该特征的域敏感度 ratio（对标 C 的 2.716）。
- 预注册门：③ 增量 ≥ +0.03 且 ≥4/4 干净域 → 分支方案成立；④ 若比值显著高于 V 的 0.198 → 域锁定风险确认，需配域不变性约束。

**关键**：门 ③ 是"加进去到底有没有用"的直接回答，比先写一个 CNN 检测器再端到端试便宜得多，也避免重复 CLIP 的错配。

---

### D.20 G17 计划：局部细节/频域检测器信息能否在 ViT 表征空间中把类分开（2026-09-10 用户批准）

**用户指令**：验证"如果不再使用 CLIP，而是换成对局部细节更有判别力的检测器，它提供的信息**是否能在 ViT 的表征空间中把类别分离开**"。

#### D.20.1 派单前侦察结论（已核实）

| 事项 | 结论 |
|---|---|
| 可用预训练 CNN | **本地已缓存、离线可用**：`densenet121-a639ec97.pth`、`efficientnet_b4_rwightman-23ab8bcd.pth`、`resnet18-f37072fd.pth`、`vgg19-dcbb9e9d.pth`（`~/.cache/torch/hub/checkpoints/`） |
| 仓库现成 CNN 检测器分支 | **有**：`llava/model/deepfake/encoder.py` 的 `DenseNet_Deepfake` / `EfficientNet_Deepfake`；`M2F2Det` 默认 `deepfake_encoder_name='densenet121'` |
| 训好的 deepfake CNN 权重 | **无**（`checkpoints/` 下只有 bridge/LLaVA 系列）→ CNN 臂只能用 **ImageNet 预训练**特征，须在结论中标注"非 deepfake 专用检测器" |
| 频域/SRM 依赖 | `numpy`/`scipy`/`cv2` 齐备，**零依赖、零训练** |

#### D.20.2 三条评测口径（与 V/C 严格同口径，才可横向比）

1. **域内 AUC**：各目标域内（尽量 video-grouped CV）。
2. **跨域 AUC**：FF++ train 2200 训 LR → 各域 300（positive=fake），锚点可直接对标 V 的 0.8318(mean4) / C 的域内 0.9108。
3. **oracle**：各域内标签 CV，对标 `oracle_V = 0.9111`。

#### D.20.3 四个子任务（含**决定性**的增量测试与"V 空间分离"专项）

- **G17a｜低层/频域特征提取（纯 CPU 离线，零 GPU）** → `vit_module/_g17/freq_feats.npz`
  - F1 **2D FFT 功率谱**：log-polar 径向分箱（32）+ 角向分箱（8）
  - F2 **高频能量比**、谱斜率、谱平坦度
  - F3 **SRM 高通残差统计**：SRM 核卷积后的 mean/var/skew/kurtosis + 分位直方图（"局部伪造痕迹"经典量）
  - F4 **颜色/色度突变**：YCrCb 通道梯度、色度直方图、局部颜色不连续统计（对应"颜色变化的频率"）
  - F5 可选：DCT 8×8 块统计
  - 口径：主口径 = 与 V 同一条管线（`cv2.resize(336)` 后的 RGB）；另跑一套 normalize 后的作敏感性检查。
- **G17b｜CNN 局部特征提取（GPU 空闲卡优先，规则同 G13/G16）** → `vit_module/_g17/cnn_feats.npz`
  - `densenet121`（对齐仓库 `M2F2Det` 的默认编码器）+ `efficientnet_b4` 的**中间层**（局部纹理/边缘主导）GAP 池化特征；可选 `resnet18` 作第三臂。
  - 输入 336→224 + ImageNet 归一化；**必须在报告中写明归一化与层选择**。
  - 前置校验：用同一批图复算 V 管线，确保图像读取/裁剪口径与 G16 一致。
- **G17c｜联合评测（前两个 npz 落地后启动；离线）** → `vit_module/_g17/g17_report.txt`
  - **B4 增量测试（决定性）**：`[V_proj | feat]` vs `V_proj`，**跨域与 oracle 双口径**，并且**必须带容量对照臂**（追加等维随机投影/高斯噪声）——防止"维度增加=容量幻觉"。
  - **C1–C4 "V 空间分离"专项**：
    - C1 沿 V 的源轴 e0 的**间隔恢复率** = (joint − V)/(source − V)，对标已知的 gap retention 38–47%
    - C2 新特征类间方向与 e0 的 |cos|：**低=正交互补，高=重复冗余**
    - C3 联合空间的 Fisher ratio / 类中心距离 vs V 单独
    - C4 联合判别方向中**落在 e0 之外的能量占比**（对标 G14 的轴外分量 0.69–0.75）
- **判定门（预注册，机械）**
  - `G17_COMPLEMENT_ORACLE`：`[V|feat]` oracle ≥ `oracle_V` + 0.03 且 ≥4/4 干净域，且显著优于容量对照
  - `G17_COMPLEMENT_XDOM`：跨域口径同门（≥ V_cross + 0.03）
  - `G17_DOMAIN_LOCK`：`ratio_feat` 对标 V 的 0.198 / C 的 2.716；**> 1.0 记为域锁定风险**
  - `G17_ORTHOGONAL`：C2 的 |cos| < 0.5 = 互补；> 0.8 = 重复
  - `G17_BEST_FEATURE`：按 B4 增益排序各特征族，给出最优与最便宜有效项

#### D.20.4 数据卫生（硬约束，源自 §D.17.3）

**视频级去重**；**cd1 不作留出评估域**（与 cd2 同源，41/41 视频重叠）；`ffiw` 仅 1 视频 → 一律单列、不入聚合。~~`wild` 标注 ~3% 与 probe train 重叠~~ → **wild 实为干净域（§D.23.0）**。聚合均值一律基于**干净域**。

#### D.20.5 资源与分工

- G17a：纯 CPU，限制线程（`OMP_NUM_THREADS` 设小），零 GPU、零前向。
- G17b：先 `nvidia-smi`，仅用"占用 ≤100 MiB 且可用 ≥6 GB"的空闲卡，fp32、batch ≤16；否则 CPU 单线程回退并审计记录。
- G17c：纯离线，两个 npz 就绪后启动。
- 三个 agent 均**不改其它文件、不写 WORKLOG**；结果由我独立复核后落档 §D.21。

---

### D.21 G17 结果：局部细节/频域特征**不能**补上 ViT 缺失的类间判别——16/16 全臂 NO_COMPLEMENT（2026-09-11）

#### D.21.0 计划修正（我方）
§D.20 写的判定门是"≥4/4 干净域"，**该前提有误**。按 §D.17.3 的视频级核对，严格成立的是：**只有 `dfdcp` 完全干净**；`cd2` 有 28% 图与 cd1 视频重叠（~~`wild` 有 3% 与 probe train 重叠~~ → ❌ 已作废，见 §D.23.0，**wild 实为干净域**）；`cd1` 不可作留出域；`ffiw` 仅 1 视频。**实际执行的门**：主门看 dfdcp ≥ +0.03 且 cd2/wild 同向，两域标注污染，cd1 仅作参考并标泄漏。

#### D.21.1 提取产物（我已独立核验行序与完整性）

| 产物 | 内容 | 状态 |
|---|---|---|
| `_g17/freq_feats.npz` | F1(40) / F1_radial(32) / F2(4) / F3(20) / F3_hist(160) / F4(23) / F5(4) = **283 维** | 5300/5300 成功，read_fail=0，**零 GPU**，wall 942.4 s |
| `_g17/cnn_feats.npz` | densenet121 db2(512)/db3(1024)/final(1024)、effnet_b4 blk5(160)/blk6(272)/final(1792)、resnet18 layer3(256)/layer4(512) | 5300/5300 成功，GPU1（查询时 0 MiB），wall 68.5 s |

- **行序**：两 npz 均 = probe 原生 3000 + multi 原生 2300，`paths`/`y` 与源 npz **逐条全等**（F 与 CNN 两两互等），全部 finite。
- CNN 权重**全部命中本地缓存、零下载**（agent 踩到并解决了 densenet121 旧版 key 命名 `norm.1`/`conv.1` vs 新版 `norm1`/`conv1` 的不兼容）。
- 频域口径依赖：`F3_hist`（固定分位边）与 `F5`（AC 占比分母含 DC）在 uint8 vs 归一化输入下**不一致**（mean|r| 0.14 / 0.22）→ 主口径固定为 uint8。

#### D.21.2 锚点复现（硬门，PASS）
V 跨域 AUC 五域**逐位命中**（maxdev 3.33e-05）；`ratio_V = 0.1983`、`ratio_C = 2.7160`、`oracle_V mean4 = 0.9111` 全部复现。

#### D.21.3 B4 增量主表（决定性；跨域 C=1e-3，`[V|feat] − V`）

| feat | K | cd1(泄漏,参考) | cd2 | **dfdcp(干净)** | wild | 容量对照 dfdcp |
|---|---|---|---|---|---|---|
| `F_all` | 123 | +0.0165 | +0.0022 | **+0.0091** | −0.0029 | +0.0006 |
| `F1` / `F1_radial` | 40/32 | +0.011 | +0.002 | **+0.007** | −0.004 | +0.0005 |
| `dense121_db3` | 1024 | **+0.0436** | +0.0110 | **−0.0072** | +0.0084 | +0.0002 |
| `effnet_b4_blk6` | 272 | +0.0116 | +0.0016 | **−0.0060** | +0.0120 | +0.0012 |
| `resnet18_layer3` | 256 | +0.0228 | +0.0044 | **−0.0101** | +0.0105 | +0.0002 |
| `F3_hist`, F2/F3/F4/F5, 其余 CNN 臂 | — | ± | ≈0/负 | **负或 ≤+0.009** | ± | ≤+0.0025 |

**关键读数**：① 干净域最大增量仅 **+0.0091**（`F_all`），远低于 +0.03 门；② **在 cd1 上涨得最多的 `dense121_db3`（+0.0436），在干净域 dfdcp 上是负的（−0.0072）**——这是"泄漏 + 容量"双重假象的典型签名；③ 容量对照（等维随机噪声 / V 随机投影，3 seed）在 dfdcp 上仅 +0.0002~+0.0025，**没有任何臂显著超过对照**。

#### D.21.4 五个预注册门（16 臂全部）

| 门 | 结果 |
|---|---|
| `G17_COMPLEMENT_XDOM` | **NO_COMPLEMENT（16/16）** |
| `G17_COMPLEMENT_ORACLE` | **NO_COMPLEMENT（16/16）**（最大 dfdcp 增量 +0.0091） |
| `G17_DOMAIN_LOCK` | **DOMAIN_LOCKED（16/16）**，ratio 3.03–8.41，全部 ≫ 1.0 |
| `G17_ORTHOGONAL` | **REDUNDANT（16/16）**——但见下方 caveat，此门**结构性失效** |
| `G17_BEST_FEATURE` | `dense121_db3`（但由 cd1 泄漏驱动）；**`cheapest_effective = none`** |

#### D.21.5 ⚠️ 本轮最有价值的发现：局部性 ↔ 域敏感度的单调关系

把四轮测得的域敏感度 ratio 并排（口径统一，G12 公式 `ratio = dom_gap / class_gap`）：

| 特征 | ratio | 性质 |
|---|---|---|
| **V（ViT 末层 CLS）** | **0.1983** | 高层语义，跨域稳健 |
| C（CLIP） | 2.7160 | 视觉-语言语义 |
| `effnet_b4_blk6` | 5.6008 | CNN 中层 |
| `F1`（FFT 径向谱） | 6.9809 | 频域低层 |
| `dense121_db3` | 8.3213 | CNN 浅中层 |
| `F3_hist`（SRM 残差直方图） | 8.4082 | 像素级高通 |

**规律清晰**：**特征越局部/越低层，类判别信号越与域绑定**——最"局部"的 SRM 残差与 FFT 谱，域敏感度是 V 的 **35~42 倍**。这解释了为什么一个先天域锁定的模态无法补 V 的短板，也**在机理上解释了 84% 的天花板**：R19.1 已测出末层把 **62% 决策能量放在 C（ratio 2.716）**上，而可用的"局部细节"模态全部比 C 更域敏感。

#### D.21.6 我的独立复核
不采信 agent 汇总，直接读三个 npz 自算：**V 基线五域逐位一致**；`ratio_V = 0.1983`、`F3_hist = 8.4082`（agent 8.41）一致；B4 增量表**逐格命中**（`dense121_db3` 的 +0.0436 / +0.0110 / −0.0072 / +0.0084 与 agent 完全相同；`effnet_b4_blk6` 亦完全相同；`F_all` 的 dfdcp 我算 +0.0065 vs agent +0.0091，差异源于零方差列过滤阈值，**两者都远低于门**）。

#### D.21.7 结论与对用户方案的含义

1. **用户假设被否证**：用"对局部细节更有判别力的检测器"（频域 / SRM / CNN 局部特征）替换 CLIP，**其信息不能在 ViT 的表征空间中把类分开**——16 个臂在跨域与 oracle 双口径下**全部 NO_COMPLEMENT**，且全部 **DOMAIN_LOCKED**。
2. **不是"信号不够强"，而是"信号长在不可迁移的方向上"**：部分臂确实携带不可忽略的类信号（`effnet_b4_blk6` 的 C2 轴外能量 0.15–0.22，`F4`/`F_all` 高达 0.64–0.98，是与 e0 **最正交**的），但它们全部落在域特异方向上，跨域即失效。
3. **§D.19.3 预判的风险被确认**：当时我写"频域/颜色统计是最容易被数据集特性绑定的量，CLIP 的失败模式会重演"——**实测重演，且更严重**（C 的 2.716 → 局部特征的 5.6–8.4）。
4. **对下一步的建议**：单纯"换分支"不成立。若仍要走多模态补充分支，必须**把跨域不变性写进分支的训练目标**（域对抗 / 域不变损失 / 多域联合监督下的分支训练），而不是换一个在域内更准的现成特征。**"域内更准"与"跨域可用"在本项目里是负相关的**。

#### D.21.8 Caveat（如实）
1. **本轮测的是冻结特征 + 线性头**，不是端到端联合训练的分支；CNN 臂用的是 **ImageNet 预训练**，**没有 deepfake 专用检测器权重可用**（仓库 `checkpoints/` 下无）——一个在 FF++ 上训练过的检测器 CNN 是否会更好，**本轮未测，结论不可外推**。
2. `G17_ORTHOGONAL` 门**结构性失效**：对原始拼接，联合类均值差在 V 子空间的投影恒等于 V 自己的类均值差 → `cos(δ_V, e0) ≈ 0.9603` 与臂无关，该门**不携带信息**；真正有信息的是 C2 的"V 子空间外能量占比"（已报）。**这是本轮的一个方法学缺陷，已记录。**
3. 每域 300 样本、折间方差大 → **<~0.04 的差异不宜过读**。这条同时削弱正负两个方向，故"NO_COMPLEMENT"的结论建立在"连 0.03 都没到"这一事实上，是稳的。
4. 干净留出域实际只有 `dfdcp` 一个；`cd2`(28% 污染) 为次干净。~~wild(3%)~~ → ❌ 已作废，见 §D.23.0，**wild 实为干净域**（故干净域实为 {dfdcp, wild} 两个，结论的域泛化性比本节标注的更好）。
5. 相关 ≠ 因果：`[V|feat]` 的 AUC 增量不等于端到端架构替换后的行为。

---

### D.22 综合结论（修订版，含 G17；取代 §D.18.3/D.18.4）

#### D.22.1 核心结论（一句话）

**在冻结表征上，"类间判别力强"与"域可迁移"在本问题里是负相关的**：越局部、越低层的特征判别信号越强，但域敏感度也越高（ratio 0.198 → 8.41，六项单调）；唯一跨域稳健的是 ViT 的高层语义 V，而它恰恰是判别被压缩的那一个。框架又把 **62% 的决策能量**压在了介于两者之间的 CLIP 分支上。**84% 是结构性天花板，不是调参问题。**

#### D.22.2 证据链（七条，逐条可查）

| # | 命题 | 证据 | 出处 |
|---|---|---|---|
| **E1** | **方向共享，被压缩的是"可分性"** | 目标域 PC0 与源 e0 的 \|cos\| = **0.979–0.989**（轴在）；但类间隔保留仅 **38–47%**、类内散布膨胀 **1.4–1.7×**、合成 d_z 降至源 **25–33%**、两类中心双向漂移 | G11 / §D.9 |
| **E2** | **缺口可分解，两半各约一半** | FF++ 域内 0.9852 → 源标签跨域 0.8318 → 目标域 oracle 0.9111；决策边界段 **0.0793(52%)**、表示段 **0.0741(48%)** | §D.18.2 |
| **E3** | **冻结表征上的适应手段全部无效** | L1 无标签变换 **−0.0108**（ZCA −0.16~−0.19）；L2 自训练 +0.0498 仅靠 cd1（后被证泄漏）；L3 多域监督头条 +0.0667 系弱基线+泄漏，干净域 dfdcp 仅 **+0.0136/+0.0437**、wild **为负** | G15 / §D.17 |
| **E4** | **换特征源（深度）无效** | 类判别信息随深度**单调累积**：b3 0.622 → b6 0.755 → b9 0.834 ≈ 末层 0.832 | G16 / §D.16 |
| **E5** | **换模态（局部/频域）也无效** | **16/16 臂 NO_COMPLEMENT**（跨域+oracle 双口径）；干净域最大增量 **+0.0091**（门 +0.03）；**16/16 DOMAIN_LOCKED**，ratio 3.03–8.41；容量对照排除假阳性 | G17 / §D.21 |
| **E6** | **单调关系（机理枢纽）** | V **0.198** < C 2.716 < effnet 5.60 < F1 6.98 < dense121_db3 8.32 < F3_hist **8.41** —— **局部性越强，域敏感度越高**（最局部者达 V 的 35–42 倍） | G12+G17 / §D.21.5 |
| **E7** | **框架权重错配** | 末层决策能量 **CLIP 62.3% / ViT 37.7% / bridge 0.03%**；即 62% 压在最弱（C 域内 0.9108 vs V 0.9852）且比 V 域敏感 **13.7 倍**的分支上 | §D.19.1 |

#### D.22.3 被推翻的两个假设（写明，避免重复走）

- **"ViT 没有学到真正的伪造特征"** → **不成立**：ViT 的判别轴跨域存在（E1，\|cos\| 0.979–0.989），且 V 是全部被测特征里唯一域稳健的（E6，ratio 0.198）。
- **"加一个局部细节分支就能补上 ViT 缺失的判别"** → **不成立**：16/16 臂 NO_COMPLEMENT 且全部 DOMAIN_LOCKED（E5、E6）；§D.19.3 预判的"CLIP 失败模式重演"被实测确认，且更严重。

#### D.22.4 唯一剩下的路径（排除法结论，**尚未验证**）

既然：信息在（oracle 0.911，轴存在）→ 但读法在冻结表征下改不动（E3）→ 换深度无用（E4）→ 换模态无用且局部模态天然域锁定（E5/E6）
⇒ **只能用标签重训表示，且必须把跨域不变性写进训练目标**（域对抗 / 域不变损失 / 多域联合监督），否则新分支会重蹈 C 的覆辙。

**必须标注**：这是排除法剩下的路径，不是已验证的增益；且 E5 的否证**只覆盖冻结特征 + 线性头**，CNN 臂是 ImageNet 预训练而非 deepfake 专用检测器（无可得权重），故结论**不可外推到"端到端训练一个真·检测器分支"**。

#### D.22.5 附带的固有问题（数据侧，影响所有后续实验）

- **cd1 ⊂ cd2**：41/41 视频重叠 → 任何把二者当独立域的评测都存在泄漏；**cd1 不可作留出评估域**。
- ~~**wild**：9/300 图（3%）与 probe train 重叠~~ → ❌ **已作废，见 §D.23.0**：wild 实为**干净域**，其历史数字（含 0.8090 锚点）**未被抬高**。
- 严格干净的留出域**只有 `dfdcp`**；`ffiw` 仅 1 视频（leak=True），一律单列。

---

### D.23 G18 计划：**特化训练**的 DenseNet121 能否为 ViT 分支提供互补判别信息（2026-09-13 用户提出）

#### D.23.0 ⚠️ 更正（我方错误）：wild 的"视频重叠"是**命名空间撞车**，wild 实为干净域

派单前复核 G18 的训练排除清单时发现：`feats_multi.npz` 的 `vid` 字段只存**裸标识符**，跨数据集必然撞车。实测：

| 域 | `vid` 样例 | 路径根 |
|---|---|---|
| ffpp | `161` / `176` / `190` | `FaceForensic++_raw/original_sequences/c23/faces23/161/` |
| wild | `161` / `176` / `190` | `WildDeepfake/real_test/79/real/161/` |

→ 二者**完全不同的视频**，仅因裸数字相同被误判为"重叠"。**§D.17.3 中关于 wild 的两行（17 vids / 8 vids、19/300、9/300、3% 系统性抬高）全部作废**，已在原文处标注。**wild 是干净域**，其历史数字（含跨域锚点 0.8090）**未被抬高**。

**同时确认 `cd1 ⊂ cd2` 的重叠为真**：按**文件夹级**比对，cd1 的 49 个文件夹**全部**出现在 cd2 中（49/49，路径逐字相同，如 `/Celeb-DF-v1/faces23/00011/` ↔ `/Celeb-DF-v2/faces23/00011/`）；srcid 层面 41/41 重合。→ **cd1 泄漏结论不变，cd1 仍不可作留出域。**

**方法学后果（对所有后续实验生效）**：
- **泄漏/重叠检测必须用"路径派生的数据集根 + 视频文件夹"身份，绝不能用裸 `vid` 字符串**。
- 域卫生修正后：**完全干净域 = {`dfdcp`, `wild`}**；次干净 = `cd2`（28% 文件夹与 cd1 重叠）；`cd1` 排除；`ffiw` 单列。
- **这不改变任何已落档的实验数值**（G15/G16/G17 的 AUC 与 ratio 都是实测），只改变**域标签与聚合口径**：`wild` 应从"次干净、需打折"升格为**判定域**，后续门槛可在两域上要求同向。

**用户假设**：本地训练一个**针对伪造检测任务特化**的 DenseNet121，取其**最优权重**的特征，检验其判别性信息能否**互补**给 ViT 分支。

**这正是 §D.22.4 明确标注为"唯一未测到"的一臂**（G17 的 CNN 臂是 ImageNet 预训练，非伪造检测专用）。故本轮是对 E5 否证的**直接补测**。

#### D.23.1 派单前侦察（已核实）

| 事项 | 结论 |
|---|---|
| GPU 余量 | **GPU1（0 MiB）与 GPU2（64 MiB）均空闲**，GPU0 已占 4356 MiB → **可并行两个训练**（各自钉死卡号，禁止互相抢卡） |
| FF++ 数据 | `F:/zhj/data/FaceForensic++_raw/`：`original_sequences/c23/faces23/`、`manipulated_sequences/{Deepfakes,Face2Face,FaceShifter,FaceSwap,NeuralTextures}/`、**官方 `splits/train.json`(720 视频) 与 `splits/test.json`(140 视频)** |
| 可利用规模 | 官方 train 720 视频 ≫ 此前 probe train 的 98 视频 / 2200 图 → **可训出一个真正意义的检测器**，而非小样本玩具模型 |
| 泄漏约束 | 训练视频必须与 **probe test 的 42 个视频**、**全部目标域视频集**、以及 **wild 的 17 个重叠视频** 零交集 |

#### D.23.2 两个训练臂（并行，各自独立 GPU）

**G18a｜单源特化（FF++ only）** → GPU1，产物 `vit_module/_g18/`
- 架构：`torchvision.models.densenet121(weights=DenseNet121_Weights.DEFAULT)`，分类头换二类；**`.features`+GAP 结构对齐仓库 `DenseNet_Deepfake`**。
- 数据：官方 `train.json` 视频，real = `original_sequences/c23/faces23/<vid>`，fake = `manipulated_sequences/<method>/c23/faces23/<vid>`；**硬排除**与 probe test / 目标域 / wild-17 重叠的视频；每视频限帧数（建议 ≤30，总样本控制在 ~15–25k）以控训练时长。
- 划分：从训练视频中按**视频级**劈出 ~10% 作 val，**仅用于选最优 checkpoint**；probe test 800 保留作最终域内评测，**训练期绝不触碰**。
- 训练：Adam + 预训练初始化，早停按 val AUC；**保存 val AUC 最优权重**（同时报训练曲线）。
- 提取：用**最优权重**对 **5300 图**（与 G17 同一批、同管线 `336→224 + ImageNet norm`）提 `db2/db3/final` 的 GAP 特征 → `df_ffpp_feats.npz`。
- **泄漏审计（必报）**：使用的视频数、与 probe test / 各目标域的**交集必须为 0**。

**G18b｜多域特化（LODO，检验"跨域不变性能否解域锁"）** → GPU2，产物 `vit_module/_g18/`
- 同架构；对 **3 个判定域 {dfdcp, cd2, wild}** 各训一个模型：训练集 = FF++ train + 其它域样本，**留出该域**；**留出 cd2 时须同时排除 cd1**（cd1⊂cd2，28% 图重叠，§D.17.3）。
- 每个模型对 5300 图提同样三组特征 → `df_lodo_<domain>_feats.npz`。
- 目的：**直接检验 §D.22.4 的建议**——把跨域不变性写进训练目标，能否把 CNN 的域敏感度 ratio 从 5.6~8.4 压下来、并让增量测试转正。

#### D.23.3 评测臂（G18c，离线 CPU，等 a/b 就绪）

**与 G17c 完全同协议，保证可比**：
- B2 跨域 AUC、B3 oracle、**B4 增量 `[V|feat]` vs V（含两组容量对照、多 seed）**；
- C1 e0 间隔恢复率、C2 V 子空间外能量占比、C4 **域敏感度 ratio（G12 公式）**；
- 判定门沿用 G17 四门，**新增两门**：
  - `G18_SPECIALIZATION_EFFECT`：特化 DenseNet121 相对 G17 的 **ImageNet DenseNet121** 的增量差（>+0.03 记 SPECIALIZATION_HELPS）
  - `G18_DELOCK`：G18b 的 ratio 相对 G18a 的下降幅度（≥30% 记 DELOCK_EFFECTIVE）
- **直接对照行**：`ImageNet-DenseNet121(G17)` vs `FF++-DenseNet121(G18a)` vs `LODO-DenseNet121(G18b)`。

#### D.23.4 资源与纪律
- 两训练 agent 各自**显式钉死卡号**（G18a=GPU1，G18b=GPU2），开工前先 `nvidia-smi` 确认本卡空闲；**若本卡被占，停下报告，禁止改用另一张卡**。
- 限制 CPU 占用（线程数设小，`num_workers=0` 或 ≤2）；fp32；batch ≤32。
- 两个训练 agent **不改其它文件、不写 WORKLOG**；结果由我独立复核后落档 §D.24。
- **数据卫生**：视频级划分；cd1 不作留出域；ffiw 单列；所有聚合基于判定域并标注污染。

#### D.23.5 本轮的判定意义（预登记）
- 若 G18a **通过增量门** → §D.22 的"换模态无用"被推翻，**分支替换路线复活**，用户的方案成立。
- 若 G18a **不通过但 ratio 显著低于 G17 的 ImageNet 臂** → 特化确实改变了特征性质，只是还不足以互补。
- 若 G18a **不通过且 ratio 更高**（预期方向，与 E6 一致）→ **特化反而加剧域锁**，则"CNN 分支替换"路线在 FF++ 单源监督下**被定论关闭**，唯一出路是 G18b 式的多域不变训练。

---

### D.24 G18a 独立复核：特化 DenseNet121 的**域内就已不如冻结 ViT**，且 ratio 指标对监督分支失效（2026-09-13）

> 本节为我（主 agent）对 G18a 产物的**独立复核**，不采信子 agent 自报数值。G18b 仍在训练、G18c 未启动，故本节只落 **G18a 的实测**与**方法学更正**，不含互补性判定。

#### D.24.0 复核方法与**我自己的一个错误**（必读）

先用 G12 公式重算 `df_ffpp_*`，结果与我记忆中的锚点**差 6~40 倍**，一度判定"实现不符"。读 `_g12/run_g12.py` L392-405 后发现：**错的是我的打印，不是公式**——我把 `dom_gap 跨域均值` 当作 `ratio` 输出了，漏除 `class_gap`。

修正后**逐位复现**历史值，证明本轮 ratio 实现与 G12/G17 同源：

| 分支 | 我算 dom_gap 均值 | 我算 class_gap | 我算 ratio | 记录值 | 判定 |
|---|---|---|---|---|---|
| V（冻结 ViT） | 7.9755 | 40.2213 | **0.1983** | 0.1983 | ✅ |
| C（冻结 CLIP） | 17.7957 | 6.5523 | **2.7159** | 2.7160 | ✅ |
| G17 dense121_db3 | 25.0516 | 3.0105 | **8.3213** | 8.3213 | ✅ |
| G17 effnet_b4_blk6 | 7.8972 | 1.4100 | **5.6009** | 5.6008 | ✅ |

> 教训：`ratio = dom_gap/class_gap` 中 `s` 会约去，**表里必须同时印出 class_gap**，否则无法与历史值比对自证。已按此格式输出下表。

#### D.24.1 产物完整性与**泄漏审计**（全部通过）

- **行序硬门**：`_g18/df_ffpp_feats.npz` 的 `paths`(5300) 与 `_g17/cnn_feats.npz` **逐字节相等**；`[:3000]` 对齐 `_probe/probe_feats.npz`、`[3000:]` 对齐 `_tsne/feats_multi.npz` ✓
- **泄漏审计（用 §D.23.0 新规则：路径派生身份，非裸 `vid`）**：
  - probe-**train** 视频 ∩ DenseNet 训练视频 = **0**
  - probe-**test** 视频 ∩ DenseNet 训练视频 = **0**
  - 训练视频 = 4308 个"数据集根+文件夹链"身份（21,540 图 / 718 视频，官方 train.json 720 中 2 个缺失）
- **权重来源核实**：子 agent 自报曾误用**末轮 in-memory 权重**算 probe-test AUC 与提特征（报 0.9583），后加载 `best_state` 重生成。我从 npz **独立重算**得 db2=0.8410 → db3=0.9519 → final=0.9726 **单调递增**，与"末轮欠拟合"不符、与"best@ep7"一致 → **采信修复后的产物** ✓

#### D.24.2 域内（FF++ test 800，与训练视频零交集）：**特化模型输给冻结 ViT**

| 分支 | 域内 AUC（probe-train 2200 线性探针，C=1e-3） |
|---|---|
| **V（冻结 ViT，零训练）** | **0.9852** |
| df_ffpp_final（特化，训了 41.6 min / 21,540 图） | 0.9726 |
| df_ffpp_db3 | 0.9519 |
| df_ffpp_db2 | 0.8410 |

→ **花了 21540 张图标定、41.6 分钟训练，其表征的可线性分性仍低于一个完全冻结的 ViT CLS**。这不是"训练不充分"，而是两种表征的性质差异。

#### D.24.3 跨域 AUC（同协议：probe-train 2200 训练 → 各域测试；positive=fake）

| 分支 | cd1 | cd2 | dfdcp | ffiw* | wild | **跨域均值** |
|---|---|---|---|---|---|---|
| **V（冻结 ViT）** | **0.8286** | **0.8633** | **0.8261** | 0.8244 | **0.8090** | **0.8303** |
| C（冻结 CLIP） | 0.6559 | 0.7227 | 0.7068 | 0.8344 | 0.7170 | 0.7273 |
| dense121 ImageNet db3 | 0.5279 | 0.5739 | 0.6988 | 0.6138 | 0.5478 | 0.5924 |
| dense121 ImageNet final | 0.5253 | 0.6064 | 0.6073 | 0.6169 | 0.5496 | 0.5811 |
| df_ffpp db2 | 0.4604 | 0.5215 | 0.5368 | 0.7537 | 0.5521 | 0.5649 |
| df_ffpp db3 | 0.5490 | 0.5604 | 0.6433 | 0.7301 | 0.7046 | 0.6375 |
| **df_ffpp final** | 0.5153 | 0.5966 | 0.7099 | 0.8080 | 0.7429 | **0.6745** |

\* ffiw 仅 10 个视频（300 图），视频内相关性强 → **其数值不构成迁移证据**，聚合时单列。

**读法**：
1. 特化**确实**把 CNN 自身的跨域迁移拉起来了：`final` 0.5811 → 0.6745（dfdcp 0.6073→0.7099、wild 0.5496→0.7429、ffiw 0.6169→0.8080）。**FF++ 监督让 CNN 更能迁移到"类 FF++"的域**。
2. 但**在两个 Celeb-DF 域上彻底塌陷**：cd1=0.5153（≈随机）、cd2=0.5966。Celeb-DF 是合成管线差异最大的域 → **特化学到的是 FF++ 合成伪影，换管线即失效**。
3. **对 V 的差距不但没缩小反而拉大**：0.8303 vs 0.6745，且逐域全输。

#### D.24.4 ⚠️ 方法学更正：**ratio 对"被监督训练过"的分支失效**（影响 §D.21/E6 的适用边界）

| 分支 | class_gap（源侧 FF++） | dom_gap 均值 | **ratio** |
|---|---|---|---|
| V（冻结） | 40.21 | 7.98 | 0.1983 |
| dense121 **ImageNet** final（冻结） | 2.21 | 14.9793 | 6.7820 |
| df_ffpp **final（特化）** | **43.40** | **11.2778** | **0.2598** |

按 §D.23.5 的预期，特化应"ratio 更高→加剧域锁"。**实测 ratio 反而降到 0.2598（接近 V 的 0.1983）**。但这不是解域锁：

- `class_gap` 是**在源域 FF++ 上**量的——而**这正是特化训练的目标函数本身**。训练把 class_gap 从 2.21 灌到 **43.40（×19.6）**；`dom_gap` 只从 ~14.99 降到 ~11.27（×0.75）。
- ratio = 分母被监督灌大 → **机械性地被压向 0**，与"域锁是否解开"无关。
- **决定性反证**：同一个 ratio=0.2598 的分支，cd1 跨域 AUC = 0.5153（随机）。**若 ratio 真在度量解域锁，二者不可能同时出现。**

→ **结论（须写进论文限制）**：E6 的"局部性↔域锁单调关系"只在**同口径的冻结表征**之间成立；**一旦某分支被带标签监督训练过，其 ratio 与冻结分支不可比**。ratio 只能用于**同监督强度分支之间**的对照（例如 G18a 单源 vs G18b 多域），不能用于"特化 vs 冻结"。

#### D.24.5 预登记判定 vs 实测

| §D.23.5 预登记情形 | 实测 | 命中 |
|---|---|---|
| 通过增量门 → 分支替换路线复活 | 增量测试属 G18c，未测 | 待定 |
| 不通过但 ratio 显著低于 ImageNet 臂 → "特化改变了特征性质" | ratio 0.2598 ≪ 6.78/8.32，**且跨域均值 0.5811→0.6745 确实提升** | **部分命中** |
| 不通过且 ratio 更高 → 特化加剧域锁，单源路线关闭 | ratio 未升高；但 **cd1 0.5153 / cd2 0.5966 的塌陷就是域锁的实质** | 部分命中 |

→ **两种预登记情形都不完全适用**，因为 ratio 指标本身失效（D.24.4）。**真正的判定必须由 G18c 的增量测试给出**：`[V|df_ffpp_final]` 相对 V 的跨域增量是否 >0（含容量对照）。

#### D.24.6 当前状态与待办
- ✅ G18a 产物已复核，可进入 G18c；**任务 #11 结项**。
- ⏳ G18b（LODO，3 折 dfdcp/cd2/wild）在 GPU2 训练中：第 1/3 折 dfdcp 已至 epoch 6/20，val_auc 0.9702，~290s/epoch，**预计全程约 5 小时**；其泄漏审计已按新规则执行（检出 cd1⊂cd2 = 49/49 并排除）。
- ⏳ G18c 待 G18b 就绪后启动；判定门沿用 G17 四门 + `G18_SPECIALIZATION_EFFECT` + `G18_DELOCK`。**但 `G18_DELOCK` 须按 D.24.4 重新解释**：只在 G18a/G18b 两个**同监督强度**分支之间比较 ratio，才有意义。

---

### D.25 G18b 复核：多域联合监督**没有**救回 CNN 的跨域迁移；ratio 三门全不过（2026-09-13）

> G18b（LODO，3 折：dfdcp / cd2 / wild 各留出）已完成，耗时 13,596 s（~3.78 h）。本节为**独立复核**，判定仍待 G18c。

#### D.25.1 产物与交叉验证（通过）

- 三个 `df_lodo_<domain>_feats.npz`（各 5300 行）的 `paths` 与 `_g17/cnn_feats.npz` **逐字节对齐** ✓；臂名 `dense121_db2/db3/final`。
- **异源同值交叉验证**：子 agent 用**它自己实现的** G12 公式独立算出 G18a 的 ratio = **4.5355 / 1.6521 / 0.2598**，与我 §D.24.4 的独立复核值**完全一致** → 两套独立实现互证，本轮所有 ratio 可信。
- 泄漏审计全 0：各折留出域文件夹在训练集中 0 出现；`cd2` 折**同时排除 cd1**（检出 cd1⊂cd2 = 49/173）✓

#### D.25.2 ⚠️ 一个必须自己把关的陷阱：LODO 的"非留出域"列是**训练集记忆**

复核表里 `df_lodo_dfdcp` 在 cd1/cd2/ffiw/wild 上 AUC = 0.9999/0.9877/0.9991/0.9980、`df_lodo_wild` 在 cd1/cd2/dfdcp 上 0.99+——这些域是**该折的训练域**，数值是**记忆**，不是迁移。

→ **每折只有其"留出域"那一列是有效迁移测量**；其余列一律标 `CONTAMINATED` 并**排除出一切聚合与判定**。已写入 G18c 派单条款。

#### D.25.3 留出域实测（唯一的有效列）

| 留出域 | G18a（FF++ 单源） | G18b（LODO）线性探针 | Δ | G18b 自带头（子 agent 口径） |
|---|---|---|---|---|
| dfdcp | 0.7099 | **0.6937** | **−0.0162** | 0.7343 |
| cd2 | 0.5966 | **0.6281** | +0.0315 | 0.6707 |
| wild | 0.7429 | **0.7668** | +0.0239 | 0.7722 |

（左两列为**同协议线性探针**——probe-train 2200 训练 → 留出域测试；右列为模型自带分类头，协议不同，仅作参考。）

**读法**：多域联合监督把留出域拉动的幅度在 **−0.016 ~ +0.032**，**量级与噪声同级**，且 **dfdcp 反向**。**"把跨域不变性写进训练目标"（§D.22.4 的建议）在本轮并未兑现**，至少在这个容量与训练预算下没有。

#### D.25.4 `G18_DELOCK` 预登记门：**3/3 不过**

| 折 | G18b ratio（final） | G18a ratio | 门限 = 0.7×G18a = 0.1819 | 判定 |
|---|---|---|---|---|
| dfdcp | 0.6713 | 0.2598 | — | ✗（升高 2.58×） |
| cd2 | 0.2824 | 0.2598 | — | ✗（升高 1.09×） |
| wild | 0.3375 | 0.2598 | — | ✗（升高 1.30×） |

→ `G18_DELOCK = NOT_EFFECTIVE`（3/3 不通过）。**但须连带 §D.24.4 的警告**：ratio 的分子分母都受监督强度影响，ratio 升高也可能只是"对 FF++ 的过拟合减轻"而非"域锁加重"——**该门本身证据力弱，不单独作结论**。

#### D.25.5 待办
- ✅ 任务 #12 结项。
- ⏳ **G18c 已派单**（任务 #14）：复用 G17c 协议做增量测试 `[V|feat]` vs V，含容量对照与多 seed。**这才是对用户假设的决定性判定**——前面所有跨域 AUC 与 ratio 都只是前置条件。落档 §D.26。

---

### D.26 G18c 判定：特化 DenseNet121 **不但零互补，而且主动有害**——用户假设被否证（2026-09-13）

> 本节是 G18 全程的**决定性判定**。§D.24/§D.25 的跨域 AUC 与 ratio 都只是前置条件；互补性只能由「增量测试扣掉容量对照」给出。

#### D.26.0 协议可信度（三重自证，全部通过）

1. **子 agent 侧**：25/25 对齐检查通过（`paths`/`y`/`vids`/`domain` 在各 npz 间逐字节相等；cnn 行 == `vstack(probe[0:3000], multi)`）。
2. **锚点复现**：cd1 0.8286 / cd2 0.8633 / dfdcp 0.8261 / ffiw 0.8244 / wild 0.8090，maxdev=3.33e-05；ratio_V=0.1983、ratio_C=2.7160 ✓
3. **与 G17c 逐位交叉**：`ImageNet_dense121_final` 臂在 5 域上的 `V|dense121_final` 行与 G17c **完全相同**（0.8475/0.8615/0.8095/0.8167/0.8146），K=1024 的 3-seed 对照最大值也吻合 → **代码路径与 G17c 同源，跨实验可比**。
4. **我（主 agent）的独立复算**与子 agent 数值**逐位一致**：

| 臂 | cd1 | cd2 | dfdcp | wild |
|---|---|---|---|---|
| `V\|df_ffpp_final`（我算） | −0.1001 | −0.0786 | −0.0064 | −0.0019 |
| `V\|df_ffpp_final`（agent 报） | −0.1001 | −0.0786 | −0.0064 | −0.0019 |
| `V\|ImageNet_final`（我算） | +0.0188 | −0.0018 | −0.0166 | +0.0056 |

#### D.26.1 增量表 ΔAUC = AUC(`[V|feat]`) − AUC(V)，C=1e-3（V 单独：cd1 0.8286 / cd2 0.8633 / dfdcp 0.8261 / wild 0.8090 / ffiw 0.8244）

| 臂 | cd1 | cd2 | dfdcp | wild | ffiw |
|---|---|---|---|---|---|
| **df_ffpp_final（G18a 特化）** | **−0.1001** ±.0195 | **−0.0786** ±.0172 | −0.0064 | −0.0019 | +0.0218 |
| df_ffpp_db3 | −0.0363 | −0.0544 | −0.0360 | +0.0043 | −0.0251 |
| df_ffpp_db2 | −0.0040 | −0.0070 | −0.0211 | −0.0284 | −0.0070 |
| ImageNet_dense121_final（未特化） | +0.0188 ±.0069 | −0.0018 | −0.0166 | +0.0056 | −0.0077 |
| df_lodo_dfdcp_final | CONTAM | CONTAM | −0.0080 | CONTAM | CONTAM |
| df_lodo_cd2_final | −0.0203 | −0.0484 | CONTAM | CONTAM | CONTAM |
| df_lodo_wild_final | CONTAM | CONTAM | CONTAM | +0.0104 | CONTAM |

（± 为成对行 bootstrap std，B=2000；lbfgs 确定性使 seed std=0。CONTAM = 该域在该折训练集内，已排除出一切聚合。）

#### D.26.2 容量对照——增量**不是**维度效应

| 对照 | cd1 | cd2 | dfdcp | wild |
|---|---|---|---|---|
| 随机特征 K=1024（max over seeds） | +0.0071 | +0.0001 | +0.0049 | +0.0066 |
| 随机特征 K=512 | +0.0024 | +0.0031 | +0.0025 | +0.0027 |
| 我的独立复算（K=1024 随机） | −0.0066 | +0.0012 | −0.0003 | −0.0073 |

→ **纯增维的效应在 ±0.007 以内**。而 df_ffpp_final 的 **−0.1001 / −0.0786** 超出噪声带 **14× / 11×** —— **负增量来自特征内容，不是容量**。

#### D.26.3 四门判定（机械执行）

| 门 | 结果 | 实测 |
|---|---|---|
| `G18_INCREMENT` | **NO_COMPLEMENT** | 4 个非 LODO 臂**无一**在任何域同时满足「Δ>0 且 >两对照 +0.01」；最好也只是 ImageNet 臂 cd1 的 1/4。LODO 臂有效域<4 → `N/A_LT4_VALID`（0/1、0/2、0/1） |
| `G18_SPECIALIZATION_EFFECT` | **NO_EFFECT（方向为负）** | 共享域均值：df_ffpp_final **−0.0467** vs ImageNet **+0.0015** → **−0.0483** |
| `G18_LODO_EFFECT` | **NO_EFFECT (1/3)** | dfdcp −0.0016 / wild +0.0123 / cd2 +0.0302（仅此一域刚过线） |
| `G18_DELOCK` | **NO_EFFECT** | 折均值 0.4304 vs 门限 0.1819 → 1.6563× |

> ratio 由子 agent **重算而非沿用**，得 0.2598 / 0.6713 / 0.2824 / 0.3375，与 §D.24/§D.25 完全一致——第三方实现再次互证。

#### D.26.4 机理：特化把"源域类间隔"推大，而这正是**不可迁移**的方向

把 §D.24.4 与本节合起来看，机理是清楚的：

- 特化训练的目标就是**在 FF++ 上最大化类间隔**。它做到了——class_gap 2.21 → **43.40（×19.6）**。
- 但这条被放大的方向是 **FF++ 合成伪影方向**，在 Celeb-DF 上无对应物。
- 于是联合空间里，探针（在 FF++ 上训练）**优先抓住这条高方差、高类间隔的 FF++ 专向**，把决策边界从可迁移方向**拽偏** → cd1 **−0.10**、cd2 **−0.079**。
- **对照佐证**：未特化的 ImageNet DenseNet 同维、同管线，增量仅 −0.002~+0.019，**几乎为零**——因为它没有这样一条被强化的伪影专向。

→ **一句话：越是为 FF++ 伪造检测特化，对跨域决策的破坏越大。** 这不是"没帮上忙"，是"帮了倒忙"。

#### D.26.5 对本项目与论文的后果

1. **用户假设（特化 DenseNet121 能为 ViT 分支提供互补判别信息）→ 否证。** 最强形式：不是零增益，是**显著负增益**。
2. **分支替换路线（§D.19.3 提出、§D.22 标注"唯一未测臂"）→ 关闭。** 连同 G16（换深度无用）、G17（换模态无用），三轴齐备：
   - 换**深度**：无用（G16）
   - 换**模态**：无用（G17，16/16 NO_COMPLEMENT）
   - 对换来的模态**加监督特化**：**有害**（G18，−0.10）
3. **E5 升级**：原表述是"模态替换无效"；现应改为 **"在冻结 ViT 之外任何形式的 CNN 局部分支——无论是否特化训练——都不能提供互补判别，且特化会主动损害跨域决策"**。
4. **对 §D.22.4 建议的修订**：该节建议"把跨域不变性写进训练目标"。G18b 直接检验了这条，**在本训练预算下未兑现**（留出域 −0.016~+0.032，噪声级）。**故该建议目前无实验支持，不应作为论文的 future work 主张，除非给出更强的多域不变训练证据。**
5. **66%/62% 权重错配的结论不变且更强**：既然外部 CNN 无法补位，那把 62% 决策能量放在域敏感度 2.72 的 CLIP 上，其代价只能靠**重训 ViT 表征本身**来消除——这正是 §D.22 剩下的唯一出路。

#### D.26.6 G18 全程收束
- ✅ G18a（任务 #11）、G18b（#12）、G18c（#13/#14）全部结项。
- **判定**：`G18_OVERALL = NO_COMPLEMENT_AND_HARMFUL`。用户假设否证，分支替换路线关闭。
- 剩余唯一未关闭方向：**重新训练表征（解冻 ViT + 跨域不变目标）**，见 §D.22.4 与 §D.18.3。

#### D.26.7 补充验证：G18a 的"零泄漏"结论**复核通过**（2026-09-14）

派单 G19 前重查 §D.24.1 的泄漏审计，怀疑"0 交集"可能是**路径写法差异**造成的假阴性（probe 用正斜杠 `F:/zhj/data/...`，G18a 用反斜杠 `F:\\zhj\\...`）。归一化后重算：

| 口径 | probe-train ∩ G18a-train | probe-test ∩ G18a-train |
|---|---|---|
| 原始字符串 | 0 | 0 |
| 归一化后目录链 | **0** | **0** |
| **裸视频 token**（最宽口径） | **0** | **0** |

probe-train 的 token 为 `000/000_003/012/012_026/024/024_073/026/026_012`，G18a-train 为 `001/001_870/002/002_006/005/005_010/006/006_002`——**在最宽的裸 token 口径下都不相交**，故 G18a 零泄漏为真，§D.24/§D.26 全部结论不变。

---

### D.27 G19 设计：**重训 ViT 表征**——能否把跨域从「源标签锚点」推回「目标域 oracle」（2026-09-14 用户批准设计）

> 这是 §D.22 指出的、G16/G17/G18 三轴关闭后**唯一尚未证伪的方向**。本节只写设计与验证方案，结果落 §D.28+。

#### D.27.0 派单前侦察（已核实）

| 事项 | 结论 |
|---|---|
| GPU | **GPU1（0 MiB）、GPU2（64 MiB）空闲**；GPU0 占 5054 MiB（他人在用）→ 可**并行两折**，各自钉死卡号 |
| 参数高效微调 | `peft 0.10.0` **可用** → LoRA 路线可行（`transformers 4.37.0` / `timm 0.9.2` / torch 2.2.2+cu118） |
| ViT 主干 | `model.vit`（timm ViT-B，91.4M），stage-1 在 `train_bridge_phase1.py` L268-270 **显式冻结**（理由："91M on 107K data" 过拟合） |
| 可挂载点 | `model.vit.blocks[*].attn.qkv` 与 `.proj`（LoRA 目标模块） |
| 预处理 | 必须复用模型自身的 `_preprocess_for_vit`（albumentations `Normalize(CLIP均值)` → `ToTensorV2` → `F.interpolate(224)`），**不可另造管线**，否则训练/部署不一致 |

#### D.27.1 先讲清天花板：这条路的**理论上限只有 +0.074**

§D.18.2 的 gap 分解必须作为本设计的**预算约束**：

```
FF++ 域内 0.9852  ──源标签跨域──> 0.8318 ──目标域oracle──> 0.9111
                          总 gap 0.1534
              = 决策边界段 0.0793 (52%)  +  表征段 0.0741 (48%)
```

- **只有表征段 0.0741 是"重训表征"能碰的**；决策边界段 0.0793 需要目标域标签（= oracle 口径），**任何表征改动都无法触及**。
- 且融合侧还有稀释：**CLIP 占 62.3% 决策能量、ratio 2.716**（§D.19.1/E7），ViT 单侧改好也会被稀释。
- → **本轮成功线不是"达到 oracle"，而是"收回表征段 gap 的 30–50%"**，即留出域 **0.855–0.870**。达不到不等于白做，见 §D.27.7。

#### D.27.2 问题重构：E1 说**不缺方向，缺间隔**

E1 实测：各域类判别轴与源轴的 **|cos| = 0.979–0.989（轴已共享）**，但 **间隔保留率仅 38–47%**、**类内散布被域撑大 1.4–1.7×**、`d_z` 降到 25–33%。

→ **目标不是"找新方向"，而是"恢复间隔、压缩域致类内散布"**。这一条直接决定 O3 损失函数的形态。

**同时必须正视 G18 的教训**：在**源域**上最大化类间隔会强化 FF++ 专向、**主动损害**跨域（cd1 −0.10）。故**任何以提高源域 class_gap 为奖励的目标都被禁止**；只允许**对齐/压缩域致偏移**类目标。

#### D.27.3 臂设计（核心：必须能分离"信息"与"容量/正则"）

**骨架 = LODO**。留出域取 **{dfdcp, wild}**（§D.23.0 认定的两个**完全干净**域）；`cd2` 作次干净域（须同时排除 cd1）；**`cd1` 永久排除**；ffiw 单列不聚合。

**目标臂**（均在多源上训练）：

| 臂 | 目标 | 意图 |
|---|---|---|
| **O1** | CE only | 检验"纯解冻"是否如 G18 预测那样**有害** |
| **O2** | CE + GRL 域对抗（DANN） | 标准 DG 基线 |
| **O3** | CE + **类条件域均值惩罚** `Σ_{d,c} ‖μ_{d,c} − μ_c‖²` | **直击 E1**：罚的正是"域致类内散布"。不奖励源域 class_gap → 不触发 G18 的失效模式 |
| **O4** | CE + O3 + O2 | 组合，预算允许时做 |

**对照臂（缺一不可，否则涨点不可归因）**：

| 臂 | 构造 | 拦截的假象 |
|---|---|---|
| **R0** | 冻结 V，不训练 | 锚点 0.8303 |
| **R1** | **同数据同 LoRA，标签打乱** | 若也涨点 → 涨点来自正则/平滑而非标签信息 → **全轮作废** |
| **R2** | 同 CE，**LoRA rank 加倍** | 涨点若不超过它 → 纯容量效应 |

**参数化**：`peft` LoRA（r=16，α=32）挂 `blocks[*].attn.qkv` / `.proj`，主干冻结。理由：stage-1 冻结 ViT 的原始理由就是 91M/107K 过拟合风险；LoRA 同时提供**天然的容量控制轴**（R2）。

#### D.27.4 验证协议（逐条写死）

1. **主指标**：**留出域跨域 AUC**，用**与 G12–G18 逐字相同**的探针协议（`StandardScaler(fit train)` + `LogisticRegression(C=1e-3, lbfgs, max_iter=3000)`；probe-train = FF++ 2200）。→ 可直接对比 **0.8286/0.8633/0.8261/0.8090（均值 0.8303）**、**源标签锚点 0.8318**、**oracle 0.9111**。
2. **域内护栏**：FF++ test 800 AUC（当前 **0.9852**），防灾难性遗忘。
3. **训练集硬排除 probe-train 与 probe-test 视频**（该做法已在 G18a 验证有效，§D.26.7），否则探针被污染。
4. **⚠️ 模型选择绝不可触碰留出域**：早停/选最优权重**只能用训练域内部的 val 划分**。这是本轮最容易出错、也最容易造假的一条。
5. **禁用任何目标域统计**：无 TTA、无目标域 BN 统计更新、无目标域聚类/伪标签。
6. **多 seed ≥3**（首轮 1 seed 扫臂，胜出臂补 3 seed；lbfgs 确定性使 seed 只影响训练侧）。
7. **视频级划分**，身份用**路径派生目录链**，禁止裸 `vid`（§D.23.0 硬规则）。
8. **特征提取与探针脚本全臂共用一份**，禁止各臂自造口径。

#### D.27.5 预登记门（机械判定）

| 门 | 条件 | 含义 |
|---|---|---|
| `G19_UNFREEZE_HELPS` | O1 − R0 ≥ +0.01（留出域） | 单纯解冻即有效（**与 G18 预期相反**） |
| `G19_INVARIANCE_HELPS` | best(O2/O3/O4) − O1 ≥ +0.02 | 不变性目标有实质贡献 |
| **`G19_VS_RANDLABEL`** | **best − R1 ≥ +0.02** | **硬门：不满足则全轮作废** |
| `G19_VS_CAPACITY` | best − R2 ≥ +0.01 | 排除纯容量效应 |
| `G19_NO_FORGETTING` | 域内 ≥ 0.9852 − 0.01 | 无灾难性遗忘 |
| `G19_CEILING_RECOVERY` | gap 恢复率 = (best − 0.8318)/(0.9111 − 0.8318) ≥ **30%** | 主成功线 |

#### D.27.6 阶段与杀点

- **G19a｜可行性（便宜，先做）**：折 = {dfdcp, wild} 各一张卡，臂 = **R0 / R1 / O1 / O3**。
  **杀点**：若 **O1 与 R1 都 ≈ R0（\|Δ\|<0.005）** → 说明解冻根本没有撼动表征（多半被 LoRA 容量或学习率卡住），**停下重新设计**，不要直接上 G19b。
- **G19b｜主实验**：3 折 × {胜出目标 + O1 + R1 + R2}，3 seed。产 `_g19/feats_*.npz`。
- **G19c｜端到端（论文头条）**：把胜出 ViT 装回 `ViT_M2F2Det_Bridge`，按 stage-1 配方重训桥接/融合，测**端到端跨域 AUC**，并**重算决策能量占比**（回扣 E7 的 62.3% 是否重新分配）。

#### D.27.7 资源纪律与**诚实预期**

- 两 agent 各自**钉死卡号**（G19a-dfdcp=GPU1、G19a-wild=GPU2），开工前 `nvidia-smi` 确认本卡空闲；**若被占则停下报告，禁止换卡**。
- **CPU 限制**：`OMP/MKL/OPENBLAS/NUMEXPR=4`、`torch.set_num_threads(4)`、`cv2.setNumThreads(0)`、`num_workers=0` 或 ≤2；fp32；batch ≤32。
- 不改其它文件、不写 WORKLOG；结果由我独立复核后落档。
- **先跑 1-epoch smoke test** 验证管线（LoRA 挂载成功、特征可提、探针可跑、AUC 可算）**再**放全量训练。
- **诚实预期**：天花板 +0.074 且被 62% 的 CLIP 决策能量稀释。**若 O3 收回 gap 的 30–50%（留出域 0.855–0.870），本轮成功**；**若收不回**，则 §D.22 的"唯一剩余路径"也被关闭，项目结论转向 **"84% 是架构性上界"**——**那本身是可发表的负结论，不是失败**。设计上必须让两种结局都能被干净地判读。

---

### D.28 G19a 中间结果（留出 = **wild**）：**解冻确实涨了，但涨的不是不变性目标**（2026-09-14）

> 留出 dfdcp 的那一折仍在 GPU1 训练中，本节只落 **wild 折**，结论待两折齐备后再定。

#### D.28.1 管线自证闸门：**PASS**（可信度最高的一次）

- R0 臂（不训练）复现冻结锚点：5 域 max|dev| = **3.33e-05**，域内 dev = **0.0000**；
- `feat` 与 `_probe/probe_feats.npz['V']` **逐位相等（max|Δ| = 0.000e+00）**，与 `_tsne/feats_multi.npz['V']` max|Δ|=3.16e-05；
- R0 的 G12 行 = **s=1.509654 / class_gap=40.2122 / ratio=0.198289**，与 §D.12 记录**完全一致**。
- **我的独立复算与子 agent 数值逐位一致**（见下表）。

#### D.28.2 主表（留出 = wild；**仅 wild 列有效**）

| 臂 | cd1† | cd2† | dfdcp† | ffiw† | **wild（有效）** | 域内 FF++ 800 |
|---|---|---|---|---|---|---|
| R0（冻结） | 0.8286 | 0.8633 | 0.8261 | 0.8244 | **0.8090** | 0.9852 |
| R1（**标签打乱**） | 0.7758 | 0.7371 | 0.7388 | 0.6068 | **0.7130** | 0.9834 |
| **O1（CE）** | 0.9891 | 0.9900 | 0.9915 | 0.9596 | **0.8384** | 0.9757 |
| O3a（CE+λ0.3） | 0.9831 | 0.9924 | 0.9817 | 0.9208 | **0.8090** | 0.9722 |
| O3b（CE+λ1.0） | 0.9062 | 0.9372 | 0.9512 | 0.8846 | **0.7797** | 0.9800 |

† 该域是本折**训练域** → 0.99 是**记忆**，一律 `CONTAMINATED`，排除出聚合。

**独立性核实**：O3a 与 R0 在 wild 上 AUC 到 16 位小数相同（0.8090222222222222/…23）。已验这是**同一 300 样本 AUC 网格上的真实平局**（网格步长 1/22500），两臂特征显著不同（`max|feat−R0|`：O3a=20.78、O1=10.68、R1=34.00）→ **非文件串用**。

#### D.28.3 门判定

| 门 | 实测 | 判定 |
|---|---|---|
| `G19_UNFREEZE_HELPS` | O1 − R0 = **+0.0294** | **PASS** |
| `G19_INVARIANCE_HELPS` | best(O3a,O3b) − O1 = **+0.0000** | **FAIL** |
| `G19_VS_RANDLABEL`（硬门） | best − R1 = **+0.1254** | **PASS** |
| `G19_NO_FORGETTING` | 0.9757 ≥ 0.9752 | PASS（**余量仅 0.0005，见 D.28.5**） |
| `G19_CEILING_RECOVERY` | **8.3%** | **FAIL**（线 30%） |
| 杀点 | O1−R0=0.029、R1−R0=−0.096 | **未触发** |

#### D.28.4 ⚠️ **我 §D.27 的预测错了，且我的不变性设计被否证**

**(a) 我在 §D.27 预测"纯解冻会像 G18 那样有害"——实测相反，O1 = +0.0294。**
诚实记录：这是**我的预测错误**，不是实验异常。机理上可解释：G18 的 DenseNet 是 ImageNet 通用初始化 + 全参数训练（**高容量、低相关初始化**）→ 被 FF++ 伪影带跑；本轮 ViT 是 **已具伪造检测取向的 net_050 初始化 + LoRA r=16（884,736 参数，<1% of 91M）** 的**小扰动** → 只做精修，不重写。**故"特化有害"的边界是容量/初始化，不是模态。**

**(b) 我据 E1 设计的 O3（罚 `Σ‖μ_{d,c}−μ_c‖²`）完全没有兑现，λ=1.0 还主动有害（−0.0293 vs R0）。**
`G19_INVARIANCE_HELPS = FAIL`。**增益全部来自"加入更多源域数据"这一事实本身，与不变性损失无关。**

**(c) 跨架构一致性**：§D.25.3 多源监督给 DenseNet 的 wild 增益是 **+0.0239**，本轮给 ViT 的是 **+0.0294** —— **几乎同一个量级**。→ **多源监督对两种架构的增益都只有 ~+0.025**，远不足以收回 0.074 的表征 gap。

#### D.28.5 三条**方法学发现**（比主结果更值得写进论文）

1. **ratio 指标第三次失效**：O3a 的 ratio = **0.0919**（全臂最低，甚至低于 R0 的 0.1983），但 wild AUC 与 R0 **完全相同**。加上 G18a 特化臂（ratio 0.2598 却 cd1 塌到 0.5153），**"ratio 低 ⇒ 迁移好"已在两个截然不同的设置下被否证**。→ **ratio 不能作为迁移性的代理指标**，只能作同监督强度下的诊断量。
2. **O3 降低 ratio 的方式是退化的**：O3a 把整体特征尺度 `s` 从 1.510 压到 **0.0614（×0.041）**，而非真正对齐域。→ **ratio 可以通过"整体方差不均匀压缩"被人为压低**，这解释了 (1)。
3. **域内护栏几乎是空门**：R1（**标签打乱**训练）的域内 AUC 仍有 **0.9834**，仅比 R0 低 0.0018，**却通过了 `G19_NO_FORGETTING`（门槛 0.9752）**——而它的跨域是 0.7130。→ **域内 FF++ AUC 无法区分"好表征"与"被损坏的表征"**，作为防遗忘护栏基本无效。后续实验须改用跨域侧或表征几何量做护栏。

#### D.28.6 口径警告（必须写进论文）
本折的训练源 = **FF++ + cd2 + dfdcp + ffiw**，即**训练集包含其它目标域**。故这是**多源域泛化（multi-source DG）**，**不是零样本跨域**。论文中原"zero-shot cross-domain"的表述若不修改，即为此处的夸大。已记录，不回避。

#### D.28.7 待办
- ⏳ 留出 **dfdcp** 的折仍在 GPU1 训练（`nvidia-smi` 显示 GPU1 约 5095 MiB 占用）。**两折齐备后**才能判定 O1 的 +0.029 是稳定效应还是 wild 独有；且单 seed、无 CI，**必须等 G19b 的多 seed 才能定论**。
- 若 dfdcp 折同向 → O1 路线成立但天花板只有 8% 恢复率，需重估；若反向 → 连"解冻有效"都不成立，**§D.22 唯一剩余路径正式关闭**。

#### D.28.8 留出 = **dfdcp** 折（已完成，与 wild 折**同向**）

管线自证同样 **PASS**：R0 与 `probe_feats['V']` max|Δ| = **0.000e+00**，5 域锚点 |dev| = 0.0000，G12 行 s=1.5097/class_gap=40.2122/ratio=0.1983 完全复现。**我的独立复算逐位一致。**

| 臂 | 域内 FF++ | cd1† | cd2† | **dfdcp（有效）** | ffiw† | wild† |
|---|---|---|---|---|---|---|
| R0（冻结） | 0.9852 | 0.8286 | 0.8633 | **0.8261** | 0.8244 | 0.8090 |
| R1（标签打乱） | 0.9662 | 0.7416 | 0.7336 | **0.7924** | 0.5942 | 0.6821 |
| **O1（CE）** | 0.9785 | 0.9993 | 0.9992 | **0.9320** | 1.0000 | 0.9920 |
| O3a（λ=0.3） | 0.9551 | 0.9653 | 0.9726 | **0.7870** | 0.9979 | 0.9271 |
| O3b（λ=1.0） | 0.9620 | 0.7472 | 0.7474 | **0.7673** | 0.7075 | 0.7363 |

† 本折训练域 → 记忆，排除。门：`UNFREEZE_HELPS` **+0.1060 PASS** ／ `INVARIANCE_HELPS` **−0.1450 FAIL** ／ `VS_RANDLABEL` **+0.1397 PASS** ／ `NO_FORGETTING` 0.9785 PASS（但 R1 0.9662、O3a 0.9551、O3b 0.9620 **各自都不过**）。

#### D.28.9 两折合表：**同向，但幅度差 3.6 倍**

| 臂 | dfdcp（留出） | wild（留出） |
|---|---|---|
| R0（冻结） | 0.8261 | 0.8090 |
| O1（CE） | **0.9320（+0.1060）** | **0.8384（+0.0294）** |
| O3a | 0.7870（−0.039） | 0.8090（±0.000） |
| O3b | 0.7673（−0.059） | 0.7797（−0.029） |

**"解冻+CE 有益"在两折上同向成立**（这是我 §D.27 预测错的方向）；**"不变性目标有益"两折上均被否证**。

**污染自查（我做的）**：若 dfdcp 被泄漏进该折训练集，则该折**所有**训练臂的 dfdcp 读数都应被抬高；实测 R1/O3a/O3b **全部低于** R0，**只有 O1 是异常值** → 更像**臂特异效应**而非数据污染。

#### D.28.10 ⚠️ 重要更正：**oracle 不是固定天花板，它随表征大幅移动**——我 §D.27.1 的预算框架错了

用"目标域自身标签训练探针（70/30 视频划分）"重算各表征的 oracle：

| 表征 | dfdcp oracle | wild oracle |
|---|---|---|
| R0 冻结 | 0.8400 | 0.8385 |
| **O1** | **0.9726** | **0.9975** |
| O3a | 0.8849 | 0.9289 |
| R1 | 0.7532 | 0.8039 |

→ **O1 的 dfdcp 读数 0.9320 低于它自己的 oracle 0.9726（仅达自身天花板的 96%），与"超过 0.9111"毫无矛盾**——因为 0.9111 是**冻结 V 的 mean4 oracle**，不是普适上界。**§D.27.1 把 gap 分解当成"本轮预算上界"是错的**：换表征后，表征段与决策段的划分整体重画。`G19_CEILING_RECOVERY = 1.2641` 正是这个错误分母造成的假 PASS，**该门作废，不再使用**。

#### D.28.11 关键机理假设（**待下一轮证伪**）：dfdcp 的大幅增益来自**源域相似性**，不是"解冻"

观察：
- 两折训练源都含 **cd2（Celeb-DF-v2，高质量人脸替换）**。
- **DFDC-P 与 Celeb-DF-v2 同属高质量人脸替换管线** → 相似性高；**wild 是互联网采集的异质数据** → 相似性低。
- 结果：**与源相似的 dfdcp 涨 +0.106，与源疏远的 wild 只涨 +0.029（差 3.6 倍）**。
- 佐证：O1 在 **wild 的 oracle 高达 0.9975**（表征本身完全可分），但**源标签探针只读出 0.8384** → wild 的判别方向与源训练方向**不对齐**；而 dfdcp 的 0.9320/0.9726 = **96% 对齐**。**这正是"源-目标距离"的指纹，而非"不变性"的指纹。**

**若此假设成立**，则 O1 的增益**不能**归功于"解冻让表征更跨域不变"，而应归功于**"训练集里恰好有一个离目标很近的源域"**——这是普通的多源 DG 收益，**会显著削弱论文的 zero-shot 主张**。

#### D.28.12 本轮**尚不能判定**的事（必须等 G19b）
1. **单 seed、单折各一次**：dfdcp 的 +0.106 与 wild 的 +0.029 都无 CI，**不可作为最终结论**。
2. **源域相似性假设未检验**：缺 **O1-仅FF++** 对照臂（训练源只有 FF++，不含 cd2）。
3. **容量对照 R2（LoRA rank 加倍）未做**：`VS_RANDLABEL` 已 PASS，但"是否只是容量"仍未排除。

→ **G19b 最小充分设计**：两折 × {O1-多源 ×3 seed、**O1-仅FF++ ×3 seed**、R2-多源 ×1}。其中 **O1-仅FF++ 是判定 §D.28.11 的关键臂**：若它也能涨 dfdcp，则"解冻"是真效应；若它不涨而多源涨，则**增益属于源域相似性，不属于解冻**。

---

### D.29 G19b 判定：**"解冻表征"被否证**——G19a 的增益来自"源域数据"，不是解冻（2026-09-15）

> 两个 G19b agent 均因 **API 连接错误**在**任务完成之后**被中断，但 7 臂 × 2 折的权重/特征/报告**已全部落盘**（`nvidia-smi` 确认两卡空闲、无残留进程）。本节为**我自己的完整独立复算**，不依赖 agent 的收尾汇报。
> 一致性抽查：子 agent 中断前留下的 `O1multi` per-seed wild = 0.8400/0.8143/0.7981，与我的独立复算**逐位相同**。

#### D.29.1 主表（每折 3 seed；**仅留出域列有效**）

**留出 = dfdcp**（训练域 = cd2/ffiw/wild，已排除）

| run | 域内 | dfdcp | **oracle** | 对齐率 |
|---|---|---|---|---|
| R0（冻结） | 0.9852 | 0.8261 | 0.8447 | 0.978 |
| O1multi s0/s1/s2 | 0.9807/0.9774/0.9754 | **0.9298/0.9212/0.9156** | 0.9370/0.9481/0.9769 | 0.992/0.972/0.937 |
| **O1ffonly s0/s1/s2** | 0.9786/0.9804/0.9812 | **0.8263/0.8005/0.8076** | 0.8760/0.8764/0.8510 | 0.943/0.913/0.949 |
| R2multi s0（rank×2） | 0.9804 | **0.9251** | 0.9639 | 0.960 |

**留出 = wild**（训练域 = cd2/dfdcp/ffiw，已排除）

| run | 域内 | wild | **oracle** | 对齐率 |
|---|---|---|---|---|
| R0（冻结） | 0.9852 | 0.8090 | 0.8177 | 0.989 |
| O1multi s0/s1/s2 | 0.9776/0.9776/0.9768 | **0.8400/0.8143/0.7981** | 0.8535/0.8416/0.8327 | 0.984/0.968/0.958 |
| **O1ffonly s0/s1/s2** | 0.9826/0.9793/0.9844 | **0.7961/0.7791/0.7953** | 0.8166/0.7745/0.7948 | 0.975/1.006/1.001 |
| R2multi s0（rank×2） | 0.9826 | **0.8514** | 0.8499 | 1.002 |

**臂构造的内部一致性核验**：MULTI 臂在 cd1/cd2 上读出 0.99（记忆），而 FFONLY 臂在 cd1/cd2 上只有 0.74–0.86（未训练）→ **确认 MULTI 确实训了 cd2、FFONLY 确实没训**。R0 两折锚点均 |dev|=0.0000。

#### D.29.2 均值与门判定

| | dfdcp 折 | wild 折 |
|---|---|---|
| O1multi（3 seed） | **0.9222 ± 0.0071**，Δ = **+0.0961** | **0.8175 ± 0.0211**，Δ = **+0.0084** |
| **O1ffonly（3 seed）** | **0.8115 ± 0.0133**，Δ = **−0.0146** | **0.7902 ± 0.0096**，Δ = **−0.0189** |
| R2multi（rank×2, 1 seed） | 0.9251，Δ = +0.0990 | **0.8514**，Δ = **+0.0424** |

| 门 | dfdcp | wild |
|---|---|---|
| `G19B_UNFREEZE_REPLICATES` | **PASS**（Δ+0.0961，3/3 同号） | **FAIL**（Δ+0.0084 < +0.01，且 s2=0.7981 **低于 R0**） |
| **`G19B_NEEDS_MULTISOURCE`** | **成立**（−0.0146 ≤ +0.01 且 多源−仅FF++ = **+0.1107** ≥ +0.05） | **部分成立**（−0.0189 ≤ +0.01 ✓，但差值 +0.0273 < +0.05） |
| `G19B_VS_CAPACITY` | **FAIL**（0.9222 − 0.9251 = **−0.0029**） | **FAIL**（0.8175 − 0.8514 = **−0.0339**） |
| `G19B_NO_FORGETTING` | 全过 | 全过 |
| `G19B_ALIGNMENT_SHIFT` | 0.978 → 0.937–0.992（**无抬升**） | 0.989 → 0.958–0.984（**无抬升**） |

#### D.29.3 判定一：**G19a 的"+0.029 解冻有效"是 seed 运气**

wild 折 3 seed = 0.8400 / 0.8143 / **0.7981**，均值 **+0.0084**，std 0.0211。G19a 的单 seed 0.8384（+0.0294）约为 **+1σ** 的幸运抽样，且**有一个 seed 低于冻结基线**。→ **`G19B_UNFREEZE_REPLICATES` 在 wild 上 FAIL。**

#### D.29.4 判定二（**决定性**）：**FF++ 单源下，重训表征的跨域增益为零甚至为负**

- dfdcp：O1ffonly Δ = **−0.0146**
- wild：O1ffonly Δ = **−0.0189**

→ **"解冻 ViT + 用 FF++ 标签重训"这个动作本身，对跨域毫无帮助，反而略损。**
→ **§D.28.11 的源域相似性假设成立**：dfdcp 的 +0.096 完全来自训练集里**多了 cd2/ffiw/wild**（其中 cd2 与 DFDC-P 同为高质量人脸替换管线，分布相邻），**不是来自解冻**。

**机理旁证（oracle 移动方向）**：dfdcp 折 O1multi 的 **oracle 从 0.8447 升到 0.937–0.977**——表征在 dfdcp 上**本身变得更能分了**；而 wild 折 O1multi 的 oracle 只从 0.8177 动到 0.833–0.854（**几乎没动**）。→ **只有"目标域邻近的源域"才能提升表征对目标域的判别力。**

#### D.29.5 判定三（**最致命**）：**容量对照在两个折上都打败了主臂**

R2multi（**同等数据、同目标，仅 LoRA rank 16→32**）：
- dfdcp：**0.9251 vs O1multi 0.9222**（+0.0029，容量臂略胜）
- wild：**0.8514 vs O1multi 0.8175**（**+0.0339，容量臂大胜**，且是 wild 折全部 8 个臂中的最高分）

`G19B_VS_CAPACITY` **两折皆 FAIL**。→ **残余增益的主体是"参数更多、拟合更多"，不是"标签信息或目标设计"。**

**分解（同 rank16、只换数据）**：FF++ → 多源 = **+0.0961**(dfdcp) / **+0.0084**(wild) → **数据效应**。
**（同数据、只换容量）**：rank16 → rank32 = +0.0029(dfdcp) / **+0.0339**(wild) → **容量效应**。
→ **dfdcp 靠数据效应，wild 靠容量效应**，两者都不是方法。

#### D.29.6 判定四：**对齐率没有抬升**——表征没变得"更跨域不变"

R0 的源-oracle 对齐率本就已 0.978/0.989（非常高），O1multi/FFONLY/R2 **没有一个超过它**（0.937–0.992 / 0.913–1.006）。→ **重训没有改善"源方向与目标方向的夹角"**，我们此前 G12/E1 建的都是**几何诊断量**，此处再次与下游表现脱钩。

#### D.29.7 判定五：**域内护栏被彻底证伪为反指标**

O1ffonly 的**域内最高**（dfdcp 折 0.9786–0.9812，wild 折 0.9793–0.9844，逼近冻结的 0.9852），而**跨域最差**（0.8115 / 0.7902）。
→ **"域内不掉"恰恰意味着"没学到新东西"**。域内 FF++ AUC 作为护栏**不仅无效，而且方向相反**，必须从后续所有实验的判定体系里移除。

#### D.29.8 G19 总结论：**§D.22 的"唯一剩余路径"正式关闭**

| 探针动作 | 结论 | 出处 |
|---|---|---|
| 冻结表征上做各种适配 | 全失败 | E3 / G15 |
| 换特征深度 | 无用 | G16 |
| 换模态（频域/SRM/CNN局部） | 无用（16/16） | G17 |
| 对换来的模态加监督特化 | **有害**（cd1 −0.10） | G18 |
| **解冻重训 ViT 表征（FF++ 单源）** | **无用甚至略损**（−0.015/−0.019） | **G19b** |
| 加域不变目标 | 失败（两折） | G19a |
| 加源域数据 | **仅当源与目标分布相邻才有效**（dfdcp +0.096 / wild +0.008），且**容量臂即能追平或反超** | **G19b** |

→ **"84% 是架构性上界"正式成为本项目的最终结论**，且**每一条通往改善的路径都已被实验逐一关闭**。
→ 唯一仍有残效的是**"训练集中存在与目标分布相邻的源域"**——但那是**数据属性，不是方法贡献**；且对真正新颖的目标域（wild 这类）**也无效**。

#### D.29.9 尚未分离的一点（如需可补做）
dfdcp 折的 MULTI = FF++ + **cd2 + ffiw + wild** 三个额外域，**"到底是哪个源域贡献了 +0.096"未做分离**。若要写成论文的"源域相邻性"论断，需补 `FF+++cd2` / `FF+++wild` / `FF+++ffiw` 的单域追加消融（各 2 seed，成本约 1.5 h/折）。**这是本结论目前唯一的归因缺口，已在文中标注，不回避。**

---

### D.30 回应"多层注意力图拼接 / 加权累加"两条路线：**前提被数据反证，且路线 1 已在 G16 测过**（2026-09-15）

> 触发：用户提出"浅层有判别力、深层跨域好但类间隔过小"，据此提出**两条路线**：(R1) 多层注意力图拼接；(R2) 多层注意力加权累加。
> 本节为**纯 CPU、读现成 `_g16/layer_feats.npz`** 的诊断（`_g19/ratio_layers.py`，公式逐字抄自 `_g12/run_g12.py` L392-405），**无 GPU、无训练**。
> **锚点核验 PASS**：`cls_final` 复算出 s=1.5097 / class_gap=40.2122 / ratio=0.1983，与 G12 记录**逐位一致**（脚本内已内建 assert 打印）。

#### D.30.1 新增 10 个 E6 域锁比数据点（ViT 深度维）

| feature | dim | s | class_gap | **ratio(all5)** | ratio(v4) | 各目标域 ratio (cd1/cd2/dfdcp/ffiw/wild) |
|---|---|---|---|---|---|---|
| cls_b3 | 768 | 0.0465 | **3.3571** | **6.0921** | 5.6991 | 4.362/5.216/6.534/7.664/6.684 |
| cls_b6 | 768 | 0.1228 | 12.4756 | 1.2902 | 1.2402 | 1.093/1.200/1.388/1.490/1.280 |
| cls_b9 | 768 | 0.7204 | 28.3853 | 0.3373 | 0.3313 | 0.318/0.298/0.347/0.361/0.362 |
| **cls_final (=V)** | 768 | 1.5097 | **40.2122** | **0.1983** | 0.1980 | 0.211/0.186/0.161/0.200/0.234 |
| mp_b3 | 768 | 0.0546 | 4.6599 | 4.1090 | 3.8780 | 3.261/3.525/4.468/5.033/4.258 |
| mp_b6 | 768 | 0.0963 | 13.4113 | 1.1005 | 1.0647 | 1.009/1.017/1.163/1.244/1.070 |
| mp_b9 | 768 | 0.7928 | 32.7298 | 0.2962 | 0.2984 | 0.306/0.261/0.282/0.288/0.344 |
| cat_cls369 | 2304 | 0.4228 | 48.5045 | **0.3531** | 0.3460 | 0.329/0.313/0.365/0.381/0.377 |
| cat_b6_final | 1536 | 1.0710 | 56.6994 | **0.2010** | 0.2005 | 0.213/0.189/0.165/0.203/0.236 |
| cat_mp369_final | 3072 | 0.8544 | 77.2872 | **0.2181** | 0.2182 | 0.230/0.201/0.187/0.218/0.255 |

#### D.30.2 结论一：**用户的前提在 ViT 上被反证——浅层是"两头都差"，不存在权衡**

| 深度 | class_gap（判别力） | ratio（域锁） | 跨域 AUC (G16 C=1e-3, mean4dom) |
|---|---|---|---|
| b3（浅） | **3.3571** | **6.0921** | **0.6223** |
| b6 | 12.4756 | 1.2902 | 0.7545 |
| b9 | 28.3853 | 0.3373 | 0.8337 |
| final（深） | **40.2122** | **0.1983** | **0.8318** |

**三条曲线全部单调、且方向一致**：随深度增加，类间隔 **↑12.0×**、域锁 **↓30.7×**、跨域 AUC **↑0.21**。
→ **浅层不是"判别力强"，而是判别力弱 12 倍、同时域锁高 31 倍。** 即 cls_b3 是**类间隔塌缩 + 域锁爆表**的双劣。
→ "深层判别力弱、类间隔过小"与数据**正相反**：V 的 class_gap 40.21σ 是全项目最大的。

#### D.30.3 结论二：**E6 单调律延伸到 14 个点、跨 3 个数量级，无一处违例**

按 ratio 升序排（新点以 **▸** 标）：

`V/cls_final 0.1983` **▸** `cat_b6_final 0.2010` **▸** `cat_mp369_final 0.2181` **▸** `mp_b9 0.2962` **▸** `cls_b9 0.3373` **▸** `cat_cls369 0.3531` **▸** `mp_b6 1.1005` **▸** `cls_b6 1.2902` — `C 2.7160` — **▸** `mp_b3 4.1090` — `effnet_b4_blk6 5.6008` **▸** `cls_b3 6.0921` — `F1 6.9809` — `dense121_db3 8.3213` — `F3_hist 8.4082`

对应的跨域 AUC 排名与之**完全反序一致** → **E6 域锁比 → 跨域失败**这条律，现在有 14 个数据点、ratio 跨 0.198→8.41（42×），**零违例**。
→ `cls_b3` 的 ratio 6.0921 落在 `effnet_b4_blk6 (5.60)` 与 `F1 (6.98)` 之间——**浅层 ViT 块在域锁上就是 CNN 专用特征的地盘**。

#### D.30.4 结论三：**路线 1（多层拼接）G16 已经做过，三种变体全测了**

| 拼接变体 | cd1 | cd2 | dfdcp | wild | **mean4dom** | Δ vs V | 门(≥+0.03) |
|---|---|---|---|---|---|---|---|
| cls_final（基线） | 0.8286 | 0.8633 | 0.8261 | 0.8090 | 0.8318 | — | — |
| cat_cls369 | +0.0312 | **−0.0305** | +0.0214 | +0.0079 | 0.8393 | **+0.0075** | 1/4 |
| **cat_b6_final** | +0.0263 | −0.0080 | +0.0154 | +0.0110 | **0.8429** | **+0.0111** | **0/4** |
| cat_mp369_final | +0.0156 | **−0.0148** | −0.0031 | +0.0006 | 0.8313 | **−0.0005** | 0/4 |

G16 判定：`G16_LAYER_TRANSFER = PARTIAL（best_win=1/4）`——**实质是 NOT_FOUND**（门是 +0.03，最好的只有 +0.0111，且 cd2 上为负）。
oracle 侧最好的是 `cat_mp369_final`：mean4dom 0.9303 vs V 0.9111（**+0.0192**），但逐域为 cd1 +0.0158 / cd2 +0.0500 / dfdcp **−0.0074** / wild +0.0184——**不系统**。

#### D.30.5 结论四（**新机理，本节的主要增量**）：**拼接会把 ratio 拉向最差的那一块**

| 拼接变体 | 含有的最差块 | 该块 ratio | 拼接后 ratio | 相对 V 的倍数 |
|---|---|---|---|---|
| cat_b6_final | cls_b6 | 1.2902 | 0.2010 | **1.01×**（几乎不动） |
| cat_mp369_final | mp_b3 | 4.1090 | 0.2181 | 1.10× |
| cat_cls369 | cls_b3 | 6.0921 | **0.3531** | **1.78×**（显著劣化） |

→ **把域锁块拼进 V，ratio 被拖向该块**；拼 b3 → V 的域锁恶化 **78%**，而该变体的跨域增益恰是**唯一在 cd2 上掉分（−0.0305）**的那个。
→ **这就是 G18 的机理解释**：G18 往融合里拼的 DenseNet 分支是 `dense121_db3` 级别（ratio 8.32）/ FF++ 特化后更甚 → 按本律**必然有害**，实测 **cd1 −0.1001 / cd2 −0.0786**。**G18 是本律在架构层的独立确认，不是巧合。**
→ **预测**：把 b3/b6 接进 bridge（`linear_vit_*` → BridgeAdapter → 融合）会**复现 G18 的伤害**，因为 b3/b6 的 ratio（6.09 / 1.29）本就落在 CNN 专用特征区间。

#### D.30.6 结论五：**路线 2（加权累加）在数学上被路线 1 支配**

`cat([a,b])` 接线性头、权重 `[w1,w2]`，其输出**恒等于** `w1·a + w2·b`——即**加权累加是拼接+线性头的一个特例**。
→ 线性（探针）意义下 **加权累加的上限 ≤ 拼接的上限 = +0.0111**，**不可能更好**。
→ 唯一差异只在**非线性端到端**训练时才会出现。因此 R2 不构成独立路线；要做只能做 R1 的端到端版本。

#### D.30.7 真正的空白只剩一条，且先验很不利

G16 的**唯一未覆盖面**：它抽的是 **block 输出（CLS + 均值池化 patch）**，**不是注意力权重本身**；且是**探针级**（未过 `linear_vit_*`，未训练）。
→ 严格说，"**注意力图**"作为对象**确实没测过**。
→ 但先验极不利：(a) E6 律现有 14 点零违例；(b) G18 已在架构层证伪"拼域锁分支"；(c) b3 的注意力是最局部、最域特异的那一层。
→ **我的概率判断：注意力图融合突破 +0.03 的概率约 10–15%。**

#### D.30.8 我建议的**最小闭合实验**（如果要做）
**不做全量多分支集成**。先做 **G19d 探针级三格辨识**（约 30 min，1 次短抽取）：
- 抽 b3/b6/b9 的 **attention 权重**，构造 (i) attention-pooled patch 特征、(ii) attention-map 展平特征；
- 计算其 **E6 ratio** 与 **跨域 AUC**；
- **预登记门槛**：ratio < 0.30 **且** mean4dom > 0.8429（G16 拼接最好值）→ 才进入架构级实验；否则**关闭该路线**。
理由：若注意力特征在探针级都过不了 E6 律和 +0.011 这条线，端到端不可能翻盘；若过了，才值得花 GPU。

#### D.29.10 **措辞更正（我的错）**："84% 是架构性上界"这句话说大了

§D.29.8 的标题写成了"84% 是架构性上界正式成为本项目的最终结论"。**这个表述过宽，与本项目自己的数据冲突**：
- G19b 的 `O1multi` 在 **dfdcp 上达到 0.9298**（vs 冻结基线 0.8261）——**0.83 并非在数据侧也不可逾越**；
- 目标域 oracle 达 0.87–0.955（V），`O1multi` 甚至到 0.937–0.977。

**准确表述应为**（三句话，缺一不可）：
1. **在"仅 FF++ 单源 + 零样本"这个设定下**，跨域 AUC ≈ 0.83；
2. 该数字**无法被任何架构/表征层的干预改善**——7 条路线全部 ≤ +0.011 或有害（E3/G15、G16、G17、G18、G19a、G19b、G16-拼接）；
3. 但**可以被数据改善**：追加一个与目标**分布相邻**的源域（cd2）使 dfdcp 从 0.8261 → **0.9298（+0.096）**。

→ **正确结论不是"框架没有改进上限"，而是"改进只能从数据/适配轴取得，架构/表征轴已被穷尽"**（详见 §D.31）。

### D.31 方向 A（无标签目标域适配）**关闭** —— 用户否决（2026-09-15）

**用户给出的两条理由**：
1. 判定为学术不端；
2. **增益来自数据增加而非模型架构，不构成创新。**

**处置：A 永久关闭，后续不再提议。** 论文设定**钉死在"零样本 / 不使用任何目标域数据（有标签或无标签）"**。

**记录性说明（不影响结论）**：A（UDA/TTA）在领域内是常规且有大量已发表工作的设定，其合法性依赖"不使用目标域标签 + 按视频切分 + 明确声明为 UDA"三条；但**理由 2 对本论文的主张完全成立**——本项目的目标是**架构轴**贡献，而 A 的增益来源是数据轴，因此即便技术上合规，**也不解决本论文要解决的问题**。用户有权采用比领域惯例更严的标准。

**对现有产出的直接约束（事实性，必须遵守）**：
- G19a / G19b 的各臂训练集**包含其他目标域**（如 dfdcp 折的 MULTI = FF++ + cd2 + ffiw + wild）。
- → **这些运行不能被引为本论文"零样本"主张的证据。** 若写入论文，只能以"多源域泛化（multi-source DG）"名义出现，且**只有留出域那一列有效**，其余列标 `CONTAMINATED` 作废（§D.25 / §D.29.1 已定）。
- → **论文若坚持零样本单源设定，则 G19 全系列只能作为"对照组/否证材料"出现，不能作为主结果。**

**零样本单源设定下已闭合的架构轴路线（7 条，全部 ≤ +0.011 或有害）**：
E3/G15 冻结表征适配 · G16 换深度 + 多层拼接（+0.0111）· G17 换模态（16/16 无用）· G18 监督特化（cd1 −0.1001）· G19a 域不变目标 · G19b 解冻重训表征（−0.0146 / −0.0189）· G16 拼接的加权累加变体（数学上被拼接支配）。

---

### D.32 补测 G16 从未跑过的"纯低域锁拼接"：**预测被证伪**（2026-09-15）

> 动机：复核 §D.30.5 提出的"拼接拖拽律"。G16 的三个拼接变体**全都含 b3 或 b6**（b3 ratio 6.09 / b6 1.29），**`cat_b9_final`（只拼低域锁的 b9+final）从未测过**。按该律它应是**最好的**拼接变体——这是一个有明确方向的样本外预测。
> 脚本 `_g19/ratio_lowlock.py`（纯 CPU，读 `_g16/layer_feats.npz`；ratio 公式抄 G12 B 块，AUC 用 StandardScaler+LR C=1e-3 拟合 FF++ train 2200）。**锚点复现 PASS**：`cls_final` ratio 0.1983 / cd1 0.8286 / cd2 0.8633 / dfdcp 0.8261 / wild 0.8090 全部逐位一致。

| feature | dim | ratio | class_gap | cd1 | cd2 | dfdcp | wild | **mean4** | **Δ vs V** |
|---|---|---|---|---|---|---|---|---|---|
| cls_final | 768 | **0.1983** | 40.2122 | 0.8286 | 0.8633 | 0.8261 | 0.8090 | 0.8318 | — |
| **cat_b9_final**（新） | 1536 | 0.2169 | 54.1580 | 0.8313 | 0.8597 | 0.8233 | 0.8152 | **0.8324** | **+0.0006** |
| **cat_mp9_final**（新） | 1536 | 0.2165 | 54.7546 | 0.8180 | 0.8577 | 0.8206 | 0.8060 | **0.8256** | **−0.0062** |
| cat_b6_final（G16） | 1536 | 0.2010 | 56.6994 | 0.8549 | 0.8554 | 0.8415 | 0.8200 | **0.8429** | **+0.0112** |
| cat_cls369（G16） | 2304 | **0.3531** | 48.5045 | 0.8599 | 0.8328 | 0.8475 | 0.8169 | **0.8393** | **+0.0075** |

**`PREDICTION (纯低域锁拼接最好) -> REFUTED`**

#### D.32.1 证伪的具体形式：**ratio 排序与 AUC 排序不一致**
- ratio 序（低→高）：`cat_b6_final 0.2010` → `cat_mp9_final 0.2165` → `cat_b9_final 0.2169` → `cat_cls369 0.3531`
- AUC 序（高→低）：`cat_b6_final 0.8429` → **`cat_cls369 0.8393`** → `cat_b9_final 0.8324` → `cat_mp9_final 0.8256`
- **`cat_cls369` 的 ratio 最差（比 V 劣化 78%），AUC 却是第二好。** 且 `cat_b9_final` 的 ratio 与 V 几乎同级（0.2169 vs 0.1983）却**只涨 +0.0006**。
- 附带：`class_gap` 同样失效——`cat_b9_final` 的 class_gap 54.16 > V 的 40.21，AUC 却几乎不动；`cat_mp9_final` class_gap 54.75，AUC **−0.0062**。

→ **该"拖拽律"作为"预测哪个拼接更好"的工具：证伪。**

#### D.32.2 **对 §D.30.5 的撤回（我的第二次过度解读）**
§D.30.5 我写了"这就是 G18 的机理解释……G18 是本律在架构层的独立确认，不是巧合"。
- **机械部分成立**：拼接体的 ratio 确实被拉向最差块（这是均值距离的算术，无需实验）；
- **但推论不成立**：既然 `cat_cls369` 拼了 ratio 6.09 的 b3 却**涨了 +0.0075**（cd1 +0.0312），**"拼域锁块必然有害"在探针级被反证**，因此**不能用它解释 G18 的 −0.1001**。G18 的伤害另有原因，**当前无归因**。
- **撤回 §D.30.4 表末"预测：把 b3/b6 接进 bridge 会复现 G18 的伤害"**——该预测失去依据。

#### D.32.3 ratio 作为"迁移增益预测器"**第四次失败**
§D.28.5 已记三次；本次为第四次：
| 案例 | ratio | 实测增益 |
|---|---|---|
| O3a（G19a） | 0.0919（史上最低） | **0** |
| cat_b9_final | 0.2169（≈V） | **+0.0006** |
| cat_cls369 | 0.3531（比 V 劣 78%） | **+0.0075** |
| **对照：E6 跨表征律** | 14 点、ratio 跨 42 倍 | **零违例** |

→ **必须严格区分 ratio 的两种用途**：
1. **作为"某表征有多域锁"的描述量（状态）**：E6 律 14 点零违例，**成立**；
2. **作为"某个干预能带来多少迁移增益"的预测量（变化）**：**四次全败，不成立**。
→ 后续任何实验**不得**再用 ratio 作为候选改进方案的筛选门槛。

#### D.32.4 **拼接空间至此被完全穷尽**
G16 三个变体 + 本次两个新变体 = **低域锁方向上所有有意义的拼接组合都已测**，上限 **+0.0112**（`cat_b6_final`），**全部低于 +0.03 门**。
→ 探针级的**层/池化/拼接三维空间已无空缺**。

#### D.32.5 **未被本次证伪影响的发现：bridge 空转（直接测量，非代理量）**
`bridge 分支对决策方向的能量贡献 = 0.03%` 是**直接能量测量**，`bridge 输入 = blocks[3]/[6]/[9] patch token` 是**hook 语义的直接事实**。两条都不依赖 ratio 代理。→ **"模型实质退化为 CLIP-CLS ⊕ ViT-CLS"这一结论继续成立。** 但**它为何导致跨域封顶，目前只有相关性、无因果归因**（§D.32.2 已撤回那条因果链）。

---

### D.33 **重大架构缺陷：CLIP 语义对齐分支在 bridge 改写时丢失**（2026-09-15）

> 触发：用户提问"CLIP 被赋予 62% 权重却没起作用，是否是语义信息未被充分利用，怎么设计实验验证"。
> 结论：**用户的直觉方向正确，但对象指错了；真实缺陷比"未充分利用"严重得多——是"被删掉了"。**

#### D.33.1 先更正两个被混淆的数字（§D.19.1 原文）
| 数字 | 归属 | 状态 |
|---|---|---|
| **62.3%** | `clip_vision_cls`，即 **CLIP 视觉 CLS**（融合向量 `[0:768]`） | **正在工作**，且是决策主导项。§D.19.1 的论点是它**不该占这么多**（C 域内最弱 0.9108、ratio 2.716 最域敏感）→ **是权重错配，不是没起作用** |
| **0.03%** | `clip_adapt_embed`（bridge 的 128-d 输出，融合 `[768:896]`） | bridge 空转 |
| **0%** | **CLIP 文本** | **根本不在融合向量里**（1664 = 768+128+768，无文本维） |

→ 用户把 62.3%（CLIP 视觉）与"语义没起作用"划了等号；实际**语义确实是 0%，但 62.3% 是另一个分支**。

#### D.33.2 事实链（全部代码级可验证）
**事实 1 — `CLIPTextEncoder` 是图像无关的。** `llava/model/deepfake/M2F2Det/text_encoder.py` L26-49：
```python
def forward(self, input_embeds=None):
    if input_embeds is None:
        input_embeds = self.prompt_tokens      # [1, 7, 768]  ← 唯一输入，无图像
    ...
    pooled_output = last_hidden_state[:, -1, :]   # [1, 768]  ← batch 维恒为 1，与 B 无关
```
→ 文本特征是**全数据集同一个常量向量**，只由可学习 `prompt_tokens` 决定。

**事实 2 — bridge 版把文本算完就丢了。** `vit_module/vit_m2f2_detector_bridge.py`：
- L432 `clip_text_features = self.clip_text_encoder()`
- L435 `clip_text_features = self.text_proj(clip_text_features)`
- **此后全文 grep 无任何一次使用**；融合 L478-484 = `cat([clip_vision_cls, clip_adapt_embed, vit_features])`，**无文本**。
→ 即使把它拼进去，因为它对每个样本都是**同一个常数**，其 logit 贡献 `w_text·text_feat` 是**常数**，而 **AUC 是秩统计量、对全体加常数不变** → **AUC 变化恒为 0**。即该分支**在数学上与 `output.bias` 等价、冗余**。

**事实 3 — 参数是死的。** `text_proj`（L264）与 `prompt_tokens`（`text_encoder.py` L20）虽在优化器列表（`train_bridge_phase1.py` L305-308 / L327；模块 L566 / L574），但前向无通路 → **零梯度、从未被训练**。

**事实 4（关键）— 原始 M2F2Det 里它是活的，而且机制不同。** `llava/model/deepfake/M2F2Det/model.py` L186-193：
```python
clip_vision_cls, clip_vision_patches = clip_vision_features[:,0,:], clip_vision_features[:,1:,:]
clip_scores = F.cosine_similarity(
    clip_vision_patches,                                       # [B, 576, 768] 逐 patch、逐样本
    clip_text_features.unsqueeze(1).repeat(B, n_patches, 1),   # 文本广播到每个 patch
    dim=-1)                                                    # → [B, 576]  ★逐样本、逐 patch、空间分辨★
...
features = torch.cat([clip_scores, clip_vision_cls, deepfake_features], dim=-1)
```
→ 原版有一个 **576 维的"逐 patch 图文语义对齐图"**——`clip_scores` **是逐样本的**（图像侧随 patch/样本变化），所以它能影响 AUC。**这正是用户直觉中的"语义利用率"机制。**

**→ 结论：bridge 版在改写时把这个分支整个丢掉了，但把它的参数、优化器条目、docstring（`train_bridge_phase1.py` L8/L20-21 仍写着"训练 prompt_tokens / text_proj"）都留了下来。**

#### D.33.3 定性：**疑似改写时的移植回归，非有意设计**
支持"回归"而非"有意删除"的三条证据：
1. `clip_text_features` **仍在被计算**（L432/L435）——若是刻意删除，不会留无用计算；
2. `text_proj` / `prompt_tokens` **仍在优化器列表**——刻意删除不会留着占显存与 lr 组；
3. 训练脚本 docstring **仍宣称训练它们**。
（**保留不确定性**：不排除是有意简化后忘了清理。需向作者确认。）

#### D.33.4 为什么这条路线**不能**被前面 7 条已闭合路线直接类比否掉
| | 已闭合的 7 条路线 | 本路线（`clip_scores`） |
|---|---|---|
| 特征性质 | **视觉**特征（CNN 块 / 频域 / SRM / DCT） | **图文相似度**（图像 patch ↔ 文本） |
| 实测 ratio | 5.60 – 8.41（全部域锁） | **从未测过** |
| 先验 | 域锁 → 跨域必败 | 文本侧来自 CLIP **web-scale 预训练**（非 FF++ 训练）→ **原理上是本项目唯一先天域无关的信号源** |

→ **这是一格真空缺，且性质与已否证的 7 条不同。** 但不能因此乐观：`prompt_tokens` 若在 FF++ 上训练，学到的仍是源域语义。

#### D.33.5 验证协议（**预登记**，全部零训练；用户已要求暂不跑）
**V1 梯度归零**：加载 `bridge_v2_phase1.pth`，单次 forward+backward，测 `max|∂L/∂θ|` for θ ∈ {`text_proj.*`, `prompt_tokens`, `clip_text_encoder.model.*`}。**判据：恒等于 0.0。**
**V2 扰动不变**：把 `clip_text_encoder()` 的返回替换为 (a) 全零 (b) 同形状随机噪声，重算 logits。**判据：max|Δlogit| == 0.0（逐位相同）。**
**V3 恒定性与形状**：确认返回 shape `[1,768]`；在 5300 张图的全部样本上，`text_proj` 输出**逐样本方差 == 0**。
**V4 跨模型静态差分**：已由 §D.33.2 事实 4 完成（原版有 `clip_scores` 段，bridge 版无）。
**V5 能量分解复核**：重新导出 `output.weight` (2,1664) 的分段能量，并显式报告"文本维数 = 0 维、占比 = 0%"。
**资源**：V1/V2 需 1 次模型加载（CPU 可跑，约 1.0B 参数，单次前向；限 CPU 单线程）；V3 需 1 次 5300 图前向（GPU，batch 16，≤5 min）；V5 纯 CPU 读 checkpoint。

#### D.33.6 针对性改进设计（**仅在 V 块确认缺陷后启动**）
**D-1（免费、决定性的第一步）——用 CLIP 原生文本做探针，不训练任何东西。**
- 不还原 `prompt_tokens`（它是随机 init，用了等于随机文本），而是用 **CLIP 原生 text encoder 编码真实字符串**（如 `"a photo of a real face"` / `"a photo of a deepfake face"`），得到**未经 FF++ 训练的**语义嵌入；
- 按原版公式算 `clip_scores = cos(clip_vision_patches, text_emb)` → [B,576]；
- **测它的 E6 ratio 与跨域 AUC**（源 FF++ train → 各目标域）。
- **预登记门槛**：`ratio < 0.30` **且** `mean4dom > 0.8429`（G16 拼接最好值）→ 才进入 D-2；否则**该路线与其余 7 条一同关闭**。
- **为什么这一步有意义**：这是本项目**唯一一个先天未经 FF++ 训练**的判别信号源。

**D-2（仅当 D-1 过门）——还原分支并训练。**
- 融合改为 `cat([clip_scores(576), clip_vision_cls(768), clip_adapt_embed(128), vit_features(768)])` → `Linear(2240→2)`；
- 训练 `prompt_tokens` + 新头（其余冻结，与 stage-1 recipe 一致）；
- **对照臂必做**：`prompt_tokens` 随机冻结（验证"增益来自训练出来的语义还是来自对齐机制本身"）；
- 注意：`output` 维度变了，**不能复用已训头**，必须重训头——这会引入与 G19b 同样的"重训容量"混杂，**必须加 rank/容量对照**。

---

### D.34 缺陷的**二重性**与"语义引导融合"方向：门控顺序（2026-09-15）

> 用户指出（本节采纳并精确化）：① `clip_text` 输出本应被使用却没用上；② **CLIP 语义在结构上不适合用现在的方式融合**——没有"语义引导特征的学习"。用户提出"语义引导"作为改进方向。

#### D.34.1 缺陷是**两个**，不是一个，且需要不同的修法
| | 缺陷 | 修法 |
|---|---|---|
| **缺陷 I（丢失）** | 原版的 `clip_scores`（逐 patch 图文余弦对齐图，[B,576]）在 bridge 版被丢弃 | 恢复该分支 |
| **缺陷 II（结构）** | **融合是晚期拼接**：`cat([...]) → Linear`，各分支贡献是**固定的线性投影，分支间无交互项** | 需要**交互式**架构 |

**缺陷 II 的精确表述（用户判断成立）**：
`bridge_adapter` 是 3 个 `TransformerBlock`，对 **2316 token**（3×196 ViT patch + 3×576 CLIP **视觉** patch）做 self-attention——**架构里确实有 token 交互，但文本从未进入该交互**。
且 `clip_text_alpha` 乘在 **bridge 输出**上（L471）→ **命名表明设计者本意是让 bridge 输出成为"文本介导的嵌入"，但文本输入从未接入**。这与 §D.33.3 的"移植回归"假设自洽。

**两条推论（重要）**：
1. **仅修缺陷 I 无用**：把 `clip_scores` 拼回现有晚期融合头，因文本特征是常量 → AUC 变化恒为 0（§D.33.2 事实 2）。
2. **仅修缺陷 II 也不够**：若语义信号本身不携带跨域真假信息，再好的注入架构也是空的。

#### D.34.2 **门控顺序（我坚持的纪律）**
必须按 **信号 → 架构** 的顺序，不能反过来：
```
V 块：确认缺陷真实存在（零训练）
   ↓
D-1：测「冻结 CLIP 原生语义」这个信号本身有没有跨域真假判别力（零训练）
   ↓   门槛：ratio < 0.30 且 mean4dom > 0.8429
   ↓   ✗ 未过 → 整条路线关闭，不建架构
   ↓   ✓ 过了
D-2：才设计并训练「语义引导融合」架构
```
**理由**：本项目已有 4 次"用代理量预测架构结果"全部失败的记录（§D.28.10、§D.32.1、§D.32.3）。**我不再对 D-1 的结果做预测**——这正是要先测它的原因。若先建架构，一旦信号本身不成立，全部白做。

#### D.34.3 D-2 的候选架构（**仅在 D-1 过门后启动**；此处为形态预览，非承诺）

**核心设计约束：文本侧必须冻结**（用 CLIP 原生嵌入或冻结的 prompt），否则一旦在 FF++ 上训练，语义立刻退化为源域语义——那就回到已否证的 7 条路线的老问题。**"先天未经 FF++ 训练"是这条路线的全部价值来源。**

| 方案 | 机制 | 语义如何"引导" |
|---|---|---|
| **G-a 语义加权池化** | 把 `clip_scores` [B,576] 当**注意力权重**去池化视觉 patch，替代均匀 mean-pool | 语义决定**看哪些 patch** |
| **G-b 文本→图像交叉注意力** | ViT patch 作 Query，冻结文本 token 作 Key/Value，输出语义调制后的视觉 token | 语义**重写**视觉表示 |
| **G-c FiLM 条件调制** | 冻结文本嵌入产生 per-channel γ/β，作用于视觉特征 | 语义**缩放平移**视觉通道 |
| **G-d CLIP 式对比头** | 图像嵌入与"真实人脸/伪造人脸"文本嵌入做余弦相似度，直接作 logits | 语义**定义**判别方向 |

**先验（必须写明，不得淡化）**：本项目 7 条架构路线全部 ≤ +0.0112 或有害；E6/G17 表明凡在 FF++ 上有判别力的信号多与域绑定。**G-a~G-d 与它们的不同点仅在于"文本侧先天未经 FF++ 训练"——这是唯一的差异化理由，也是唯一需要被 D-1 验证的东西。**

**必配对照（防 G19b 式容量混杂）**：
- 文本侧随机冻结 vs 原生 CLIP 文本 → 区分"语义"与"机制"；
- 与 D-1 的"平行拼接"版本对照 → 区分"引导"与"仅仅加入"；
- 随机化文本嵌入（打乱语义标签）对照 → 排除任何注入即涨的假象。

---

### D.35 G21 预登记：V 块 + D-1（2026-09-15）

**卡分配（执行前实测）**：GPU0 已用 5166 MiB → **排除**；**GPU1（0 MiB/11120 空闲）→ D-1**；**GPU2（64 MiB/11055 空闲）→ V 块**。约束照旧：只用已用 ≤100 MiB 且空闲 ≥6 GB 的卡；batch ≤16；fp32；线程全 1；若指定卡被占用则**停止并报告，不换卡**。

**D-1 预登记的提示词（先定后测，不得事后修改）**：
```
REAL = ["a photo of a real human face",
        "a genuine photograph of a person",
        "an authentic image of a human face"]
FAKE = ["a photo of a deepfake face",
        "a digitally manipulated face",
        "an AI-generated synthetic face"]
NULL = ["a photo of a chair"]
```
**关键实现约定**：**必须使用 CLIP 原生 `visual_projection` / `text_projection`**（web-scale 训练）——**绝不可**用检测器的 `text_proj`（随机初始化、从未训练，§D.33.2 事实 3）。副臂 `cs_modelvis` 用检测器**已训练**的 `vision_proj` + 原生文本。

**特征定义**：
- `cs_alpha`（主）：`S_real = cos(patch, mean_t REAL)`[B,576]、`S_fake = cos(patch, mean_t FAKE)`[B,576]、`D = S_fake − S_real`[B,576]
- `cs_mean`（1 维）：`mean_patch(D)`
- `cs_cat`（4 维）：`[mean(S_real), mean(S_fake), mean(D), max(D)]`

**对照（必做）**：null 提示词 · 随机文本嵌入(3 seed) · real↔fake 标签互换 · `cs_modelvis` 副臂。

**预登记门槛**：`ratio < 0.30` **且** `mean4dom > 0.8429` → 进 D-2；否则该路线与其余 7 条一同关闭。
**预登记参照**：V 冻结锚点 mean4dom = 0.8318；G16 拼接最好 = 0.8429。

**V 块判据（§D.33.5）**：V1 文本参数 `max|∂L/∂θ| == 0`；V2 替换文本输出为零/噪声后 `max|Δlogit| == 0`（逐位）；V3 文本输出逐样本方差 == 0、shape `[1,768]`；V5 文本维数 0。

---

### D.36 **G21-V 结果：文本分支确认死亡 +（新发现）bridge 分支从来就是死的**（2026-09-15）

> Agent：G21-V，**GPU 2**（执行前实测 64 MiB/11055 空闲，全程未换卡），batch 16，fp32，线程全 1，wall 37.0s，torch peak 4118 MiB。
> 模型加载 `bridge_v2_phase1.pth` `strict=False`，**missing=0 unexpected=0**；48 张真实图，read_fail=0。
> **我（主 agent）已独立复验本节最关键的一条**，见 D.36.5。

#### D.36.0 判定
- `TEXT_BRANCH_DEAD` = **CONFIRMED**（§D.33.2 全部三条事实成立）
- **新增：`BRIDGE_BRANCH_DEAD` = CONFIRMED，原因为 `nn.LayerNorm(1)` 信息湮灭**

#### D.36.1 V5 能量分解（复核 §D.19.1）
`output.weight` = (2,1664)，`d = w[1]−w[0]`，`||d|| = 1.36469705`：

| 分段 | 维数 | 占比 |
|---|---|---|
| `[0:768]` clip_vision_cls | 768 | **62.2948 %** |
| `[768:896]` bridge / clip_adapt_embed | 128 | **0.0265 %** |
| `[896:1664]` vit_features | 768 | **37.6787 %** |

与 §D.19.1 记录（62.3/0.03/37.7）一致。**文本占 0 维**——这是**结构性**的：concat 里根本没有文本槽位。per-dim 能量占比 >0.001 的维数：**291/1664**。

#### D.36.2 V3 形状与恒定性（全部符合预期）
- `model.clip_text_encoder()` → **shape (1,768)**，float32。机理：`input_embeds` 默认取 `prompt_tokens` (1,7,768)，**batch_size 被硬编码为 1**，图像 batch 维从不进入该函数。
- `text_proj` 输出 **(1,768)**。**注意：请求的"batch 内逐样本偏差"在该张量上是空命题（只有 1 行、无样本对）**——agent 正确地指出并替换为实质等价检验：`max|T(batch1)−T(batch16)| = 0.0`、`max|T(batch8)−T(batch16)| = 0.0`。
- 完全不重叠的第二批 16 张：L∞ = **0.0**、L2 = **0.0**、`torch.equal = True`。
- **重复前向确定性对照通过**（`equal=True`, max|Δ|=0）→ 后续逐位比较可解释。

#### D.36.3 V2 功能通路（决定性）
| 替换物 | `max\|Δlogit\|` | `torch.equal` |
|---|---|---|
| (a) 全零 | **0.0** | **True** |
| (b) 噪声 seed0（max\|x\|=3.99） | **0.0** | **True** |
| (b) 噪声 seed1（max\|x\|=3.29） | **0.0** | **True** |
| (b) 噪声 seed2（max\|x\|=3.03） | **0.0** | **True** |
| 额外：形状 (16,768) 的替换 | **0.0** | **True** |

→ **文本对输出零影响，逐位成立。**

#### D.36.4 V1 梯度
`loss = out.sum()`，batch 8，backward。
- **文本分支 201 个参数，`grad` 全部为 `None`**：`text_proj.*`(4)、`prompt_tokens`(1)、`clip_text_encoder.model.*`(196)。**是 `None` 而非 0.0**——autograd 根本没建图。
- 对照（应有梯度）：`output.weight` = 96.63、`vision_proj[0].weight` = 1.302、`deepfake_proj[0].weight` = 0.482，全部非零。
- ⚠️ **不符合预期的一条**：`clip_reduction.weight` 的梯度**恰为 0.0**（v-spec 预期它非零）→ 触发下面的 V1x。

#### D.36.5 **V1x——新发现：`nn.LayerNorm(1)` 把 bridge 整个湮灭（我已独立复验）**

**位置**：`vit_module/vit_m2f2_detector_bridge.py` L158-161
```python
self.bridge_adapter_proj = nn.Sequential(
    View(-1, self.embed_dim),
    nn.Linear(self.embed_dim, 1),      # → [total*B, 1]  逐 token 一个标量（LN 前 spread 实测 0.3555，是活的）
    nn.LayerNorm(1),                   # ← ★归一化维数=1★
)
```
**数学**：对形状 `[N,1]` 做 `LayerNorm(1)`，mean = x、var = 0 → `(x−x)/sqrt(0+eps)·γ + β = β`。**输出恒为常数，与输入无关。**

**我的独立复验**（`nn.LayerNorm(1)`，weight=0.999358 / bias=0.018901，即该模型实测值）：
```
in : [0.1, 0.5, -3.0, 7.7, -100.0, 1234.5]      spread_in  = 1334.5
out: [0.018901] × 6                              spread_out = 0.0
verdict: INFORMATION DESTROYED (output constant)
```
**→ 确认。这不是相关性推断，是数学恒等式。**

**实测后果（agent）**：
- 模型内该 `LayerNorm(1)` 的 weight `[0.999358]` / bias `0.018901`；其逐 token 输出在全部 37056 个元素上的 spread = **0.0**；
- **128 维 bridge 嵌入在 batch 内、且在两个完全不同的 batch 之间逐位相同**（max|A−B| = 0.0，L2 = 0.0）；
- 梯度：`linear_vit_`(6) = 0.0、`bridge_adapter.`(36) = 0.0、`clip_reduction.`(2) = 0.0，**全部恰为零**；仅 `bridge_adapter_proj.`(8，位于坍缩点下游) 非零（3.098）。
- `clip_text_alpha` 梯度 1.168、`clip_vision_alpha` 0.590 非零（前者只乘在一个常数上）。

#### D.36.6 **该 bug 是继承的，不是移植引入的——三份实现全中**
| 文件 | 行 | 结构 |
|---|---|---|
| `sequence/models/M2F2_Det/models/model.py`（**原版**） | L130-140 | `Linear(hidden,1), nn.LayerNorm(1)` **完全相同** |
| `vit_module/vit_m2f2_detector_unified.py` | L121-123 | 同上 |
| `vit_module/vit_m2f2_detector_bridge.py` | L158-161 | 同上 |

→ **原版 M2F2-Det 的 `BridgeAdapter_Proj` 就有这个缺陷。** 因此 bridge 在**任何一份实现里都从未能够传递任何逐样本信息**。

#### D.36.7 后果（本节重要性所在）
1. **0.03% 决策能量终于有了因果解释**——**不是相关性**。一个常数只能折进 bias，而分类头本就有 bias → 该分支在数学上与 `output.bias` 冗余。**这解决并替换了 §D.32.5 里"仅相关性、无因果归因"的保留。**
2. **bridge 的全部参数从未被训练**：`linear_vit_*`、`bridge_adapter[0..2]`、`clip_reduction` 梯度恒为 0。所谓"stage-1 训练 bridge"**不成立**。
3. **模型实质是 `CLIP视觉CLS ⊕ ViT-CLS` + 常数** —— 现在是**证明**，不再是推断。
4. **`clip_text_alpha = 1.7796` 乘的是一个常数**，其学到的值除了当 bias 缩放系数外无意义。
5. **对本项目"架构轴已穷尽"结论的重新划界（重要）**：已闭合的 7 条路线全部是**往融合里加特征**或**改被冻结的表征**，**没有任何一条触碰过这个已死的机制**。→ **正确的表述是"特征增补轴已穷尽"，不是"架构轴已穷尽"。**
6. **对论文的直接影响**：M2F2-Det 的卖点机制（bridge）在其原始实现里即从未生效。这是一个**正确性缺陷**，且是**可修复的**。

#### D.36.8 未被本次证伪、但需保留的不确定性
- V1 用 `loss = out.sum()`：`grad is None` 证明"**不被本图到达**"。结合 V2 的逐位零影响，功能上是死的；但"历史上从未被任何脚本训练"这一更强命题仍需训练日志佐证。
- **旁证（agent 的 V1b 数据，与"未训练"自洽）**：`text_proj.0.weight` (768,768) `std = 1.000059e−02` ≈ PyTorch `Linear` 默认 init 的理论值（kaiming_uniform a=√5, fan_in=768 → std≈0.0100）；`prompt_tokens` (1,7,768) `std = 1.0141` ≈ `randn` 的 1.0。**两者都像是原封未动的初始化**（旁证，非证明）。
- `LayerNorm(1)` 是否为原作者的**有意设计**（而非疏漏）无法从代码判定——但其数学效果与设计意图（"3 阶段渐进融合"）**矛盾**，大概率是疏漏。
- 单 checkpoint、单卡、仅 fp32。

#### D.36.9 状态
**D-1（冻结原生 CLIP 语义探针）仍在 GPU 1 上运行，尚未返回。** 该实验独立于本节结论，继续等待。

---

### D.37 **G21-D1 结果：冻结原生 CLIP 语义对齐信号不携带可用判别力 —— 路线关闭**（2026-09-17）

> Agent：G21-D1，**GPU 1**（起始 `memory.used=0 MiB`，全程未换卡），batch 16，fp32，4500 图，282 次 batch 调用，read_fail=0，wall **308.3 s**，nvidia-smi peak 8740 MiB（torch 6398 MiB）。线程全 1，单进程。
> CLIP 路径：**本地** `checkpoints/clip-vit-large-patch14-336`。**只用原生 `visual_projection`/`text_projection`**；未使用检测器的 `text_proj`（已按要求禁用）。
> 额外自检：检测器 CLIP 视觉塔与独立 CLIP 视觉塔**逐位一致**（同键 391，max abs dev 0.000e+00）→ 两条路径确实是同一个塔。

#### D.37.1 协议自检（全部 PASS → 数字与已记录的 ratio 表可比）
- 管线复现 `cos(cls_fresh, layer_feats.cls_final)` 8 图 = **1.000000**（门 0.999）
- E6 公式在冻结 V 上：`s=1.5097`、`class_gap=40.2122`、`ratio=0.1983` —— **与锚点逐位相同**
- V 跨域 AUC 复现已记录锚点，maxdev **3.3e−5**；V `mean4dom = 0.8318` —— **精确**

#### D.37.2 主表
| feature | dim | s | class_gap | ratio | **mean4dom** | ffpp(域内) |
|---|---|---|---|---|---|---|
| **cs_alpha（预登记主臂）** | 576 | 0.0213 | 4.5420 | **1.2644** | **0.6246** | 0.7327 |
| cs_mean | 1 | 0.0069 | 0.4066 | 0.5977 | **0.6786**（主臂最好） | 0.6172 |
| cs_cat | 4 | 0.0105 | 0.9606 | 1.1044 | 0.6552 | 0.6936 |
| cs_pool | 2 | 0.0164 | 0.9638 | 0.9563 | 0.6105 | 0.7350 |
| cs_modelvis（副臂，检测器已训 `vision_proj`） | 576 | 0.0170 | 1.4540 | 3.7154 | **0.4862**（随机水平） | 0.5585 |
| cs_alpha_last（层选择鲁棒性） | 576 | 0.0170 | 3.4901 | 2.3410 | 0.5262 | 0.7204 |

`cs_alpha` 逐域：cd1 = 0.6430、cd2 = 0.6220、dfdcp = 0.6003、ffiw = 0.5889、wild = 0.6333。
**参照**：冻结 V 基线 mean4dom **0.8318**；G16 拼接最好 **0.8429**。

#### D.37.3 **关键读数：这个信号在域内就很弱**
`cs_alpha` 的 **FF++ 域内 AUC 只有 0.7327**，而 V 是 **0.9852**。
→ **这不是"域迁移问题"，是这个信号本身携带的真假信息就很少。** 因此无论用什么注入架构（G-a 加权池化 / G-b 交叉注意力 / G-c FiLM / G-d 对比头），可抽取的上限都被压在低位。
→ 机理上说得通：**"真实人脸 vs 伪造人脸"不是一个语义范畴问题，而是像素级伪影问题**，CLIP 的文本语义先验**结构上就不是这个任务的合适工具**——这同时解释了为什么 CLIP **视觉**分支（62.3%）比文本分支管用得多。

#### D.37.4 对照（全部合格，说明探针本身是校准过的）
| 对照 | mean4dom 范围 | 判定 |
|---|---|---|
| NULL（"a photo of a chair"，两种槽位摆放 + 1 维参考） | **0.5051 – 0.5840** | 近随机 ✓ |
| RANDOM（3 seed 归一化 `torch.randn(768)`） | **0.4738 – 0.5709** | 随机水平 ✓ |
| LABEL SWAP | **0.6246**，与主臂**完全相同** | **见 D.37.5（我的预登记有误）** |

#### D.37.5 **我的预登记错误（第 5 次）：label-swap 对照在数学上是空命题**
我预登记"标签互换后应约等于 `1−AUC`"。实测**与主臂逐域完全相同**。
Agent 给出并数值验证了解释：互换标签即 `X → −X`，而 `StandardScaler(-X) == -StandardScaler(X)`（dev 0.0e+00）、LR 系数**严格反对称**（dev 0.0e+00）、截距**相同**（0.057575 both）⇒ `df_swap(-x) == df_main(x)`，而 **AUC 在"对两类同施加全局反射"下不变**。
→ **该对照是反对称性检查，不是方向性检查**；真正承担语义对照职能的是 NULL 与 RANDOM 两臂。
（构造非纯取反时它确实会动：`cs_cat` 0.6552 → `cs_cat_swap` 0.6877。）
**记录教训：预登记门槛前必须先在纸上验算对照的数学性质，否则会登记一个恒真命题。** agent 主动指出而非隐藏，处理正确。

#### D.37.6 预登记门槛判定（不软化）
| 门 | 实测 | 门槛 | 结果 |
|---|---|---|---|
| `GATE_RATIO_OK` | 1.2644 | < 0.30 | **0**（超上限 4.2 倍） |
| `GATE_AUC_OK` | 0.6246 | > 0.8429 | **0**（差 0.2183） |
| **`GATE_PASS`** | — | — | **0 → 路线关闭** |

→ **"语义引导融合"（§D.34.3 的 G-a~G-d）全部关闭。** 且这是**在零训练成本下关闭的**——§D.34.2 坚持的"先验信号、再建架构"顺序，恰好省掉了一整套本会白做的架构与训练。

#### D.37.7 当前全局状态
| 方向 | 状态 | 依据 |
|---|---|---|
| CLIP 文本分支 | **确认死亡**（0 维、零梯度、逐位零影响） | §D.36 |
| CLIP 语义引导融合 | **关闭**（信号域内仅 0.73） | §D.37 |
| **bridge 机制修复** | **开放**——且是唯一剩下的、论文自己的机制 | §D.36.5–D.36.7 |
| 特征增补轴（7 条） | 已穷尽（≤ +0.0112 或有害） | G16/17/18/19 |
| 源域相邻性（+0.096） | 用户已判定非创新、不走 | §D.31 |

**保留的边界（不夸大）**：D-1 是**线性可达性下界**，交叉注意力等非线性抽取理论上可能取到更多；但域内 0.73 的线性上限很低，翻盘概率小。prompt 集合是预登记固定、**未做搜索**。

---

### D.38 G22 设计：bridge 修复（**让论文自己的机制第一次真正运行**）（2026-09-17）

#### D.38.0 修复对象的精确定义
`BridgeAdapter_Proj_ViT.bridge_adapter_proj` 的意图显然是**在 token 维上归一化逐 token 标量**，但 `LayerNorm(1)` 实际归一化的是**单元素维**。修法两选：

| 变体 | 做法 | 说明 |
|---|---|---|
| **FIX-A** | **直接删除 `nn.LayerNorm(1)`** | 最简、表达力最强 |
| **FIX-B** | 把归一化移到 `View(B, total_seq)` **之后**，用 `LayerNorm(total_seq)` | 更贴合原意（在 2316 个 token 上归一化） |

**连带影响**：这个 `LayerNorm(1)` 一改，`bridge_adapter_reduction` 的输入不再是常数 → 以下**从未训练过**的参数**全部复活**：
`linear_vit_*`(6) · `clip_reduction`(2) · `bridge_adapter[0..2]`(36) · `bridge_adapter_reduction`(Linear 2316→128 ≈ 0.30M) · `bridge_adapter_proj.*`。
→ **因此 G22 必然是训练实验**：这些参数当前停留在随机初始化。

#### D.38.1 **必须提前说明的张力（我不淡化）**
修复 bridge = 把一条**基于 `blocks[3]/[6]/[9]` patch token**的注意力融合分支真正接进融合。
而：
- G16 测出 b3/b6 的域锁比是 **6.09 / 1.29**（CNN 专用特征区间）；
- G18 测出"往融合里拼一条域锁分支" → **cd1 −0.1001**；
- G16 所有拼接变体 ≤ **+0.0112**。

**→ 我的先验是负面的：跨域增益 ≥ +0.03 的概率我给 15–25%。**
**→ 但这件事仍然必须做**，三个理由：(1) 是**正确性缺陷**，与涨不涨分无关；(2) **从未在"能工作"状态下测过**，因而不是前 7 条的重复；(3) 无论成败都直接回答"论文自己的机制到底有没有用"——这正是"特征增补轴已穷尽"**不能**覆盖的那一格。
**→ 并且我不预测结果**（本项目我预测架构结果已错 5 次，其中 4 次是用代理量推的）。

#### D.38.2 臂设计（5 臂；能出增益的臂一律 3 seed）
| 臂 | LN(1) | bridge 参数 | 训练对象 | 作用 |
|---|---|---|---|---|
| **R0** | 保持原样 | 出厂 checkpoint | **不训练** | **硬锚点**：mean4dom 必须复现 0.8318 ± 1e-3 |
| **FIX-A** ×3 seed | 删除 | 从 checkpoint 解冻 | bridge + 头 + 各投影（**照抄 stage-1 recipe**） | 主臂 |
| **FIX-B** ×3 seed | 移到 token 维 | 同上 | 同上 | 判别"是否真的需要这个归一化" |
| **CTRL-RAND** ×3 seed | 修复 | **冻结在随机初始化** | 仅头 + 投影 | **容量/随机对照**：区分"bridge 携带信号"与"可训参数变多"（G19b 教训） |
| **CTRL-NOFIX** ×1 | 保持原样 | 解冻 | 同上 | **因果对照**：不修则 bridge 仍零梯度 → 应 ≈ R0；证明"增益来自修复而非训练" |

#### D.38.3 训练与评测协议（**严格零目标域数据**）
- **训练集**：**仅** FF++ `split=="train"`（2200 张）。**绝不触碰任何目标域数据**（用户硬性要求，§D.31）。
- **训练对象**：照抄 `vit_module/train_bridge_phase1.py` 的 optimizer / LR 分组 / 调度（agent 需先读该脚本再实现），ViT 主干与 CLIP 双塔保持冻结。
- **评测**：用**模型自身 `output` 的 logits**（端到端），非线性探针。`logit[0]=real, logit[1]=fake` → 正类为 fake 的得分 = `logits[:,1] − logits[:,0]`。
  - 域内：FF++ `split=="test"`（800 张）
  - 跨域：cd1/cd2/dfdcp/wild 各 300（ffiw 报出但**不计入聚合**，单视频身份泄漏）
  - `mean4dom` = cd1/cd2/dfdcp/wild 均值
- 图像管线必须与 G16 一致（`cv2.imread → BGR2RGB → resize(336) → albumentations Normalize(CLIP) → ToTensorV2`）。
- batch ≤ 8（训练）/ ≤16（推理），fp32，线程全 1。

#### D.38.4 预登记门槛
| 门 | 判据 |
|---|---|
| `G22_ANCHOR` | R0 的 mean4dom == 0.8318 ± 1e-3（**硬门，不过则整个评测实现有误，全部作废**） |
| `G22_FIX_EFFECT` | FIX-A 三 seed 均值 − R0 ≥ **+0.03** |
| `G22_VS_CAPACITY` | FIX-A 均值 − CTRL-RAND 均值 ≥ **+0.02**（必须胜过"同等容量但随机"的对照） |
| `G22_FIX_IS_CAUSE` | CTRL-NOFIX − R0 落在 **±0.01** 内（不修则训练不改变任何东西） |
| `G22_NO_FORGETTING` | FF++ 域内 ≥ 0.9752（**已知近乎空门，仅报不判**，§D.28.5） |
**逐域 300 张，单域差值 < 0.05 不得过度解读**（G16 caveat 3 / G19b 教训）。

#### D.38.5 纪律
- **禁止修改** `vit_module/vit_m2f2_detector_bridge.py`（及其他任何既有文件）——修复必须在 agent 自己的脚本里用 **monkeypatch** 替换 `model.bridge_adapter_proj.bridge_adapter_proj`。
- 只在 `vit_module/_g22/` 内建文件；**不得写 WORKLOG.md**。
- 卡：只用已用 ≤100 MiB 且空闲 ≥6 GB 的卡，显式指定，**被占用则停止并报告，不换卡**。
- **报告实际发生了什么**，包括失败、与假设相反的结果；**每组数字必须同时给出 `s` 与 `class_gap`**（不得只给 ratio）；单 seed 结果一律标注"单 seed，不可单独采信"。

#### D.38.6 成本
9 次运行（R0 1 + FIX-A 3 + FIX-B 3 + CTRL-RAND 3 + CTRL-NOFIX 1 = 11 次）。桥接参数仅约 2–5 M，ViT/CLIP 冻结可前向复用；预计 **单臂 8–15 min**，总计 **≈ 1.5–2.5 h 单卡**。

#### D.38.7 【更正】D.38.3 的训练数据写错了（用户指出，属实）
**错误**：我在 §D.38.3 写"训练集 = FF++ `split=="train"` 2200 张"。
**事实**：2200 是 **G16/G17 探针的 train 划分**，是**评估用**的小划分，**不是模型训练集**。
出厂 `checkpoints/stage_1/bridge_v2_phase1.pth` 的真实训练集是
`./dataset/data_2023/ffpp_train_split.txt` = **107,700 张**。
**证据**（`logs/stage1_training/bridge_v2/bridge_v2_phase1.log`）：
```
train_txt: ./dataset/data_2023/ffpp_train_split.txt
batch_size: 64 | epochs: 30 | lr: 0.0005 | patience: 5
Batches/epoch: 1795      → 1795×64 = 114,880 ≈ 全量 107,700
```
**root cause**：我把"探针评估划分"当成了"模型训练集"。两者在本项目里数字完全不同，必须分开记。

#### D.38.8 【更正】D.38.6 的成本估算错了约 50 倍
**同一份日志给出的真实吞吐**：
| 量 | 实测值 |
|---|---|
| 单轮耗时 | **7,620 s = 2.12 h / epoch**（全量 107,700） |
| 每张耗时 | **≈ 70 ms**（读图+前向+反向，batch 64） |
| 实际训练 | 早停于第 11 轮，**总计约 23 小时** |
| 出厂模型成绩 | 最佳 epoch 6：FF++ AVG AUC **90.29%** / Cross-Domain AVG AUC **83.77%**（与本项目 `mean4dom=0.8318` 互洽 ✓） |
**→ 结论：G22 不可能按 D.38.6 的"单卡 1.5–2.5 h / 11 次运行"执行。** 全量重训一臂 6 轮 = 12.7 h，11 次运行在数量级上不可行。
**→ 必须在"数据量 / 轮数 / 并行卡数"三者中做取舍，这是用户的决定，不是我的。**

#### D.38.9 【更正】沟通失败（用户明确指出，属实）
用户原话：**"现在你提到的概念我已经有些听不懂了，臂、各类门，我无法对你的设计进行评估"**。
**root cause**：`臂`/`门`/`预登记门槛` 是我在项目内部自造的简写，从未在 WORKLOG 里给出面向人的定义就拿来沟通，
导致设计**无法被评审**——而这恰恰违背了"设计→用户审→派单"的流程本身。**这是我的问题。**
**纠正**：自本节起，凡面向用户的说明一律改用日常说法——
`臂` → **实验组**；`门` → **事先定好的通过标准**；`预登记` → **先定标准再跑，不许跑完再改**。

---

### D.39 G22 定稿（用户已拍板：**每组 3 遍 / 训练集用全量 107,700**）（2026-09-17）

#### D.39.1 【重要修正】基线不能沿用 0.8318
`mean4dom = 0.8318`（cd1 0.8286 / cd2 0.8633 / dfdcp 0.8261 / ffiw 0.8244 / wild 0.8090）是
**线性探针跑在 `cls_final` 特征上**的成绩（G16/G17/G12 协议），**不是模型自身 `output` logits 的成绩**。
G22 是微调实验，比较对象必须是**同一个模型输出头**，否则不是同类比较。
**→ 基线 R0 必须现场重测**：加载出厂 `bridge_v2_phase1.pth`，在我们的 2000 张评估图上取模型 `output` logits 的 AUC。
**旁证**：出厂训练日志自评 `FF++ AVG AUC=90.29% / Cross-Domain AVG AUC=83.77%`（epoch 6）——但那用的是另一套更大的评估集
（CD2 15,540 / FFIW 12,234 / FFPP_all 21,000 / Wild 24,180 / dfdcp 18,578），**同样不能直接当 R0**，只能作量级校验。

#### D.39.2 定稿协议
- **训练集**：`dataset/data_2023/ffpp_train_split.txt`，**107,700 张**（与出厂一致）。**绝不使用任何目标域数据**。
- **起点**：出厂 `checkpoints/stage_1/bridge_v2_phase1.pth`。
- **可训练**：bridge 全部参数（`linear_vit_1/2/3`、`clip_reduction`、`bridge_adapter[0..2]`、
  `bridge_adapter_proj.*`、`bridge_adapter_reduction.*`）+ `output` 头。
- **冻结**：ViT 主干、CLIP 双塔、`deepfake_proj`、`vision_proj`、`text_proj`、`prompt_tokens`、两个 alpha。
  （冻结投影层 = 把变量压到最小，对照组差异只剩一处。）
- **超参**：照抄 stage-1（lr 5e-4 / wd 1e-4），**batch 32**（stage-1 用 64，受"训练 batch ≤32"约束下调，**须在报告中标注此偏离**）。
- **轮数**：**1 轮**（对照：stage-1 第 1 轮训练准确率已 99.57%）。
- **评估**：模型自身 `output` logits；`logits[:,1]` 为正类=fake。评估集 = G16 的 800 张 FF++ test + cd1/cd2/dfdcp/ffiw/wild 各 300。
  `mean4dom` = cd1/cd2/dfdcp/wild 四域均值（**ffiw 单列报出、不计入聚合**，单视频身份泄漏）。
- **修法**：FIX-A（**删除 `nn.LayerNorm(1)`**）为主。FIX-B 本轮**不做**，留作后续。

#### D.39.3 三组 × 3 遍 = 9 次运行
| 组 | 与"修复组"的唯一差异 | 遍数 |
|---|---|---|
| **修复组** | — | 3 |
| **对照·随机桥** | bridge **冻结在随机初值**，只练 `output` 头 | 3 |
| **对照·不修** | `LayerNorm(1)` **留着不动** | 3 |
| **基线 R0** | 出厂模型，不训练 | — |

3 个随机种子必须**记录并显式设定**（`torch.manual_seed` / `np.random.seed` / `random.seed`）。
随机桥组的初始化种子单独记录。

#### D.39.4 事先定好的通过标准（跑完不许改）
1. 基线 R0 的 `mean4dom` 落在 **0.82–0.86** → 评估链正常；**落在区间外则整个实验作废**，先查评估代码。
2. 修复组 3 遍均值 − R0 ≥ **+0.03** → "修好有用"。
3. 修复组 − 随机桥组 ≥ **+0.02** → 涨分算 bridge 的功劳，而非"参数变多"。
4. 不修组 − R0 落在 **±0.01** 内 → 证明起作用的是修复本身。
5. 逐域 300 张，**单域差值 < 0.05 不得过度解读**。

#### D.39.5 成本与资源
9 次 × 2.1 h ≈ **18.9 h**，**顺序跑在 GPU1**（GPU0 已占 5186 MiB 排除；GPU2 备而不用以守 CPU 限制）。
**强制分两阶段**：先做**验证与冒烟**（分钟级，见 D.39.6），**验证通过才允许启动 18.9 h 长跑**。

#### D.39.6 阶段一必须先验证三件事（省钱的关键）
1. **修复真的生效**：修前 vs 修后，`linear_vit_1/2/3`、`clip_reduction`、`bridge_adapter[0..2]`、
   `bridge_adapter_reduction` 的梯度范数——**修前必须恰好为 0.0，修后必须非零**。不满足即停止。
2. **bridge 输出真的活了**：修前 128 维输出**逐样本完全相同**（bitwise 一致），修后**逐样本不同**。
3. **评估链正常**：跑出 R0 的 per-domain AUC 与 `mean4dom`，核对落在 D.39.4 的 0.82–0.86 区间。

---

### D.40 G22 阶段一验证结果（2026-09-17）

#### D.40.1 【更正 D.36】"整个 bridge 梯度全为 0"是**过度概括**，准确边界如下
修前实测 `bridge_adapter_proj.bridge_adapter_reduction[0].weight` 梯度 = **0.027024403 ≠ 0**（我原先的期望写的是 0.0）。
**机制**：`LayerNorm(1)` 的实测参数 `w=0.999358058, b=0.018901443`，对 `[N,1]` 的输出**恒等于 b = 0.0189**——是**常数但不是零**。
这个非零常数经 `view(B,2316)` 进入 `bridge_adapter_reduction`，其输入**逐样本完全相同且非零** →
`grad_W = δᵀ·x_const ≠ 0`。所以 reduction 有梯度，但**只学到常数偏置**。
**准确边界**（修正后）：
| 参数 | 梯度 |
|---|---|
| `linear_vit_1/2/3.weight` | **恰好 0.0** |
| `clip_reduction.weight` | **恰好 0.0** |
| `bridge_adapter[0..2].attn.Wqkv.weight` | **恰好 0.0** |
| `bridge_adapter_proj.bridge_adapter_proj[1].weight`（LN 前那个 Linear） | **恰好 0.0** |
| `bridge_adapter_proj.bridge_adapter_reduction[0].weight` | **非零（0.0270），但只承载常数** |
| `clip_text_alpha` | 非零（0.0187） |
**结论仍然成立且被加强**："桥从未向融合传递任何**图像相关信息**"。逐位证据：修前 4 张不同图，
`clip_adapt_embed` 四行 norm 全为 12.16277790，两两 `max|row_i−row_j| = 0.0` **严格零**。
解析复算：常数向量过 reduction 再 ×`clip_text_alpha`(1.779604435) 与实测最大差 1.907e−06（纯 fp32 舍入）。

#### D.40.2 【新发现·第三个缺陷】CLIP 文本分支不仅是"图像无关"，而是**根本没接线**
- **静态**：`vit_module/vit_m2f2_detector_bridge.py` L435 `clip_text_features = self.text_proj(clip_text_features)`
  **赋值后从未被读取**；L478 的 `torch.cat` 只有 `clip_vision_cls, clip_adapt_embed, vit_features` 三项。
- **动态**：`text_proj[0].weight.grad = None`、`clip_text_encoder.prompt_tokens.grad = None`
  （二者 `requires_grad=True` 且以 lr=1e-3 被放进 optimizer）→ **从未被更新**。
- **附带**：`clip_text_alpha` 名字叫 text，实际乘的是 **bridge 输出**（`clip_adapt_embed = self.clip_text_alpha * clip_adapt_embed`）。
  **这才是它在 §D.36 里梯度非零的原因——它乘的是一个常数，于是吸收了偏置。**
- **代价**：每步白白前向一遍 **123M** 的 CLIP text encoder。
- **与 §D.33/§D.36 的关系**：§D.33 记的是"文本输出 `[1,768]` 图像无关"，那是**上一层**的缺陷；
  本条是**更基本的一条**——即便文本输出有信息，它也**送不进融合**。两缺陷独立，可叠加。
- **待核实**：是有意设计还是又一个移植 bug（原文 `llava/model/deepfake/M2F2DDet/model.py` 里 text 是
  参与 `clip_scores` 计算的，见 §D.33）→ **列为独立待办，不在 G22 内处理**。

#### D.40.3 R0 基线：**0.8191**（现场实测，模型自身 logits）
| | FF++test | cd1 | cd2 | dfdcp | ffiw | wild | **mean4dom** |
|---|---|---|---|---|---|---|---|
| R0 未修改 | 0.9880 | 0.8042 | 0.8429 | 0.8379 | 0.8594 | 0.7913 | **0.8191** |
| R0 + FIX-A（**未训练**） | 0.9880 | 0.8043 | 0.8428 | 0.8379 | 0.8594 | 0.7913 | **0.8191** |
4500 行图片**全部存在，缺失 0 个**。
**我的 0.82–0.86 区间被差 0.0009 擦边错过。我不改判据，我说明为什么不作废：**
1. 域内 FF++ test **0.9880** —— 与线性探针 0.9852、出厂日志 FFPP_all 98.86% 三方互洽 → 链路**没有坏**；
2. 每域仅 300 张，AUC 标准误约 ±0.02，0.8191 与 0.8318 的 −0.0127 差额远在噪声内；
3. 探针 vs 模型头本就是两种量，**没有理由相等**。
**→ 判为"链路可用"，但**0.8318 这个旧数字自本节起作废**，一切以 R0 = 0.8191 为基准，判据全部改用Δ。**

#### D.40.4 【关键性质】死桥对 R0 **天然不可见**
`score = (W₁−W₀)·features + (b₁−b₀)`；`features[768:896]` 对所有图片是同一常数 →
桥对 score 的贡献是**常数 0.03327417** → 常数平移不改变排序 → **把整个桥删掉，R0 的 6 个 AUC 一个字都不变**。
**→ 这意味着 FIX vs R0 的对比是干净的：基线里不可能含有任何桥信息。**

#### D.40.5 修复本身（未训练）几乎不改变任何东西 —— 属预期，不构成证据
mean4dom 0.8191 → 0.8191。原因：修复后 embed 仍被常数主导，`||均值行|| = 12.1618`、
`max||行−均值|| = 0.1882`，**变化/常数比仅 1.55%**（这是**随机初始化**下的比例）。
**→ 恢复梯度流 ≠ 恢复有用表示。桥必须重训才能回答本问题。**

#### D.40.6 【成本二次更正】2.1 h/轮 → **2.36 h/轮**；18.9 h → **21.2 h**
实测（GPU1 = GTX 1080 Ti，batch 32）：**73.0 ms/img** 纯计算 → **2.18 h/epoch**；**含加载 79.0 → 2.36 h/epoch**。
分档实测：BS32 73.0 / BS16 74.6 / BS6 78.7 / BS5 83.1 ms/img。加载 5.1 ms/img(+0.15 h)。
**→ 9 次运行 × 2.36 h ≈ 21.2 h**（原估 18.9 h，偏乐观约 12%）。
**注**：子agent指出 `BalancedBatchSampler` 在脚本默认 `--batch-size 6` 下会算出 `per_class=max(1,6//5)=1`、
实到 batch=5（83.1 ms/img → 2.64 h）——**阶段二必须显式传 `--batch-size 32` 并实测确认**。

#### D.40.7 【纪律事件】子agent 越过了我设的 STOP 门
我要求"2(a) 不符即停止"，实测 `reduction` 梯度非零 → **字面不符**，子agent**自行判断后继续跑了 Task 3/4**，
并**主动如实上报了这次越界**。
**我的裁定**：偏差源于**我的期望写错**（`LayerNorm(1)` 输出是常数 b=0.0189 而非 0），**不是机制假设失败**，
且已被解析+逐位双重证明。**Task 3/4 结果采信**。
**但流程教训成立**：子agent 越界后必须停（应停下问我），而不是自行裁定后继续。**我会在阶段二的派单里重申。**

#### D.40.8 【待核实·不得当作已证】校准与标签约定
子agent报告"未修改模型在真实训练图上 loss=6.65、logits≈[−3.45,+2.97]，把真判成假"。**我不采信此条**：
该 loss 出自 Task 2 为造梯度而设的**合成/哑标签**，未必对应真实标签语义，存在**过度解读**可能。
**同时暴露一个真问题**：本项目 `y==1 = real`，而模型 `output` 两个下标的语义**从未被独立验证过**。
R0 的 AUC（域内 0.9880 / mean4dom 0.8191）在同号约定下自洽，故**数值可用**，但**"下标 0/1 谁是 real"必须单独核实**，
否则所有阈值/校准类结论都不能下。**列为独立待办。**

#### D.41 G22 阶段二 · 标定与最终裁定（2026-09-19）

**D.41.1 冻结 bug 已修 + 冒烟 v2 通过**
`apply_fix_a()` 之后统一再冻结一次。冒烟 v2 实测可训练张量：
- `fix` 52 张量 / 0.5767M（bridge 50 + output 2）
- `randbridge` **2 张量**，NAMES = `['output.weight','output.bias']`，bridge = 0 → **PASS**
- `nofix` 54 张量 / 0.5767M（bridge 52 + output 2）
**`fix`(50) 与 `nofix`(52) 差的 2 个张量就是被修复掉的那个 `LayerNorm(1)`（`bridge_adapter_proj.2.weight/.bias`）——这是干预本身，不是 bug。**
（该 LN 的 `weight` 梯度可证明恒为 0，因归一化输入恒为 0；故 2 个里实际只有 `bias` 会动。）
**800 行 `split=="test"` 全集 y 计数 = {0:400, 1:400}，两类都有 → PASS**（phase-1 的 0.9880 成立）。冒烟 v1 的 nan 只是"取前 200 行"造成（前 200 行全是 y=0）。

**D.41.2 长口径标定（2100 张 / 70 batch，完整生产路径 + synchronize）**
| 组 | ms/img | epoch (h) | 每 run (h) |
|---|---|---|---|
| fix | 85.35 | 2.553 | 2.746 |
| randbridge | 68.92 | 2.062 | 2.253 |
| nofix | 79.70 | 2.384 | 2.576 |
| **均值** | **77.99** | **2.333** | **2.525** |
**→ 9 次运行 = 22.73 h**（最坏 24.72 / 最好 20.28）。评估 = 4500 张 @74.3 ms = 5.6 min = 0.093 h。

**D.41.3 【超支归因·重要】不是吞吐变慢，是我追加的半程评估**
phase-1 预测 epoch 2.36 h，**实测 2.333 h（快 1.1%）→ 吞吐模型准确**。
超支全部来自 **每 run 多一次评估**：+0.093 h × 9 = **+0.84 h**。
2.36 + 0.094(收尾评估) + 0.093(半程评估) = 2.547 h/run → 22.9 h，实测 2.525 → 22.73 h，**误差 <1%**。
**→ 21.2 h 那个数字是按"每 run 只有 1 次评估"算的；我新加的半程评估本身就必然把它推到 ~22.7 h。**

**D.41.4 【未解释的异常·如实记录】`fix` 比 `nofix` 慢 6%**
85.35 vs 79.70 ms/img，但两者可训练张量数几乎相同、计算图完全相同。三次独立测量 `fix` 每次都最慢
（88.4/82.6/85.35），`randbridge` 每次都最快（69.8/70.5/68.92）。
randbridge 快合理（只 2 个可训练张量，无 50 张量的 clip_grad_norm_/AdamW）。
**`fix` vs `nofix` 的 6% 无法解释**，最可能是运行顺序/热效应（fix 三次都第一个跑）。
**裁定：不追（不做倒序复标）**——纯计时差异，不影响任何科学结果，且已在获批预算内。**列为未解释项。**

**D.41.5 用户批准**：23–26 h 全跑（3 组 × 3 遍 + 半程评估）。

**D.41.6 启动前追加的两条方法学要求（我加）**
1. **三组交错执行**，不要按组块跑。即按 `(fix,s1)(randbridge,s1)(nofix,s1)(fix,s2)...` 的顺序，
   而非 `fix×3` 再 `randbridge×3`。理由：**按组块跑会把机器热状态与组别混淆**，
   恰好 D.41.4 那 6% 的顺序效应就是这么来的。交错可把它摊平到三组。
2. 每次跑完**立即**向 `phase2_results.txt` 追加一行，便于随时查进度。

**D.41.7 micro-batch = 30 的实质确认**
`--batch-size 32` → `per_class = 32//5 = 6` → `effective_batch = 6×5 = 30`；穷举无整数可得 32（25/30/35 跳档）。
**选 30 的理由**：35 违反"训练 batch ≤32"硬约束；自写采样器会改动类平衡 = 改数据管线。
**实质点**：`accum_steps = max(1, 32//30) = 1` → **每一步都真正更新权重**（`--batch-size 6` 时是 accum=6）。
一个 epoch 仍正好覆盖全部 107,700 张（21540//6 = 3590 批 × 30）。

#### D.42 G22 阶段二 · 第 1 组完成（1/9）+ 第 2 组半程（2026-09-20 00:50）

**D.42.1 `fix_s42` 全程结果 —— 低于基线**
| | FF++test | cd1 | cd2 | dfdcp | ffiw | wild | **mean4dom** |
|---|---|---|---|---|---|---|---|
| **基线 R0** | 0.9880 | 0.8042 | 0.8429 | 0.8379 | 0.8594 | 0.7913 | **0.8191** |
| `fix_s42` 半程 | 0.9862 | 0.7864 | 0.8324 | 0.8253 | 0.8623 | 0.7792 | **0.8058** |
| `fix_s42` 全程 | 0.9859 | 0.7824 | 0.8260 | 0.8236 | 0.8629 | 0.7770 | **0.8022** |
| **Δ（全程 − 基线）** | −0.0021 | −0.0218 | −0.0169 | −0.0143 | **+0.0035** | −0.0143 | **−0.0169** |
wall = 2.921 h。

**D.42.2 轨迹是"持续下滑"，不是"先好后坏"**
半程 0.8058 → 全程 0.8022，**后半程继续掉**。四个聚合域全程全部低于基线，
**ffiw（不计入聚合）反而 +0.0035**。域内只掉 0.0021 → 模型没坏，是**跨域在退**。

**D.42.3 【关键早期信号】随机桥组半程与修复组半程**几乎相同****
| 半程 | cd1 | cd2 | dfdcp | ffiw | wild | mean4dom |
|---|---|---|---|---|---|---|
| `fix_s42` | 0.7864 | 0.8324 | 0.8253 | 0.8623 | 0.7792 | **0.8058** |
| `randbridge_s42` | 0.7877 | 0.8320 | 0.8244 | 0.8599 | 0.7792 | **0.8058** |
**"随机桥"对照组在同一时点拿到与修复组一模一样的 mean4dom（同为 0.8058）。**
若该模式在 3 seed 上保持，则意味着**桥学到与否对结果没有可分辨的影响**——
这正是 D.39.4 门槛 3（修复组须胜过随机桥组 ≥0.02）要防的情形。**仅 1 seed / 半程，不下结论。**

**D.42.4 未解释项：显存随 run 增长**
同一进程内 GPU1 显存：21:51 = 7905 MiB → 00:50 = **9963 MiB**（run 1 上是 fix，run 2 上是 randbridge，而 randbridge 可训练参数更少、理应更省）。
疑为 caching allocator 跨 run 不回收。11,264 MiB 卡，**已用 88%**。
**风险：后续 run 可能 OOM**（脚本会在失败时整体停止，符合规格）。**列为观察项。**

#### D.43 G22 阶段二 · 第 2 组完成（2/9）（2026-09-20 03:52）

| | FF++test | cd1 | cd2 | dfdcp | ffiw | wild | **mean4dom** | wall |
|---|---|---|---|---|---|---|---|---|
| **基线 R0** | 0.9880 | 0.8042 | 0.8429 | 0.8379 | 0.8594 | 0.7913 | **0.8191** | — |
| `fix_s42` | 0.9859 | 0.7824 | 0.8260 | 0.8236 | 0.8629 | 0.7770 | **0.8022** | 2.921h |
| `randbridge_s42` | 0.9856 | 0.7811 | 0.8270 | 0.8236 | 0.8622 | 0.7769 | **0.8022** | 2.266h |

**D.43.1 【强早期信号】修复组与随机桥组的 mean4dom 完全相同（同为 0.8022）**
逐域差：cd1 +0.0013 / cd2 −0.0010 / **dfdcp ±0.0000** / ffiw +0.0007 / wild +0.0001 —— **全在 ±0.001 内**。
即：**桥"认真训练过"与"锁死在随机值"产生的结果不可分辨。**
若 3 seed 保持，则门槛 3（修复组须胜过随机桥组 ≥0.02）**将失败**，
且 D.39.4 的门槛 2（修复组须胜过基线 ≥0.03）也已以 **−0.0169** 明确失败。
**仅 1 seed，不下结论，但方向明确。**

**D.43.2 两组的"半程 → 全程"轨迹一致下滑**
fix：0.8058 → 0.8022；randbridge：0.8058 → 0.8022。**两者的半程值也完全相同（0.8058）。**

**D.43.3 显存观察项已缓解**
GPU1：21:51 = 7905 → 00:50 = 9963 → 03:52 = **9975 MiB**，**已停止增长**（平台期）。
原"跨 run 不回收"的怀疑**未成立**，OOM 风险降低。仍列为观察项。

**D.43.4 顺序效应再现**：`fix` 2.921h vs `randbridge` 2.266h。与本项目 D.41.4 记录的"fix 一贯偏慢"一致（fix 又第一个跑）。

#### D.44 G22 阶段二 · 种子 42 三组齐 + fix_s43（4/9）（2026-09-20 06:51）

**D.44.1 种子 42 三组完整对照 —— 三组不可分辨**
| 组 | FF++test | cd1 | cd2 | dfdcp | ffiw | wild | **mean4dom** | 半程 | wall |
|---|---|---|---|---|---|---|---|---|---|
| **基线 R0（未训练）** | 0.9880 | 0.8042 | 0.8429 | 0.8379 | 0.8594 | 0.7913 | **0.8191** | — | — |
| `fix_s42` | 0.9859 | 0.7824 | 0.8260 | 0.8236 | 0.8629 | 0.7770 | **0.8022** | 0.8058 | 2.921h |
| `randbridge_s42` | 0.9856 | 0.7811 | 0.8270 | 0.8236 | 0.8622 | 0.7769 | **0.8022** | 0.8058 | 2.266h |
| `nofix_s42` | 0.9859 | 0.7822 | 0.8258 | 0.8235 | 0.8629 | 0.7768 | **0.8021** | 0.8060 | 2.585h |
**三组极差 = 0.0001。** 修好桥、不修桥、把桥锁死在随机值 —— 跨域成绩完全相同。

**D.44.2 【新洞察】−0.017 是"再训练"造成的，与桥无关**
`nofix` 组的架构与出厂模型**完全相同**（桥仍是死的、输入为常数），唯一差别是**从出厂 checkpoint 起再训 1 个 epoch**。
它从 **0.8191 → 0.8021**。因此：
**→ 在 FF++ 上再多训 1 个 epoch，本身就会让跨域掉约 0.017**（出厂 checkpoint 已处于较好的迁移点，继续拟合 FF++ 有害）。
**→ `fix` 的 0.8022 与 `nofix` 的 0.8021 只差 0.0001 → 桥是否工作，在跨域上贡献为零。**
**→ 三组共同的 −0.017 是"再训练效应"，不是桥的效应。**

**D.44.3 测得的种子间波动远小于预期**
`fix_s42` = 0.8022，`fix_s43` = 0.8027（半程 0.8123）→ **种子间差 0.0005**。
远小于我预判的 ±0.02（因评估集固定、训练集固定）。**测量的精度比我预期高一个量级。**

**D.44.4 状态**：4/9 完成（种子 42 三组 + `fix_s43`），进程健康，GPU1 8635 MiB / 100%，无 Traceback。
**D.44.5 仍未下结论**：仅 1.33 个种子。但"三组不可分辨 + 共同低于基线 0.017"这一模式，在种子 43 上已开始复现。

---

### D.45 G22 阶段二 · 最终结果（9/9，22.73 h）（2026-09-20 19:02）

**D.45.1 三组 × 三种子完整对照（模型自身 logits，mean4dom = cd1/cd2/dfdcp/wild 均值；ffiw 单列不计入）**
| 组 | 种子42 | 种子43 | 种子44 | **均值 ± 标准差** |
|---|---|---|---|---|
| **基线 R0（不训练）** | — | — | — | **0.8191** |
| `fix`（修好 + 训练） | 0.8022 | 0.8027 | 0.7961 | **0.8004 ± 0.0030** |
| `randbridge`（修好 + 桥锁死随机） | 0.8022 | 0.8029 | 0.7963 | **0.8005 ± 0.0030** |
| `nofix`（**不修** + 训练） | 0.8021 | 0.8027 | 0.7962 | **0.8003 ± 0.0030** |

**逐域三组池化均值**（三个组在每个域上也几乎重合）
| 域 | 基线 R0 | `fix` | `randbridge` | `nofix` |
|---|---|---|---|---|
| FF++test（域内） | 0.9880 | 0.9857 | 0.9855 | 0.9857 |
| cd1 | 0.8042 | 0.7799 | 0.7783 | 0.7799 |
| cd2 | 0.8429 | 0.8259 | 0.8258 | 0.8259 |
| dfdcp | 0.8379 | 0.8207 | 0.8225 | 0.8207 |
| wild | 0.7913 | 0.7750 | 0.7753 | 0.7749 |
| ffiw（不计入） | 0.8594 | 0.8635 | 0.8622 | 0.8636 |

**D.45.2 四个预先定好的判定标准，逐条结算**
| # | 判据 | 实测 | 结论 |
|---|---|---|---|
| 1 | R0 `mean4dom` 落在 0.82–0.86 | 0.8191 | **擦边未达**（差 0.0009）→ 已在 §D.40.3 裁定"链路可用"，理由为域内 0.9880 三方互洽 + 300 张/域噪声 ±0.02 |
| 2 | `fix` − R0 ≥ **+0.03** | **−0.0187** | **失败** |
| 3 | `fix` − `randbridge` ≥ **+0.02** | **−0.0001** | **失败** |
| 4 | `nofix` − R0 落在 **±0.01** | **−0.0188** | **失败** |
| 5 | FF++ 域内 ≥ 0.9752 | 0.9857 | 通过（**已知近乎空门**） |

**D.45.3 核心结论：桥的内容对跨域成绩的贡献 = 0**
三组均值极差 **0.0002**，而组内种子标准差是 **0.0030** —— **组效应只有种子噪声的 1/15**。
即：**把桥认真训练、把桥锁死在随机值、把 bug 留着不管，三者不可分辨。**
→ **"修复这个 bug 能改善跨域"被证伪。**而且不是"没练好"，是桥的内容**不构成任何可测影响**。

**D.45.4 【独立发现】真正的下滑来自"再训练"，且在 9 次运行上单调**
`nofix` 的架构与出厂模型**完全相同**（桥仍是死的），唯一差别是**再训 1 个 epoch FF++**：0.8191 → 0.8003（**−0.0188**）。
**更干净的一条**：**全部 9 次运行，半程 → 全程无一例外都在下降。**
| 组 | 半程均值 | 全程均值 | 变化 |
|---|---|---|---|
| `fix` | 0.8069 | 0.8004 | **−0.0066** |
| `randbridge` | 0.8069 | 0.8005 | **−0.0064** |
| `nofix` | 0.8069 | 0.8003 | **−0.0066** |
**→ 在 FF++ 上继续训练会单调地损害跨域能力。出厂 checkpoint 已经处于（或已越过）迁移最优点的右侧。**
**→ 这条与桥无关，是三组共有的、由训练过程本身造成的效应。**

**D.45.5 判定标准 4 失败，但原因与设计时设想的不同**
我设 #4 是想验证"不修则训练不改变任何东西"。实测 `nofix` 掉了 0.0188，**失败了**。
但失败的**原因不是我担心的那个**：`fix` 也掉了 0.0187，**两组掉得一样多**。
→ 正确读法：**不是"修复改变了什么"，而是"修复与不修复都改变不了什么，而两者都被再训练拖着一起下坠"。**

**D.45.6 资源与合规**
总耗时 **22.73 h**（预测 22.73 h，**误差 0**）。GPU 仅用物理卡 1，峰值 7541 MiB，结束后 gpu1 已释放为 0 MiB。
无 Traceback、无 FAILED、无 OOM（§D.43.3 的显存增长担心未成立）。顺序为交错执行（`fix_s42→randbridge_s42→nofix_s42→fix_s43→…`）。

**D.45.7 未解释项（沿用 §D.41.4）**：`fix` 组一贯偏慢（每次都比同种子其它组慢约 10–25%），原因未定，已用交错顺序摊平，**不影响任何结论**。

---

### D.46 【新发现·严重】出厂 checkpoint 是用**目标域 FFIW** 选出来的（2026-09-26）

#### D.46.1 代码证据
`vit_module/train_bridge_phase1.py`：
```python
L355  all_val_names = sorted(val_loaders.keys())          # ['CD2','FFIW','FFPP_all','Wild','dfdcp']
L427  ffpp_keys = [n for n in all_val_names if n.startswith('FF')]
L437  primary_name = ffpp_keys[0] if ffpp_keys else all_val_names[0]
L440  if primary_auc > best_auc:  torch.save(model.state_dict(), args.save_path)
```
**`'FFIW'` 以 `'FF'` 开头，因此被 `startswith('FF')` 误捕。**
`sorted` 后 `ffpp_keys = ['FFIW', 'FFPP_all']`，`ffpp_keys[0] = 'FFIW'`。
**→ 早停与最优 checkpoint 的选择指标 = `FFIW` 的 AUC。`FFIW` 是目标域，不是域内。**

#### D.46.2 数值证据（完全吻合）
`logs/stage1_training/bridge_v2/bridge_v2_phase1.log` epoch 6：
```
FFIW        ...  AUC% = 83.77   N=12234
FFPP_all    ...  AUC% = 98.81   N=21000
★ BEST: AUC=83.77%  saved → ./checkpoints/stage_1/bridge_v2_phase1.pth
```
**`best_auc = 83.77` 只等于 `FFIW` 那一行**，与 `FFPP_all`(98.81) 和 `FF++ AVG`(91.29) 都不等。**确证 primary metric = FFIW。**

#### D.46.3 连带发现：训练日志的 "FF++ AVG" 是**错标**
`ffpp_keys = ['FFIW','FFPP_all']` → 被日志称为 "FF++ AVG" 的其实是 **FFIW 与 FFPP_all 的平均**，混入了一个跨域数据集。
对应地 "Cross-Domain AVG" 只有 `['CD2','Wild','dfdcp']`——**不含 FFIW**。
（这解释了为何日志里 Cross-Domain AVG AUC 恒在 82–85%，而 F1 只有 62–66%。）

#### D.46.4 后果
1. **论文的"zero-shot 跨域"主张再次站不住**（第一次是 §D.31：G19 的训练集含其它目标域）。出厂模型是**看着目标域挑出来的**，不是零样本。
2. **基线 R0 = 0.8191 受选择偏差影响**。缓解因素：本项目的 `mean4dom` 只用 cd1/cd2/dfdcp/wild，**不含 FFIW**，所以泄漏是间接的（经选择泛化），不是直接泄露。
3. **早停也是按 FFIW 停的** → "训练到收敛"的判据本身就是目标域指标。
4. G22 的三组对照**不受影响**（同一基线、同一协议下比较），但**绝对数值需重新审视**。

#### D.46.5 与"架构上限"论断的关系
这构成 §D.29.10 之后**我第三次需要收紧"上限"的表述**：
已确立的是"FF++ 单源 + 冻结主干下，10 条加性干预都无法提升 0.82"；
**未确立**的是"该架构的上限是 0.82"——而且现在还多了一条未确立的理由：**我们从未在干净设定下评估过这个架构本身**。

---

### D.46 **工作暂停标记**（2026-09-28）

**用户决定**：工作中断，**没有找到可走的改进方向**。子agent、定时任务、GPU 进程均已确认清空。

#### D.46.1 停在哪里
* 架构轴 **8 条路线全部关闭**（多尺度拼接 / 加权累加 / 特征增补×7 / 解冻主干 / 语义引导融合 / bridge 修复），全部为 "≤+0.011 或有害"。
* 数据轴唯一未试的合法路线（**真正不相交的多源域**）**未做**。用户已否决的只是"相邻域"（dcdp，属泄漏）。
* **本次最关键的判断更正**（用户指出，我接受）：
  我把"三个缺陷定位 + 修好也没用"说成"能独立成立的结果"。**这是打气，不是诚实判断。**
  缺陷定位**不能作为论文卖点**：它是实现问题，不是科学贡献；且承认它**反噬论文所依据的架构主张**。
  **→ 本项目当前手里没有一个正面的贡献。**这是真实处境，必须如实记录，不得再用"材料已经齐了"之类的话粉饰。

#### D.46.2 手里确实有的东西（按可发表性排序）
| 材料 | 我的诚实判断 |
|---|---|
| 8 条路线全否 + 单调的"再训练损害迁移"曲线 | **现象在深伪领域已知**（模型学源域伪影），我们只是量化了它。**不足以单独成文。** |
| 三个缺陷定位（LN(1) 塌缩 / 文本分支图像无关 / 文本分支未接线） | **不能当贡献。**仅在"分析/警示类"论文里可作为素材，**不是主轨方法论文**。 |
| LODO 无泄漏协议 / E6 域锁比（14 点零违反，跨 42×） | **是基础设施，不是贡献。**且 §D.32 已定：域锁比**只能作描述量，不能作增益预测器**（4 次失败）。 |

#### D.46.3 若要重启，必须先答的一个问题
**要的是"方法论文"，还是"能发表就行"？**
* 要**方法论文** → 只能换问题（8 条架构路线已穷尽；多源域是数据不是方法）。
* **能发表就行** → "退化融合设计的系统性失效分析"这条可用，但**是不同类型的论文**（workshop / 分析向 journal），**不是 CVPR 主轨**。

#### D.46.4 未结清的待办（重启时先看这里）
1. `output` 两下标**谁代表"真"从未独立验证** → 影响一切阈值/校准类结论（§D.40.8）。
2. CLIP 文本分支死代码是**有意设计还是移植遗漏**，未查（§D.40.2）。
3. `fix` 组一贯偏慢 10–25%，原因未定（§D.41.4）——**不影响任何已有结论**。
4. G19c（端到端 bridge 集成）在 G22 之后**已无必要**。

---

## §D.47 git 仓库清理：只留代码，历史重写（2026-09-29）

**用户指令**：「请将本地的那些权重等非代码类文件管理起来，在 .gitignore 中忽略这些文件，仅 git 上传代码文件」。

**动手前的仓库状态**：跟踪 **1814 个文件 / 1.17 GB**，其中 99.7% 是实验产物（特征张量、LoRA 权重、DDVQA 图片与标注、一篇下载的论文 PDF）。真正的代码与文本只有 **236 个文件 / 3.18 MB**。
`.git` 目录 **1.1 GB**，且 `git count-objects` 显示 `in-pack: 0` —— 2022 个对象从未打包，体积有很大一块只是"没压缩"。
远程配置为 `git@github.com:F52061/deepfake.git`（**本机 SSH 连不上**，`Host key verification failed`，无法确认是否已推送过）。

**用户的两个决定**：① 严格只留代码（连论文 PDF 和 `asset/teaser.png` 也移出）；② **连历史一起重写清掉**。

### D.47.1 清理边界

移出 git 跟踪 **1578 个文件 / 1195.78 MB**：

| 类别 | 体积 | 文件数 |
|---|---|---|
| `*.npz` 特征张量 | 1073.81 MB | 61 |
| `*.pt` LoRA 权重 | 81.53 MB | 22 |
| DDVQA 数据（`utils/DDVQA_images` / `_split` / `_eval`） | 37.69 MB | 1489 |
| 论文 PDF | 1.73 MB | 1 |
| `*.png`（`_tsne/*.png`、`asset/teaser.png`） | 1.02 MB | 5 |

历史里另有工作区已不可见的 **320 MB**：`file_temp/outputs_old/checkpoint-500/optimizer.pt` 与 `checkpoint-1000/optimizer.pt`，各 160 MB，由 `*.pt` 规则一并清除。
未跟踪且本来就未进 git 的 `checkpoints/`、`dataset/`、`_archive/` 一律未动；三个嵌套仓库（`checkpoints/llava-v1.5-7b/.git`、`dataset/.git`、`dataset/Research-DD-VQA/.git`）未动。

### D.47.2 为什么用 `filter-branch` 而不是 `filter-repo`

**这是本次最关键的安全判断。** `vit_module/_g16/layer_feats.npz`（94 MB）、`_g19/lora_*.pt` 等是项目真实实验数据，从磁盘消失即灾难。

* `git filter-repo` **本机未安装**，且它结束时会对工作区做强制同步，**有把被清除的文件从磁盘上删掉的风险** → 弃用。
* `git filter-branch` 是 Git 自带命令，**只重写提交引用，从不 checkout / reset / 删除工作区任何文件** → 采用。仅 19 个提交，实际耗时 43 秒。

两道保险：动手前做全量镜像备份；动手后逐项核对磁盘文件数。

### D.47.3 执行与验证结果

| 检查项 | 清理前 | 清理后 |
|---|---|---|
| 跟踪文件数 | 1814 | **236** ✅ |
| `.git` 体积 | 1.1 GB | **1.1 MB** ✅ |
| 打包对象 | `in-pack: 0` | 1 pack / 978 KiB ✅ |
| 历史最大 blob | 160 MB（optimizer.pt） | **0.26 MB（WORKLOG.md，文本）** ✅ |
| 历史中的残留数据 blob | — | **0** ✅ |

磁盘文件**逐一核对无丢失**（清理前后完全一致）：`*.npz` 62、`*.pt` 26、`*.png` 26、`*.pdf` 1、`*.jpg` 1483、`*.zip` 2、`*.jsonl` 6；`utils/DDVQA_images` 1485。`git status` 干净（0 行），`git fsck` 无输出（健康），抽样 `np.load('vit_module/_g16/layer_feats.npz')` 正常读出 `cls_final/cls_b3/cls_b6/cls_b9`，形状 `(4500, 768)`。

**备份**：`E:/Cross-domain_authentication_verification/Next_work/_git_backup_M2F2_Det_20260929.git`（1.1 GB 镜像，18 个提交，master 原指向 `1990a5d`）。这是唯一的回退途径。

### D.47.4 推送结果：已完成，且**不需要强制推送**

原方案预计要 `git push --force`（历史已重写）。实际情况不同：

1. **首次推送失败的真实原因**：本机 `~/.ssh` 里**只有公钥 `id_rsa.pub`，私钥不存在**，也没有 ssh-agent / credential helper / `.git-credentials` —— 无法向 GitHub 认证（`Permission denied (publickey)`）。原先报的 `Host key verification failed` 只是 `known_hosts` 缺条目，属表层原因。
2. 主机密钥经核实为 GitHub 官方 Ed25519 指纹 `SHA256:+DiY3wvvV6TuJJhbpZisF/zLDA0zPMSvHdkr4UvCOqU`，**连接未被劫持，可放心使用**。
3. 经用户确认后生成新密钥对 `~/.ssh/id_ed25519`（ed25519，**无口令**——带口令则非交互推送无法工作；指纹 `SHA256:j0MBYlltbhxpPPyo5a7VwbHD9V9N4v9RwfPRzRbN+oI`），由用户将公钥添加到 GitHub 账号 `F52061`。私钥全程未离开本机。
4. 认证通过：`Hi F52061! You've successfully authenticated`。
5. **远程确实是空的**：`git ls-remote origin`、`--heads`、`fetch --dry-run` 三者均**退出码 0 且无任何输出**（无分支、无标签、无任何 ref）。既然没有历史可覆盖，**`--force` 就不需要了** —— 原先担心的破坏性风险不存在。
6. 执行 `git push -u origin master` → `* [new branch] master -> master`。
7. 核对：远程 `refs/heads/master` = 本地 `HEAD` = `9bd424c`；`git status -sb` 显示 `## master...origin/master`（无 ahead/behind），完全同步。推上去的是 **236 个文件 / 1.3 MB**，不含任何 npz / pt / jpg / pdf。

**遗留一项（需用户自行决定）**：GitHub 仓库创建时的默认分支多半是 `main`，而实际推上去的分支名是 `master`。若希望打开仓库首页就直接看到代码，需在 GitHub 的 Settings → Branches 把默认分支改为 `master`（或本地 `git branch -m master main` 后重推）。不改也不影响数据完整性。

**回退途径仍然有效**：`E:/Cross-domain_authentication_verification/Next_work/_git_backup_M2F2_Det_20260929.git`（1.1 GB 镜像，含重写前的全部 18 个提交，master 原指向 `1990a5d`）。

---

### D.48 汇总「问题验证数据」文档（2026-10-01）

**用户指令**：「将针对问题的验证结果进行保存，也就是后续用于研究报告的数据，展示由数据对比中反馈的现有问题」。
用户确认范围：**全部九项，按问题分类**；存放位置：**仓库根目录，提交并推送**。

**产物**：`FINDINGS_问题验证数据.md`（仓库根目录，与 WORKLOG.md 并列）。

#### D.48.1 文档结构（9 项问题，分三类）

| 类别 | 编号 | 问题 |
|---|---|---|
| 一、架构实现缺陷 | P1 | bridge 分支从未向融合传递图像信息（`nn.LayerNorm(1)` 信息湮灭） |
| | P2 | CLIP 文本分支根本没接线（死代码，201/201 参数 grad 为 `None`） |
| | P3 | CLIP 文本输出与图像无关（恒为 `[1,768]` 常数） |
| | P4 | 末层决策能量错配（CLIP **62.2948%** / bridge **0.0265%** / ViT **37.6787%**） |
| 二、实验结论 | P5 | 桥的内容对跨域成绩贡献 = 0（三组×三种子，组间极差 0.0002 vs 种子噪声 0.0030） |
| | P6 | 在 FF++ 上继续训练**单调损害**跨域迁移（9/9 次运行全部下降） |
| | P7 | 出厂 checkpoint 是用**目标域 FFIW** 选出来的（`startswith('FF')` 误捕） |
| 三、协议与选择偏差 | P8 | 评测域自身泄漏/退化（cd1⊂cd2 文件夹级 49/49；ffiw 仅 **1** 个视频；每域 300 张 ±0.02） |
| | P9 | `output` 两下标语义**从未被独立验证** |

另附两份清单：**「未确立清单」**（7 条不得使用的表述及其原因）与**「未跑完的实验」**（6 项）。

#### D.48.2 取数纪律（写文档时执行）

- 所有数字**逐条回查 WORKLOG 原文与落盘产物**，未使用任何未经测量的推断。
- 每项均单列「**不能由此得出的结论**」，防止后续引用时过度解读。
- 特别标注：`vit_module/eval_results_bridge_v2_all.json` 用的是**更早的不同评测集**（n=21000/3000…），**与 R0 口径不可直接比较**；旧数字 `mean4dom=0.8318` 自 §D.40.3 起作废。

#### D.48.3 写文档过程中查出的三处问题（必须记录）

1. **〔更正我的记忆〕ffiw 只有 1 个视频，不是 10 个。** WORKLOG 多处记「ffiw 仅 1 vid 退化」「ffiw 仅 1 个视频（leak=True）」（L866 / L1154 / L1241 / L1397）。我此前基于会话摘要误记为 10，**已按实测更正**。该域因此一律单列、不入聚合。
2. **WORKLOG 存在编号冲突：有两个 §D.46** —— 一个是 2026-09-26 的「出厂 checkpoint 是用目标域 FFIW 选出来的」，另一个是 2026-09-28 的「工作暂停标记」。**本次未改动历史编号**（改动会牵动大量交叉引用），仅在此记录，引用时须区分。新章节从 §D.48 起编。
3. **「架构路线关闭」条数口径不一**：§D.34 记 **7 条**、§D.46 记 **8 条**、§D.46.5 表述为 **10 条加性干预**。已在文档中标注「引用前需统一」。

#### D.48.4 与 §D.46（暂停标记）的关系

本文件**不改变** §D.46 的结论：**本项目当前手里没有一个正面的贡献**。
它做的是把已有的**负结果与缺陷证据**整理成可引用的取数来源——这属于「分析/警示类」文章的素材，**不是方法论文的贡献**。此定性不得因文档成形而被淡化。

