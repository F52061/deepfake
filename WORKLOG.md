---
name: M2F2-Det-stage3-worklog
description: M2F2-Det 项目 Stage-3 微调工作日志与检查点 —— 环境、配置、进度、待办、可恢复状态
metadata:
  type: project
  updated: 2026-07-31
  status: stage3-finetuning-running
---

# M2F2-Det 项目工作日志（检查点格式）

> 本文档作为**项目检查点**，供后续会话/其他模型读取以恢复上下文。修改代码前先读此文件。
> 项目路径: `E:\Cross-domain_authentication_verification\Next_work\M2F2_Det-main-hyy`

---

## 1. 项目目标

将 **M2F2-Det**（CVPR 2025 多模态伪造人脸检测器）用自己的 **ViT + BridgeAdapter 检测器**（方案B）复现三阶段流程：
- Stage-1（✅ 完成）：检测器预训练（二分类）
- Stage-2（↩️ 跳过）：MLP 多模态对齐 —— 因 Stage-3 的 `--tune_deepfake_mlp_adapter` 会训练 MLP，故从骨架直接进入 Stage-3
- Stage-3（🔄 进行中）：LoRA 微调 LLaMA，输出检测+解释

---

## 2. 环境与硬件

| 项目 | 值 |
|------|-----|
| 操作系统 | Windows 10 Enterprise 10.0.19045 |
| conda 环境 | `M2F2_Det` (`C:\Users\Supor2\.conda\envs\M2F2_Det`) |
| Python | 3.10.14 |
| PyTorch | **2.2.2+cu118**（曾误装 2.13.0+cpu，已重装 CUDA 版） |
| torchvision | 0.17.2+cu118（已 patch 移除 `_meta_registrations` import） |
| transformers | 4.37.0 |
| flash_attn | ✅ 可用 |
| deepspeed | ❌ 未安装（Windows 编译困难，不可用） |
| bitsandbytes | 0.43.0（8-bit 前向可用，但反向传播 dtype 不兼容） |
| GPU | 4× GTX 1080 Ti (11.8GB/卡)，Pascal 架构（**不支持 bf16**） |

### 关键约束（踩坑结论）
1. **Windows PyTorch 不读取 Python 内 `os.environ["CUDA_VISIBLE_DEVICES"]`** —— 必须用 `.bat` 在进程启动时设置系统环境变量
2. **fp16 + transformers GradScaler 与冻结 backbone 冲突** —— 报 `"Attempting to unscale FP16 gradients"`；需 no-op GradScaler monkeypatch（见 §7）
3. **多卡 `dispatch_model` 下 trainer 日志显示 loss=0.0 是 bug** —— 真实 loss 用 `[DBG-LOSS]` 打印判断（见 §7）
4. **8-bit 反向传播 dtype 不兼容**（Half vs Float）—— 不可用于训练，仅前向
5. 1080 Ti 不支持 bf16，不能用 `--bf16`

---

## 3. 权重文件清单

| 权重 | 路径 | 大小 | 状态 | 用途 |
|------|------|------|------|------|
| ViT backbone | `E:\Cross-domain_authentication_verification\PDI\results\Ama1_aps1_1\net_050.pth` | 365MB | ✅ | Stage-1 检测器 ViT 初始化 |
| 方案B检测器 | `checkpoints/stage_1/bridge_v2_phase1.pth` | 2.0GB | ✅ **最终选型** | Stage-3 检测器 |
| 方案A检测器 | `checkpoints/stage_1/cosine_realclip_phase1.pth` | 2.0GB | ✅ 备用 | cosine 融合对比 |
| CLIP 本地 | `checkpoints/clip-vit-large-patch14-336/` | 1.6GB | ✅ | 离线 CLIP（重定向来源） |
| LLaVA 基座 | `checkpoints/llava-v1.5-7b/` | ~20GB | ✅ | Stage-3 基座（已下载） |
| 骨架权重 | `checkpoints/llava-1.5-7b-deepfake-rand-proj-v1/` | ~14GB | ✅ | Stage-3 起点（含 bridge_v2 + LLaVA） |
| M2F2-Det 成品 | `checkpoints/llava-v1.5-7b-M2F2-Det/` | 13.6GB | ✅ | 参考（官方成品，含 model-00003） |

---

## 4. Stage-3 训练当前配置

### 启动方式
```bat
vit_module\run_stage3.bat   ← 设置 CUDA_VISIBLE_DEVICES=1,2,3 后调用 run_stage3.py
```

### 关键参数（`vit_module/run_stage3.py`）
| 参数 | 值 | 说明 |
|------|-----|------|
| `--model_name_or_path` | `./checkpoints/llava-1.5-7b-deepfake-rand-proj-v1` | 骨架（bridge_v2 + LLaVA） |
| `--data_path` | `./utils/DDVQA_split/c40/train_DDVQA_format.json` | 完整 DD-VQA（含解释） |
| `--image_folder` | `./utils/DDVQA_images/c40/train` | DD-VQA 图片 |
| `--deepfake_ckpt_path` | `./checkpoints/stage_1/bridge_v2_phase1.pth` | 方案B 检测器 |
| `--lora_enable` / `--lora_r` / `--lora_alpha` | True / 128 / 256 | LoRA 配置 |
| `--tune_mm_mlp_adapter` / `--tune_deepfake_mlp_adapter` | True / True | 训练 MLP 投影 |
| `--fp16` | True | 配合 no-op GradScaler |
| `--output_dir` | `./checkpoints/llava-v1.5-7b-deepfake_stage-3-delta` | 输出 |
| `--num_train_epochs` | 1 | 1 epoch |
| `--per_device_train_batch_size` | 1 | 小 batch |
| `--gradient_accumulation_steps` | 16 | 有效 batch=16 |
| `--save_steps` / `--save_total_limit` | 100 / 2 | 每 100 步保存 |
| `--gradient_checkpointing` | True | 省显存 |
| `--dataloader_num_workers` | 0 | Windows spawn 兼容 |

### GPU 使用
- **`CUDA_VISIBLE_DEVICES=0,2,3`**（物理 GPU 0,2,3；GPU 1 有残留进程占用故跳过）
- torch 内部逻辑索引 0,1,2 映射物理 0,2,3
- fp16 7B ≈ 14GB，3 卡分担 ~4.7GB/卡

---

## 5. 训练进度（截至 2026-07-31 18:35）

| 指标 | 值 |
|------|-----|
| 状态 | 🔄 **运行中**（修复后 3 卡 GPU 0,2,3） |
| 进度条 | 6/1734（真实 step；进度条因日志 bug 更新慢，以 DBG-LOSS 为准） |
| 真实 loss | 2.58~3.77 波动，**无 nan**（105 个 micro-step 全正常） |
| 速度 | ~71s/真实 step（≈ 4.4s/micro-step） |
| 预计总时长 | ~34 小时/epoch |
| checkpoint | 未保存（`save_steps=100`，未到 100 步） |

### 训练日志位置（本次修复后训练）
```
C:\Users\Supor2\AppData\Local\Temp\claude\...\bf0bv4t2x.output   ← 当前（checkpoint 修复后重启）
C:\Users\Supor2\AppData\Local\Temp\claude\...\bnco4gboh.output  ← 之前（100 步崩溃）
```
监控: `python vit_module/watch_stage3.py` 或 `grep "DBG-LOSS" <输出文件>`

---

## 6. 已完成的代码修改（为适配本机）

| 文件 | 修改内容 |
|------|---------|
| `vit_module/run_stage3.py` | Stage-3 启动脚本 + no-op GradScaler monkeypatch |
| `vit_module/run_stage3.bat` | 设 `CUDA_VISIBLE_DEVICES=1,2,3` |
| `llava/train/train_deepfake.py` | ①`torch_dtype` 显式 fp16/fp32 ②vision_tower dtype 跟随 compute_dtype ③Phase2 后 deepfake_encoder 统一 fp16 ④`vt.to(cuda:1)` 仅 2 卡时执行 |
| `llava/model/language_model/llava_llama.py` | ①deepfake_encoder 构建用 fp16 ②8 处 image_tensor 硬编码 fp16 → self.dtype ③load_deepfake_encoder 后统一 fp16 |
| `llava/train/llava_trainer.py` | compute_loss 加 `[DBG-LOSS]` 打印（真实 loss 监控） |
| `vit_module/vit_m2f2_detector_unified.py` | 统一检测器（`fusion_mode='cosine'/'bridge'` 双模式） |
| `torchvision/__init__.py`（site-packages） | 移除 `_meta_registrations` import（兼容 bug） |
| `transformers/trainer.py`（site-packages） | `_maybe_log_save_evaluate` 加 `[LOGBUG]` 打印（调试用） |

---

## 7. 关键调试结论（避免重复踩坑）

### 7.1 loss=0.0 之谜（已解决）
- **现象**：trainer 日志 `{'loss': 0.0}`，但 `[DBG-LOSS] lm_loss.item()` 显示真实非零（2.85, 3.50...）
- **根因**：多卡 `dispatch_model` 下 `tr_loss` 聚合异常 + `logging_nan_inf_filter` 将异常替换为 0 显示
- **结论**：**训练本身正常**，日志显示是 bug。判断训练进度用 `[DBG-LOSS]` 打印

### 7.2 fp16 GradScaler 冲突（已解决）
- **现象**：`ValueError: Attempting to unscale FP16 gradients`
- **根因**：冻结 backbone（fp16）+ transformers GradScaler 假设 fp32
- **解决**：no-op GradScaler monkeypatch（scale/step/unscale/update 全 no-op）

### 7.3 8-bit 训练不可行（结论）
- bitsandbytes 8-bit 前向可用，但**反向传播 dtype 不兼容**（Half vs Float）
- 8-bit 仅适合推理，训练用 fp16

### 7.4 显存方案演变
| 方案 | 结果 |
|------|------|
| fp32 3卡 | ❌ OOM（28GB/3 > 11.8GB） |
| fp32 4卡 | ❌ dtype 不一致（vision_tower 硬编码 fp16） |
| fp16 2卡 | ✅ 能跑（5min/步） |
| fp16 3卡 + no-op scaler | ⚠️ 能跑但 16 步后 loss=nan |
| **fp16 3卡 + 真实 GradScaler + 可训练参数 fp32** | ✅ **当前方案**（无 nan，~71s/step） |

### 7.5 checkpoint 保存崩溃（已解决）
- **现象**：第 100 步保存 checkpoint 时 `ModuleNotFoundError: No module named 'deepspeed'`
- **根因**：`llava_trainer.py` 的 `maybe_zero_3` 无条件 `from deepspeed import zero`（ZeRO-3 专用），而本机无 deepspeed
- **解决**：`maybe_zero_3` 加 try/except ImportError，无 deepspeed 时直接 detach 到 CPU

### 7.6 loss=nan 根因（已解决，重要）
- **现象**：第一个真实 step（16 micro-step）后 loss 变 nan
- **根因**：`--fp16 True` 把所有参数（含 LoRA + projector）转 fp16 → 配合 no-op GradScaler（跳过缩放）→ fp16 梯度直接 backward → overflow → nan
- **解决**（`llava/train/train_deepfake.py`）：
  1. LoRA 注入后转 fp32
  2. trainer 创建前所有 `requires_grad=True` 参数统一转 fp32
  3. 移除 no-op GradScaler，用真实 GradScaler
- **原理**：冻结主模型 fp16 + 可训练参数 fp32 = LLaVA 标准做法

---

## 8. 待办 / 下一步

- [ ] **等待 Stage-3 训练完成**（~90h/epoch，1734 步）
- [ ] 每 100 步保存 checkpoint（`save_total_limit=2`）
- [ ] 训练完成后 **合并 LoRA delta**:
  ```bash
  python scripts/merge_lora_weights_deepfake.py \
      --model-base ./checkpoints/llava-v1.5-7b-deepfake-stage-2 \
      --model-path ./checkpoints/llava-v1.5-7b-deepfake_stage-3-delta/checkpoint-xxx \
      --save-model-path ./checkpoints/llava-v1.5-7b-M2F2-Det
  ```
- [ ] 推理验证: `python -m llava.serve.cli_DDVQA_det --model-path <合并后权重>`
- [ ] 修复 trainer 日志显示 bug（可选，低优先级）
- [ ] 清理 `checkpoints/llava-v1.5-7b-deepfake_stage-3-delta/` 中的 0 字节日志文件（多次失败尝试残留）

---

## 9. git 备份记录

```
19ef50b Worklog: Stage-3 3-GPU training running status
79f8f05 Stage-3: 3-GPU dispatch config (GPU 1,2,3), vision_tower move only for 2-GPU
1faa074 Backup: project code + config (exclude dataset nested repo, large weights)
```
- git 仓库在项目根目录（`dataset/` 是独立嵌套仓库已排除）
- 大权重已 gitignore（`*.pth` `*.safetensors` `*.bin` 等）

---

## 10. 恢复检查点指引

若新会话接手，请：
1. 读本文件 §2（环境约束）、§4（配置）、§5（进度）
2. 检查训练是否还活着: `nvidia-smi | grep python`
3. 监控 loss: `grep DBG-LOSS <输出文件>`
4. 训练完成后按 §8 合并权重
