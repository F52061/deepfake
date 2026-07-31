# M2F2-Det Stage-3 调试工作日志

> 日期: 2026-07-31
> 环境: Windows 10, M2F2_Det conda env, 4× GTX 1080 Ti (11.8GB)

---

## 当前状态

### 训练运行情况
- **能跑** ✅ Stage-3 LoRA 微调在 GPU 2,3 双卡 fp16 上正常运行
- **已验证**：`compute_loss` 返回真实非零 loss（DBG-LOSS: 2.85 → 3.50），梯度正常
- **速度**：~5 分钟/步 × 1734 步 = ~145 小时/epoch（硬件限制）
- **日志 bug**：trainer 日志显示 `{'loss': 0.0}`，但实际 loss 非零（多卡 dispatch 下 tr_loss 聚合异常）

### 关键配置文件
- `vit_module/run_stage3.py` — Stage-3 启动脚本
  - fp16 + no-op GradScaler monkeypatch
  - LoRA r=128, alpha=256
  - tune_mm_mlp_adapter + tune_deepfake_mlp_adapter
  - gradient_accumulation_steps=16, batch_size=1
- `vit_module/run_stage3.bat` — 设置 CUDA_VISIBLE_DEVICES 并启动

### 已修改的关键代码（为适配本机）
| 文件 | 修改 |
|------|------|
| `llava/train/train_deepfake.py` | torch_dtype 跟随 fp16/fp32；vision_tower dtype 跟随 compute_dtype；Phase2 deepfake_encoder 统一 fp16 |
| `llava/model/language_model/llava_llama.py` | deepfake_encoder dtype fp16；image_tensor 用 self.dtype 替代硬编码 fp16 |
| `llava/train/llava_trainer.py` | compute_loss 加 DBG-LOSS 打印 |
| `transformers/trainer.py` | _maybe_log_save_evaluate 加 LOGBUG 打印（调试用） |
| `vit_module/vit_m2f2_detector_unified.py` | 统一检测器（cosine/bridge 双模式） |

---

## 关键问题与解决方案

### 问题 1: CLIP 离线加载失败
- **现象**: huggingface.co 连不上，CLIP 加载失败
- **解决**: 重定向所有 `from_pretrained` 到 `checkpoints/clip-vit-large-patch14-336/` 本地路径

### 问题 2: PyTorch 是 CPU only
- **现象**: `torch 2.13.0+cpu`，无 CUDA
- **解决**: 重装 `torch==2.2.2+cu118 torchvision==0.17.2+cu118`

### 问题 3: torchvision 0.17 + torch 2.2 兼容性
- **现象**: `_meta_registrations` 崩溃 "operator torchvision::nms does not exist"
- **解决**: 移除 torchvision `__init__.py` 中的 `_meta_registrations` import

### 问题 4: Windows 下 os.environ 设置 CUDA_VISIBLE_DEVICES 无效
- **现象**: Python 内 `os.environ["CUDA_VISIBLE_DEVICES"]="2"` 不生效，仍看到 4 卡
- **解决**: 用 .bat 文件在进程启动时设置

### 问题 5: 7B 模型显存不足
- **现象**: fp32 28GB / fp16 14GB > 单卡 11.8GB
- **解决**: 双卡 `dispatch_model`（fp16），或 4 卡 fp32

### 问题 6: fp16 GradScaler 与冻结 backbone 冲突
- **现象**: "Attempting to unscale FP16 gradients" / NaN 梯度
- **解决**: no-op GradScaler monkeypatch（scale/step/unscale/update 全 no-op）

### 问题 7: trainer 日志显示 loss=0.0
- **现象**: DBG-LOSS 显示真实 loss 2.85/3.50，但 trainer 日志 `{'loss': 0.0}`
- **根因**: 多卡 dispatch_model 下 `tr_loss` 聚合异常 + `logging_nan_inf_filter` 将 NaN 替换为 0 显示
- **验证中**: 加 DBG-LOSS/LOGBUG 打印定位

---

## 下一步计划

### 目标: 3 卡微调（GPU 1,2,3，留出 GPU 0）
1. 更新 `run_stage3.bat`: `CUDA_VISIBLE_DEVICES=1,2,3`
2. 确认 dispatch_model 在 3 卡上工作
3. 若 3 卡有问题，回退到已验证的 2 卡配置
4. 修复 trainer 日志显示 bug（可选）

### 需要用户注意
- GPU 0 留空（避免影响其他工作）
- 训练速度 ~5 分钟/步，1 epoch ~6 天（硬件限制，非配置问题）
