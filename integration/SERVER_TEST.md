# ViT-only 接入：服务器运行步骤

本轮接入完整原生 ViT 二分类检测器，不训练融合分支。`vit_native` 不构建检测器内部的 CLIP、文本编码器、Bridge 或替代分类头。完整 LLaVA 可继续使用视觉 CLIP 给语言模型提供 token；该塔不参与 ViT 的真假分数。

## 同步范围

同步整个 `integration/` 源码目录（排除 `runs/`、`outputs/`、缓存）和修改后的 `llava/model/language_model/llava_llama.py`。服务器复用已有的 `vit_module/vit_adaptive_mattn_aps.py`。不需要同步旧归档或三个旧模型转发文件。本次没有修改、上传或运行服务器文件。

也可使用本地 `integration/outputs/vit_native_server_bundle_20261017.zip`：先解压到新暂存目录，核对 `BUNDLE_MANIFEST.json` 与文件差异，再备份服务器原 `llava_llama.py`，将 `payload/` 的文件复制到项目同名位置。服务器文件若有额外改动，先合并本轮显式分支。`reference/` 仅供核对原 ViT 基础实现，包内不含权重、图片和实验结果。

## 1. 原项目环境

```powershell
Set-Location -LiteralPath 'E:\Cross-domain_authentication_verification\Next_work\M2F2_Det-main-hyy'
$vitPython = 'C:\Users\Supor2\.conda\envs\M2F2_Det\python.exe'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:PYTHONIOENCODING = 'utf-8'
& $vitPython -c 'import sys,torch; print(sys.executable); print(torch.__version__)'
```

后续均使用 `M2F2_Det`。不需要另行安装训练依赖。

## 2. 不需要真实权重的接口自检

```powershell
& $vitPython integration/tests/test_vit_native.py -v
```

预期 8 项通过：原生头和输出保留；缺头/融合/错误哈希 checkpoint 拒绝；独立预处理与类别排列；冻结推理模式；真实 LLaVA 的 token 投影与语言前向；视觉 token 不影响检测；HF 保存/重载保留头与验证状态。合成测试不作为检测性能证据。

## 3. 使用真实 ViT 权重检查接入

核对 `configs/vit_native.json`：`native_checkpoint` 为原 PDI `net_050.pth`，SHA256 为此前服务器核对值；默认两张图是此前验证存在的真实/伪造图片。可以用 `--checkpoint`、`--images` 改路径，但更换权重时也必须正确更新配置哈希。缺图、缺头、错权重会报错，不会跳过或以随机头继续运行。

CPU 单线程版本，不占显卡：

```powershell
& $vitPython integration/evaluation/smoke_vit_native.py `
  --device cpu `
  --output integration/runs/vit_native_cpu_run01
```

或用 `nvidia-smi` 确认物理卡 1 空闲后，在 Python 启动前指定该卡：

```powershell
$env:CUDA_VISIBLE_DEVICES = '1'
& $vitPython integration/evaluation/smoke_vit_native.py `
  --device cuda:0 `
  --output integration/runs/vit_native_gpu_run01
```

程序加载完整 ViT-B/16（164 个 state key，原生头 `[2,768]`），比较裸 ViT、包装检测器和 LLaVA 内检测器的输出；再检查检测 token 和小型随机语言模型前向。后者无需 CLIP/7B 权重，仅检查真实接入路径，不生成有意义的解释，不拟合模型。

| 输出字段 | 预期 |
|---|---|
| `checkpoint.head_loaded` / `loaded_keys` | `true` / 当前原生权重应为 164 |
| `bare_vit_vs_detector_max_abs` | 同批同像素 FP32 为 0 |
| `bare_vit_vs_llava_detector_max_abs` | 同批同像素 FP32 为 0 |
| `alone_vs_batch_max_abs` | 允许预设浮点误差；脚本会断言 |
| `detector_has_clip_or_bridge` | `false` |
| `token_shape` | 两张图为 `[2,1,64]` |
| `language_forward_shape` | 两张图为 `[2,3,64]` |

成功目录含 `summary.json`、`input_config.json` 与最后写入的 `COMPLETE`。重跑改为新的目录名，不覆盖旧结果。

## 4. 可选：真实 7B LLaVA 整体运行

先通过前三步。使用原始本地 LLaVA 基座与本地 CLIP，**不传旧 Bridge 合并模型或 LoRA**；新检测投影尚未按 ViT 对齐，首次文字只用于运行检查。

例如确认物理卡 1、2 空闲并据实选择显存预算后：

```powershell
$env:CUDA_VISIBLE_DEVICES = '1,2'
& $vitPython integration/evaluation/run_llava_vit.py `
  --llava-model checkpoints/llava-1.5-7b `
  --clip checkpoints/clip-vit-large-patch14-336 `
  --image 'F:/zhj/data/FaceForensic++_raw/original_sequences/c23/faces23/102/127.png' `
  --max-memory-gib 8 `
  --max-new-tokens 96 `
  --output integration/runs/llava_vit_runtime_run01
```

程序仅使用 shell 指定的可见卡。先加载语言模型，再严格载入原生 ViT 并保持 FP32；新检测投影是否被初始化会明确记录。生成期间捕获真实检测 logits，与直接 ViT 前向核对。保存输出和加载信息，不保存或修改权重。原生检测器若被磁盘/meta offload，当前加载器会拒绝加载；请调整预算使检测器实际驻留内存。

**完整 7B/GPU 路径尚未在本地实际运行，需要本轮服务器测试确认。**暂不使用旧训练、合并脚本开始本轮训练。

## 预处理、标签与验收范围

当前契约：原始 PIL RGB → `[0,1]` → 224 双线性缩放（无 antialias）→ mean/std=0.5。该管线与 CLIP 的裁剪和归一化独立。原生 logits 不改顺序；送入语言投影前明确排列为 `[fake,real]`。

配置的 `fake_logit_index=0` 和输入管线尚待与 PDI 原训练/验证脚本核对，因此两个 `verified_against_pdi` 字段保持 `false`。本轮能证明相同像素、权重和头下接入前后的数值一致性，**不能冒充 `acc.txt` 或原 PDI 图像管线的正式复现**。性能评测前确认原始配置，不按目标集 AUC 反转类别或挑预处理。

本地已通过 8 项合成测试，并用完整 91.4M ViT 拓扑、合成 checkpoint/图片跑通步骤 3 的端到端 CLI；三个输出一致性差值均为 0。该完整拓扑测试标记 `input_is_test_fixture=true`，仍不是服务器真实权重结果。

返回结果时先提供 `summary.json` 或完整报错。先验收加载和运行，再决定投影对齐；本轮不引入 CLIP/ViT 检测融合。
