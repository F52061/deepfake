# 单轴脆弱性与语义增量诊断

此目录提供冻结检测器的数据提取和离线统计，不修改训练模型或权重。
当前支持完整的 ViT_M2F2Det_Bridge stage-1 checkpoint；其他架构不能直接复用。
这是一组预先固定的诊断条件，不根据目标域结果搜索提示词、超参数或阈值。

## 输入与标签

准备 CSV，列名为：

```csv
path,y,domain,split,video_id
C:/data/ffpp/real/video001/frame001.jpg,0,ffpp,train,real_video001
C:/data/ffpp/fake/video002/frame001.jpg,1,ffpp,train,fake_video002
C:/data/ffpp/real/video003/frame001.jpg,0,ffpp,test,real_video003
C:/data/cd2/fake/video004/frame001.jpg,1,cd2,test,fake_video004
```

- y=1 是伪造，y=0 是真实。既有报告/npz 中 y=1 经常表示真实，转换时必须显式翻转。
- 路径必须是本机存在的绝对路径。video_id 必须表示真实的视频分组，不能使用帧号。
- 每个 domain/split 至少两个视频。只有源域允许 train；val 保留但不参与拟合或报告。
- 同一 domain/video_id 不得跨 split。FF++ 同一原始视频的真实与派生伪造应共享分组标识，避免原始视频泄漏。
- 主轴、标准化、线性读取器仅使用源域 train。test 标签仅计算最终诊断统计。
- CSV 示例仅说明格式，完整数据必须满足每个训练/评估域的类别与视频要求。

## 运行

在项目根目录、具有项目训练依赖的 Python 环境中运行。CLIP 参数指向包含原生视觉/文本 projection 的本地完整 CLIPModel 目录。

```powershell
python vit_module/diagnostics/extract.py --manifest manifest.csv --checkpoint checkpoints/stage_1/bridge_v2_phase1.pth --clip checkpoints/clip-vit-large-patch14-336 --device cuda:0 --batch-size 4 --output diagnostics_runs/extract_01
python vit_module/diagnostics/analyze.py --input diagnostics_runs/extract_01 --source ffpp --repeats 5 --bootstrap 1000 --output diagnostics_runs/analysis_01
```

选择实际可用设备；CUDA_VISIBLE_DEVICES 由运行者设置。提取默认 CPU，可通过 --variants clean 做最小冒烟；正式诊断保留默认所有图像扰动。--save-tokens 另外保存最后层 ViT/CLIP token 和 ViT 中间层 token，磁盘开销明显增加。
--fake-logit 默认 1，必须与权重的类别顺序核对，不通过目标域标签自动猜测。
输出目录必须不存在，以免覆盖历史结果。提取中断时已完成的分片会保留，没有 COMPLETE 标记的目录禁止分析；当前版本不自动续跑。

## 实验一：单轴受控干预

源域训练 CLS 的 center-only PCA 得到 PC0，干预半径采用源域轴坐标标准差的 0.25/0.5/1/2 倍。每个半径、种子、样本使用等 L2 能量的五类干预：

1. PC0 方向随机正负干预。
2. 随机正交方向，逐样本生成。
3. 全空间随机方向，逐样本生成。
4. 同一随机正交方向，跨样本仅随机正负。
5. 源域 PC1 方向，跨样本仅随机正负。

固定随机方向与 PC1 控制用于避免仅将一维干预与高维噪声比较。
干预后的 CLS 重放 deepfake_proj、LayerNorm、最终分类头，其余已保存的 CLIP/bridge 输入固定。
开始分析前必须通过原始 logits 重放检查。另保存 V-only 读取器的对应结果，以区分完整检测器读出与线性探针。

JPEG quality=70、Gaussian blur(5x5,sigma=1)、336→168→336 缩放和 BGR 通道缩放(0.95,1,1.05) 是固定图像干预。
这些条件重新执行整个检测器。其 ViT CLS 位移又分解为轴向与正交分量，分别重放分类头。
真实图像干预与 CLS 重放是不同干预，不把分解结果当成端到端因果归因。
JPEG/模糊可能抹除真正的伪造证据；即使标签保持，性能下降也不能自动解释成捷径。
轴向更敏感证明方向依赖，不能单凭该结果证明跨域差完全由 PC0 引起。

## 实验二：语义增量

S 是原生 CLIP pooled image embedding 与 8 个固定文本提示词的 cosine。
提示词覆盖光照、眼齿、纹理和边界一致性，逐词内容固定保存在 config.json。
这只是一个可复现的语义读取候选，不等同于全部 CLIP 语义，也不能证明局部几何一致性已经被测量。
patch_S 保存最后层 patch 与相同文本的对齐图，供后续空间诊断；原生 projection 在 CLS 上训练，patch 对齐属于探索性观测，当前不进入主要读取器。
提取会验证 checkpoint 的视觉塔与原生 CLIP 完全相同，防止错用原生投影。

源域上拟合统一 StandardScaler + LogisticRegression(C=1e-3)：
V、S、C、V+S、V+C。C 是原始 CLIP CLS，是视觉表示对照，不直接称为语义。
对 V+S 添加每种子两个置换控制和同维度随机特征控制：

- 固定正常读取器，仅将语义换为同 domain/split 的其他视频语义。
- 使用置换后的源域语义重新训练读取器，并在置换后的测试语义上评估。
- 使用源域语义均值/标准差匹配的独立 Gaussian 特征训练/评估。

置换不使用标签，不跨 train/test，不从同视频取 donor。它没有匹配姿态/质量，可能受这些因素混淆，结果应联合解释。
保存 donor 行号及随机特征，便于追加匹配置换等控制。
轴向干预时保留 S，并比较 V+S 与受相同干预的 V-only；图像干预下使用重新提取的 S。

## 数据与统计

提取目录：

- config.json：提示词、参数、库版本、manifest/checkpoint SHA256。
- head.npz：重放需要的投影、归一化与最终 head 参数。
- semantic_parameters.npz：固定提示词、文本嵌入和原生视觉投影。
- 每 variant/batch 一个 npz：行标识、路径、标签、domain/split/video_id、V/C/S/F/logits/patch_S。
- 可选完整 token。COMPLETE 仅在所有分片成功写入后生成。
- clean 分片附图像 SHA256；首批会检查同一图像单样本与批量预测是否一致，若失败则要求 batch-size=1。

分析目录：

- source_axis.npz：源域均值、PCA 基/奇异值、PC0、轴标准差及训练行号。
- reader_*.npz：所有读取器的标准化、系数、截距、类别顺序。
- directions_rep*.npz：实际干预方向；config.json 保存各干预 L2 范围。
- image_*_decomposition.npz：逐样本轴位移、正交/整体 L2、语义位移。
- sample_scores.npz / samples.csv：所有条件的逐样本分数，可直接重算任意统计。
- summary.json：逐域 AUC、相对基线 AUC 差、视频分组 paired bootstrap 95% CI、预测翻转、救回/误伤数。

失败子集默认来自源域训练 V-only 读取器的固定零阈值；额外报告完整 detector 失败子集上的 V+S。
只在原始基线错误集合上分析 rescue，harmed 统计原始正确样本被改错的数量。
失败子集 AUC 存在选择偏差且可能单类，单类返回 null；以总体净收益、rescue/harm 和配对区间共同判断。
只有一个视频的域无法给出可靠的视频 bootstrap 区间，也不应作为跨域平均的主要证据。
当前不自动合并不同域 AUC，以免混合域身份和不同采样比例。

诊断支持预先声明的候选机制；不能由某个条件改善直接宣称架构创新成立。
若 V+S 没有超过置换/随机对照，记录负结果，不通过目标域调提示词直到改善。

## 本地验证

```powershell
python vit_module/diagnostics/self_check.py
```

使用临时合成数据执行两次离线分析，验证分类头重放、等能量、正交方向、置换隔离、视频泄漏拒绝，以及改变目标域数据不改变源域 PCA/读取器。合成结果不作为研究证据；验证目录结束后自动清理。
真实实验仍需 manifest、完整 checkpoint、原生 CLIP 目录及项目推理依赖。

---

## 本项目（M2F2_Det-main-hyy）专用说明

> 以下为本仓库实测确认的口径与坑位，**与本文档其他部分的通用约定不同**，使用时以此为准。
> 实测记录见 `WORKLOG.md` §D.51–§D.57 与 `FINDINGS_问题验证数据.md` 第四部分。

### 必须显式传 `--fake-logit 0`

本仓库 checkpoint 的 **logit 下标 0 = fake**，而工具默认 `--fake-logit 1`。

| 分数取法 | ffpp 域内 AUC |
|---|---|
| `logits[:,1] − logits[:,0]`（工具默认） | 0.0207 |
| `logits[:,0] − logits[:,1]`（**本仓库正确**） | 0.9793 |

工具自带的 `replay_head` 检查**查不出**这个错误——它在重放与对照两边用同一个下标，放反了照样自洽通过。**放反不会报错，只会让全部结果方向相反。**

### 标签与 split 需先转换

本仓库 `_g16/layer_feats.npz` 的约定与工具不同，**不能直接喂入**：

| 项 | 本仓库 | 工具要求 | 转换 |
|---|---|---|---|
| 标签 | `y == 1` 表示 **real** | `y == 1` 表示 **fake** | `y = 1 − y` |
| `split` 列 | 目标域直接存域名字符串（`cd1`/`cd2`/`dfdcp`/`wild`） | 只接受 `train`/`val`/`test` | 目标域改为 `test` |
| `ffiw` | 300 张全部来自**同 1 个视频** | 语义置换要求 ≥2 视频 | **剔除**，不入任何聚合 |

现成适配器：`vit_module/_g23/make_manifest.py`，输出 `vit_module/_g23/manifest_full.csv`（4200 行）。

### 本仓库实测可用的命令

```bash
# 提取（注意 --fake-logit 0；输出目录不得预先存在）
python vit_module/diagnostics/extract.py \
  --manifest vit_module/_g23/manifest_full.csv \
  --checkpoint checkpoints/stage_1/bridge_v2_phase1.pth \
  --clip checkpoints/clip-vit-large-patch14-336 \
  --device cuda:1 --batch-size 4 --fake-logit 0 --variants clean \
  --output vit_module/_g23/diag_runs/extract_full

# 局部/结构实验需再加 --save-regions
python vit_module/diagnostics/local_analyze.py \
  --input vit_module/_g25/diag_runs/extract_regions \
  --output vit_module/_g25/local_run01 --bootstrap 1000
```

**分辨率**：4200 张 1 个变体约 13.5 分钟；5 个变体 29 分 12 秒。`--save-regions` 后 1 个变体约 525 MB。

### 运行时

- 分析侧**瓶颈是 `summarize` 的配对自助法**，不是读取器拟合。全量规模（2200 × 16128 维）单次 LogisticRegression 拟合仅 **1.31 秒**（10 次迭代收敛，远未触及 `max_iter=3000`）；但 `--bootstrap 1000` 下约 130 次调用需 **9–10 分钟**。
- 工具默认 `--bootstrap 1000`；本项目 §D.54 用的是 300。**bootstrap 只影响区间宽度，全部点估计与之无关。**

### 已知需修复处（本仓库已打补丁）

原始提交的 `extract.py` / `local_analyze.py` / `residual_clip.py` 存在若干**阻断性**问题，本仓库已修复，逐条记录在 `FINDINGS_问题验证数据.md` 的「附 · 诊断工具链自身的缺陷」。**未打补丁的原始版本无法产出任何结果。**

### 下一轮实验规格

在现有残差/局部实验之后，下一步验证“保留 ViT 主决策、条件读取 CLIP image 局部区域并以残差纠错”的假设。实验设计、对照、判定和数据保存要求见 `CONDITIONAL_CLIP_EXPERIMENT.md`；该文件只描述协议，尚未包含 G26 实测结果。
