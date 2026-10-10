# 单轴脆弱性与语义增量诊断

研究导航见 [EXPERIMENT_INDEX.md](EXPERIMENT_INDEX.md)，最新 G27 解读见 [G27_REVIEW.md](G27_REVIEW.md)。历史目录和脚本路径保持不变；当前 checkout 的 JSON 汇总不等于完整原始产物。

此目录提供冻结检测器的数据提取和离线统计，不修改训练模型或权重。
当前支持完整的 ViT_M2F2Det_Bridge stage-1 checkpoint；其他架构不能直接复用。
这是一组预先固定的诊断条件，不根据目标域结果搜索提示词、超参数或阈值。

## G30：改进 ViT 上的融合增量

新的四组对照见 [INCREMENTAL_FUSION_EXPERIMENT.md](INCREMENTAL_FUSION_EXPERIMENT.md)。它把 `acc.txt` 对应的独立改进 ViT 与当前融合检测器严格分开，要求相同 ViT 权重、相同评测行、源域视频级选参，并保存逐样本分数。只有在修复 `LayerNorm(1)` 后重新训练并提取的 Bridge 特征，才可进入 `V_BRIDGE`；原 checkpoint 的 Bridge 常数输出不得复用。

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

### G26 条件读取实验：规格与实现

**基线口径更正（2026-10-05）**：G26的`s_V`来自完整检测器logits，不是纯ViT分数。若验证“CLIP局部信息相对于纯ViT的增量”，使用下面G27独立脚本，不复用G26分数作为ViT-only基线。

验证"保留 ViT 主决策、条件读取 CLIP image 局部区域并以残差纠错"的假设。
**规格**：`CONDITIONAL_CLIP_EXPERIMENT.md`（预注册，只描述协议）。
**实现**：`conditional_clip.py`（本仓库补写）+ `check_conditional.py`（合成自检）。
**实测结果**：`FINDINGS_问题验证数据.md` 的 **P14**；完整过程见 `WORKLOG.md` §D.60–§D.61。
**产物**：`vit_module/_g26/conditional_clip_run01/`。

```bash
# 正式运行（输出目录不得预先存在；纯 CPU，不占显卡）
python vit_module/diagnostics/conditional_clip.py \
  --input vit_module/_g25/diag_runs/extract_regions \
  --output vit_module/_g26/conditional_clip_run01 \
  --seeds 20261004 20261005 20261006 \
  --folds 3 --epochs 200 --bootstrap 1000 --repeats 5 --threads 4
```

**使用前必须知道的三件事**：

1. **参数量自动配平**。规格要求 F 的参数量与 B–E 接近，但按字面取值 F 会是 B 的约 25 倍，容量控制（判定链第 2 条）随之失效。实现中 B/C/D 的隐藏宽度由二分搜索解出，保证**其参数量不低于 F**。若改动 `--dim` / `--width`，配平会自动重算。
2. **区域位置控制被有意跳过**。F 的查询只来自 `V`、区域只经 softmax 加权求和进入，**数学上置换不变**；规格 §4.6 明确禁止把区域顺序置换当作有效控制。自检中有对应断言。
3. **`E` 组退化的旧结论已更正**。P14.8 与 WORKLOG D.65 复核发现 E 从不等于 `s_V`，因此 E 是有效基线，`F vs E` 比较有意义。选中强正则或验证 BCE 接近基线，不能证明逐样本修正量为零。run01/run02 选参不同，结果不直接相减。

**首次运行前的自检**：

```bash
python vit_module/diagnostics/check_conditional.py
```

覆盖 `s_V` 方向、关闭控制（`delta≡0` 逐元素精确还原 `s_V`）、donor 不跨视频/不跨 (domain, split)、源域视频同时含两类、F 的置换不变性。

### G27 纯 ViT 增量实验

规格见`PURE_VIT_CLIP_EXPERIMENT.md`。纯ViT基线仅用源域V训练线性头；原检测器分数只作参考。MEAN与ADAPTIVE共享参数结构、初始化和投影/分类头，只改变区域聚合权重；同时提供校准、ViT-only容量和局部/全局线性拼接对照。

```powershell
python vit_module/diagnostics/check_pure_vit_clip.py
python vit_module/diagnostics/pure_vit_clip.py `
  --input vit_module/_g25/diag_runs/extract_regions `
  --output vit_module/_g27/pure_vit_clip_run01 `
  --primary-domains cd2 dfdcp wild `
  --seeds 20261010 20261011 20261012 --epochs 200 --bootstrap 1000 --threads 4
```

主统计是CD2/DFDCP/Wild逐域AUC宏平均及域内视频bootstrap，不是混合帧的pooled AUC。所有种子/控制重复均保存；单模型统计与ensemble分开报告。需要完整clean分片，输出目录不得存在。

**本次运行结果**（2026-10-05，110 分钟，纯 CPU）见 `FINDINGS_问题验证数据.md` **P15** 与 `WORKLOG.md` **§D.67–§D.68**。四条要点：

1. **基线复核通过**：`V_BASE` 逐域 AUC 与 G16 的 `cls_final` 探针锚点**小数点后 4 位全同**（0.8286/0.8633/0.8261/0.8090，mean3 = 0.8328）；不代替运行时代码与其他模块核查。
2. **当前 ADAPTIVE 的增量假设未获支持**：ADAPTIVE vs V_BASE 主宏平均 **−0.0144 [−0.0308,−0.0002]**，三个种子全为负；相对 POOLED 三种子均显著负，相对两种线性拼接多数显著负。规格列的是五组不同问题，不是五条同性质的成败检验。
3. **当前读取协议未获支持，区域本身未被否定**：同一批九区域做**线性**拼接（0.8392）或只读**全局 C**（POOLED，0.8411）都高于 ADAPTIVE（0.8184）；两种线性组与 POOLED 相对 V_BASE 方向为正（+0.006~+0.008）但区间含 0。
4. **限制与新线索**：λ 上边界 1e-1 被选中 7/12；ADAPTIVE 未检出超过固定噪声的优势（0/15 排除 0，不代表等价）。POOLED 在 DFDCP 的 ensemble 增量为 **+0.0153 [+0.0017,+0.0323]**，Wild 正向但区间含 0，仍需独立留出和 POOLED 自身内容控制。DONOR 的正区间次数经 JSON 复核更正为 **3/15**。

本运行使用 `--threads 4`（规格 §7 写的是 1），理由与记录见 `WORKLOG.md` §D.67.3。

### G26 域条件性复验

修改后的 `conditional_clip.py` 还会生成 `F_cross_domain`：不同 domain、相同 split、不同 video 的 CLIP donor，并在每个评估域写入 `correction_diagnostics`（`delta_auc`、rescue/harm、修正幅度等）。新运行必须使用新目录，不覆盖已有 G26 结果：

```powershell
python vit_module/diagnostics/conditional_clip.py `
  --input vit_module/_g25/diag_runs/extract_regions `
  --output vit_module/_g26/conditional_clip_run02_domain `
  --seeds 20261007 20261008 20261009 --folds 3 `
  --epochs 200 --bootstrap 1000 --repeats 5 --threads 4
```

交叉验证选参阶段现在只使用 fit fold 的标准化统计；旧的 `conditional_clip_run01` 是修复前产物，不能与新运行的数值直接混合。

**本次运行结果**（2026-10-04，69 分钟，与登记预期全部命中）见 `FINDINGS_问题验证数据.md` **§14.8** 与 `WORKLOG.md` **§D.64–§D.65`**。三条要点：

1. **主结论方向不变**：四域合并的 F vs A 为 **+0.0092 [−0.0082,+0.0288]，区间跨 0**；正增益仍只在 cd1/cd2（同一数据集）。
2. **判定的补充条件第 2、4 条成立**（E **不是**退化基线；跨域 donor 未比同域 donor 更大地破坏增益，合并 −0.0073 [−0.0242,+0.0101] 跨 0）。
3. **cd1↔cd2 的"跨域" donor 有 15–17% 是"同数据集换视频"**（cd1 ⊂ cd2，路径推导的文件夹身份已核）。解读该控制的 cd1/cd2 列时必须带此前提。

判定链的更正表、逐域 `correction_diagnostics`、donor 组成统计均在 §14.8。

### G28 内容增量与局部读取失效验证（**已于 2026-10-05 真实运行，见 §D.72 / FINDINGS P16**）

规格见 [CLIP_READOUT_EXPERIMENT.md](CLIP_READOUT_EXPERIMENT.md)，实现为 `clip_readout.py`，合成自检为 `check_clip_readout.py`。POOLED 和 ADAPTIVE 各有 donor/固定噪声/噪声重训控制；所有组按统一超参条件比较，同时保留仅由源域选定的结果。冻结模型的修正缩放、平滑/均匀区域权重是诊断干预，不按目标域选择部署配置。

本轮使用嵌套视频 OOF 基线分数训练修正头，缓解 G27 的训练内/样本外基线分数失配；不同协议的非线性成绩不直接相减归因。输出包括全网格逐样本分数、源域训练/验证轨迹、控制身份、权重、参数和运行时代码快照。

```powershell
python vit_module/diagnostics/check_clip_readout.py
python vit_module/diagnostics/clip_readout.py `
  --input vit_module/_g25/diag_runs/extract_regions `
  --output vit_module/_g28/clip_readout_run01 `
  --seeds 20261020 20261021 20261022 `
  --primary-domains cd2 dfdcp wild --folds 3 --inner-folds 3 `
  --epochs 200 --lambdas 0.01 0.1 1 --weight-decays 0.001 `
  --gains 0 0.25 0.5 1 --temperatures 2 4 `
  --bootstrap 1000 --repeats 5 --threads 1
```

必须到有完整 `clean_*.npz` 的实验机运行；当前 checkout 只有提取配置和完成标记。输出目录不得存在。默认是探索性复验；新留出必须提供预先固定、整视频选择的 `--evaluation-manifest` 并声明 `--evaluation-status new-holdout`，该声明不能替代源域选模 checkpoint 或独立视频审计。

**本次运行结果（`vit_module/_g28/clip_readout_run01`，159 分钟，纯 CPU，`--threads 4` 偏离规格 §7 的 `--threads 1`，已登记）：**

1. 预注册两项主问题**都未达门槛**：POOLED vs V_BASE 宏平均 +0.0079（三 seed 区间全含 0）；ADAPTIVE vs MEAN 无差异。
2. 全局 CLIP 有**小而真实**的增量：vs 容量匹配 V_ONLY 三 seed 全显著为正；对 DONOR/NOISE_FIXED/NOISE_RETRAIN **各 15/15 区间排除零**；λ=1 行与 alpha=0.25/0.5 均显著为正。但与 `[V,C]` 线性读法无差异 → 非残差头特有。
3. 局部九区域**打不过噪声重训**（ADAPTIVE vs NOISE_RETRAIN **0/15**，均值 +0.00004），而 POOLED 为 15/15 → 本轮最锋利的路径判别。
4. 跨域损伤**由修正幅度驱动**：λ 从 0.01→1 各变体单调回归 V_BASE，λ=1 时 ADAPTIVE 与 MEAN 不可分；ADAPTIVE 缩到 alpha=0.25 即消除损伤；温度/均匀干预只在最差 seed 上有效。**注意力尖锐不是原因**。
5. **λ 上边界 0/12 命中**（G27 为 7/12）→ G27 的角点解是网格截断假象，该限制关闭。
6. G27 的 DFDCP 正线索**未复现**；ffpp 域内四个变体全为正（修正同域有用、跨域有害）。
7. 不得按目标域 AUC 选 λ/alpha 部署；本轮默认**探索性**，非正式确认。运行前须按**路径身份**审计分组（裸 `video_id` 存在跨数据集同名巧合；`cd1 ⊂ cd2` 只由视频名全含体现，路径判据看不出）。

### G29 标量向量组合验证（待真实运行）

规格：[VECTOR_FUSION_EXPERIMENT.md](VECTOR_FUSION_EXPERIMENT.md)。独立脚本 `vector_fusion.py` 冻结 V/C 特征，仅训练 CLIP 的 1024→768 投影、标量 alpha 和分类头。V_MATCHED 与融合组使用同一 V 归一化及同形状/初始化分类头；FIXED(alpha=0.5) 与 LEARNED 仅相差一个门控参数。另有 normalized CONCAT、两组自身的 donor/噪声重训/固定噪声控制及相同 decay 的门控消融，避免把选参差异解释成 alpha 学习收益。alpha 不作为分支贡献比例。

```powershell
python vit_module/diagnostics/check_vector_fusion.py
python vit_module/diagnostics/vector_fusion.py `
  --input vit_module/_g25/diag_runs/extract_regions `
  --output vit_module/_g29/vector_fusion_run01 `
  --primary-domains cd2 dfdcp wild `
  --seeds 20261030 20261031 20261032 --folds 3 --epochs 200 `
  --weight-decays 0.0001 0.001 0.01 --bootstrap 1000 --repeats 5 --threads 1
```

新分类头不是残差修正，不保证原 ViT 判别保持不变。目标域不选参/阈值；历史探针只作参考，主要比较使用同轮 V_MATCHED。需要完整 clean_*.npz（仅 V/C+身份即可，无需区域），当前本地只有配置不能执行真实验证。全部逐样本分数、模型/标准化、alpha轨迹、内容控制、向量诊断及代码/输入/输出哈希均保存；不要覆盖旧 G27/G28 目录。
