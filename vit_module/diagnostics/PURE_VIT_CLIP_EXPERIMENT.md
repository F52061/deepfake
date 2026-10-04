# G27：相对于纯 ViT 的 CLIP 局部判别增量

## 1. 验证目的与此前口径更正

G26 使用的 `logits[:,0]-logits[:,1]` 来自整个原检测器，其中已经有 CLIP image CLS；它不是纯 ViT 分数。G26 的结果回答“是否能改善原融合检测器”，不能回答“CLIP 局部信息比纯 ViT 多贡献多少”。不修改既有产物；本轮独立使用 `pure_vit_clip.py`。

本轮依次验证：

1. 在同一冻结 ViT CLS 上，加入 CLIP 局部读取能否超过纯 ViT 读取器？
2. 收益是否只是分数校准或增加非线性参数？
3. 自适应区域权重是否超过同结构的均匀权重？
4. 改善是否依赖当前图像的 CLIP 内容，并在独立数据集及训练种子上稳定？

当前“门控”实现实际是注意力加权区域读取，不包含显式置信度开关。不能预先声称它识别了 ViT 的失败样本。

## 2. 输入与纯 ViT 基线

复用 `_g25/diag_runs/extract_regions` 的完整 clean 分片，不必重跑编码器。需要 V/C/clip_regions/logits 及样本身份字段，标签必须是 `1=fake, 0=real`。

`V_BASE = StandardScaler + LogisticRegression(C=0.001)`：仅在 FF++ train 的 V（768 维）上拟合。CLIP、logits、目标标签均不进入这个头。该基线是“冻结 ViT 表征 + 源域真假线性探针”，不是 ViT 原生分类头，也不是从原融合头切出一段权重。

原模型分数另存为 `DETECTOR_REFERENCE`，只用于背景比较，不进入残差模型或选参。旧实验的 V≈0.8318 可作复核参考，但不是强制要求新数据得到相同数值。

冻结 checkpoint 曾存在 FFIW 选模问题（见 FINDINGS P7）。本轮只保证新增读取器的拟合和选参不使用目标域，不把当前 checkpoint 描述成完全干净的零目标域选模基线。正式确认需使用干净选模的 checkpoint。

## 3. 实验组：每组只回答一个问题

| 输出组 | 定义 | 主要用途 |
|---|---|---|
| V_BASE | 固定纯 V 线性头 | 核心参照 |
| CALIBRATED | a × V_BASE + b，a>0 | 温度/偏置校准对照；不改变 AUC 排序 |
| V_ONLY | V 与 V_BASE 输入容量匹配 MLP，输出修正量 | 排除新增容量、非线性和重新读取 V |
| V_C_LINEAR | `[V,C]` 源域线性探针 | 原全局拼接基线 |
| V_REGIONS_LINEAR | `[V,C_regions]` 源域线性探针 | 保留 ViT CLS 的局部拼接基线 |
| POOLED | 同结构读取 CLIP pooled CLS，输出修正量 | 判断局部区域是否必要 |
| MEAN | 同结构读取 CLIP 九区域，权重固定 1/9 | 排除区域投影/非线性聚合本身 |
| ADAPTIVE | 同结构读取 CLIP 九区域，权重由 V 查询产生 | 检验自适应区域选择 |

所有非线性组都是 `score = V_BASE + delta`，基线头不更新。V_ONLY 的可训练参数量不低于 ADAPTIVE；MEAN、POOLED、ADAPTIVE 的参数形状和数量完全相同，同种子初始化也完全相同。

均匀和自适应组共享：V 的 LayerNorm、查询投影、每个区域的 LayerNorm/投影，以及 `[V_norm, query, pooled_region, V_BASE]` 修正头。两组的唯一结构差异：

```text
MEAN:     weight_i = 1 / region_count
ADAPTIVE: weight_i = softmax(query(V) · projected(C_i) / sqrt(dim))
```

查询在 MEAN 中也进入修正头，避免用未参与计算的参数虚假配平。修正头最后一层零初始化，训练起点精确恢复 V_BASE。无位置编码，所以区域顺序置换不是有效的内容破坏控制。

## 4. 源域训练、选参与校准

仅 FF++ train 按 video_id 做 GroupKFold（默认三折）。关联真实/派生伪造视频必须共组；已有 video_id 是否真实表示这种关联仍需上游审计，代码不能凭字符串自动证明。

每个选参折：

1. 仅 fit 视频拟合 V_BASE 线性头和 V/C/区域标准化；
2. 仅 fit 视频训练修正头；
3. validation 视频使用该折的 V_BASE 分数和标准化；
4. 比较 validation BCE，选择 lambda/weight_decay。

最终在全部 source train 重拟合纯 V 头、标准化和修正模块。残差训练看到的是 fit 视频内拟合的基线分数，而验证/目标是样本外分数；这是仍需保留的协议限制，不解释为纯独立信息分解。

默认固定网格：lambda_delta=[0,0.001,0.01,0.1]，weight_decay=[0.0001,0.001]，200 epochs，单线程 CPU，学习率 0.001，投影/隐藏宽度各128。损失为 `BCE(V_BASE+delta,y)+lambda_delta*mean(delta²)`。每组同训练预算；每个候选/折在构造模型前固定种子，不受遍历顺序影响。不根据目标表现调整网格。

校准使用全部源域 out-of-fold V_BASE 分数和源标签拟合正仿射 a/b。a>0，所以其 AUC 必须和 V_BASE 相同；它只用于说明阈值纠错是否也能由校准实现。CALIBRATED 不能被称为新判别信息。

## 5. CLIP 输入控制

对每个 ADAPTIVE 种子执行五次控制，全部结果都报告，不能只报告第0次：

- DONOR：同 domain/split、不同 video 的整组 CLIP 区域。保持当前 V/V_BASE 不变；没有有效 donor 的行不进入 donor 比较。
- NOISE_FIXED：源训练标准化区域的逐维均值/方差匹配高斯噪声，读取器固定。
- NOISE_RETRAIN：同结构、同初始化、同训练预算，在噪声上重训修正头。使用真实 ADAPTIVE 的源域选定超参数，因此它是固定协议容量控制，不是穷尽噪声模型的最佳能力。

不做跨数据集 donor 主实验：它同时破坏样本和类别对应、质量及数据集分布，不能可靠隔离“域兼容性”。同域 donor 和噪声控制下降，只说明输入依赖，不单独证明条件互补或机制性能优势。

噪声不再把五份大数组逐个落盘，而保存 noise_mean/noise_std/生成种子/公式。donor 行号、可用掩码逐次保留；scores 保存所有控制的分数。精确复现应保持 NumPy 版本、float32 转换与数组形状不变。

## 6. 统计口径与判定逻辑

### 6.1 主指标

每域 AUC及配对视频 bootstrap（1000次）；主宏平均默认只使用 CD2、DFDCP、Wild，CD1单列，FFIW排除。每次 bootstrap 在每个域内抽视频，再计算各域 AUC 的算术平均。不能把混合1200行的 pooled AUC区间当成宏平均区间。

每个训练种子的比较独立报告；另列各种子 AUC 均值/标准差，以及分数平均后的 ensemble AUC。ensemble 是不同部署模型，不能称为“平均单模型能力”。这些区间条件于当前已训练模型，不覆盖重新采样训练集的不确定性。

预先固定的比较为：

1. ADAPTIVE vs V_BASE：纯 ViT 之上的实际增量。
2. ADAPTIVE vs V_ONLY / CALIBRATED：排除容量和校准解释。
3. ADAPTIVE vs MEAN：隔离自适应权重本身。
4. ADAPTIVE vs POOLED / V_C_LINEAR / V_REGIONS_LINEAR：是否真的需要局部自适应结构。
5. ADAPTIVE vs 每一次 DONOR/NOISE：是否使用当前 CLIP 内容。

第1和第3是两项主要问题，其他是机制/容量对照。所有多比较区间作为探索性描述，不从中挑一个正区间就宣布确认成功；已有目标样本反复被查看，正式确认需要新的预先固定视频留出。

### 6.2 阈值指标

所有分数统一用零阈值；源域校准的偏置单独保留。以 V_BASE 原错误集固定定义 rescue，以原正确集定义 harm，同时报告总体 AUC和准确率。如果只多救回但不改善 AUC，且 CALIBRATED 同样能救回，优先解释为校准/阈值现象。

### 6.3 如何解释结果

| 观察 | 可得结论 |
|---|---|
| ADAPTIVE超过V_BASE但不超过V_ONLY | 没有排除更强读取器的解释 |
| ADAPTIVE超过V_BASE/V_ONLY且真实CLIP超过输入控制 | 支持当前读取协议下存在CLIP判别增量 |
| ADAPTIVE还超过MEAN | 进一步支持自适应权重优于固定读取 |
| MEAN与ADAPTIVE相近，两者都有效 | 利用CLIP有效，但注意力机制优势未证明；优先简单结构 |
| 只有Celeb-DF有效 | 域条件性结果，不声称普遍跨域有效 |
| 各域未排除0 | 当前样本下未检出稳定增量，不等于信息不存在 |

“充分利用”没有已知信息上界，本实验不能证明把 CLIP 的全部信息读完，也不能证明 CLIP 包含 ViT 从未学习的因素。

## 7. 运行命令和产物

先运行合成自检（不产生真实实验结论）：

```powershell
python vit_module/diagnostics/check_pure_vit_clip.py
```

正式运行：

```powershell
python vit_module/diagnostics/pure_vit_clip.py `
  --input vit_module/_g25/diag_runs/extract_regions `
  --output vit_module/_g27/pure_vit_clip_run01 `
  --primary-domains cd2 dfdcp wild `
  --seeds 20261010 20261011 20261012 `
  --folds 3 --epochs 200 --bootstrap 1000 --repeats 5 --threads 1
```

输出目录必须不存在，G26原结果不会覆盖。需要torch/numpy/scikit-learn。当前checkout若只有COMPLETE/config而没有clean_*.npz，须到有完整分片的实验机器运行，不能用汇总AUC生成新实验。

保存features/scores/controls/scalers/region_weights/reader参数、每个种子的修正器和噪声修正器；baseline_oof.npz保存源域样本外基线分数；split.json保存每折行号和视频组；config.json保存选参网格、校准和参数量。scores在统计前落盘；COMPLETE只在全部统计完成后创建。

summary.json包含per_domain、primary_macro、single_seed_auc三部分。primary_macro直接给每个预设比较的delta_macro_auc与ci95；查看结果时不能只选择ensemble或表现最好种子。
