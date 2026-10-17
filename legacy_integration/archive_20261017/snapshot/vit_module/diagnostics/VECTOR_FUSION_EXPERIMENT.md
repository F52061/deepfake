# G29：同维度向量组合与可学习全局权重

## 1. 验证目的与边界

检验 `z=alpha*c'+(1-alpha)*v'` 这种受约束的全局融合，是否比同口径纯 ViT、线性拼接和固定权重更稳定地利用 CLIP image。不是验证“两种表征已经相同”，也不把组合解释成创造新信息。

只读取冻结特征，**两个编码器和原检测器均不更新、不加载训练**。仅训练 CLIP 投影、一个全局 alpha 和新分类头。本轮不是 G28 的残差修正，不保证保留 ViT 原决策。默认目标集已反复查看，属于探索性实验。

## 2. 六组操作一一对应

| 组/操作 | 定义 | 隔离的问题 |
|---|---|---|
| V_MATCHED | 同样预处理的 V，经非仿射 LayerNorm，再用线性分类头 | 加入 CLIP 是否超过同口径纯 ViT |
| CONCAT | `[LN(V_std),LN(C_std)]`，直接线性分类头 | 是否超过简单全局拼接；线性是相对于归一化后的输入 |
| FIXED | 投影 C 后归一化，与 V 按 alpha=0.5 相加，同维分类头 | 共同空间固定组合是否有效 |
| LEARNED | 与 FIXED 相同，alpha=sigmoid(a)，a 初值 0 | 学习一个全局权重是否优于固定组合 |
| NOISE_RETRAIN | FIXED/LEARNED 的同结构、同初始化与预算，CLIP 换高斯噪声并重训 | 收益是否只是投影/新增参数或重新读取 V |
| DONOR | 固定正常 FIXED/LEARNED 模型，C 换同域同 split 其他视频的 C | 是否依赖当前图像的 CLIP 内容 |

额外提供 NOISE_FIXED（固定正常模型换噪声）、CLIP_OFF（固定正常模型，强制 alpha=0），以及历史 V_REFERENCE / V_C_REFERENCE 源域 LR 探针。它们用于诊断，不取代上表的主要基线。CLIP_OFF 的分类头已经适应融合，不能称为独立训练的纯 ViT 成绩。

## 3. 维度、特征层位与归一化

本地 V 为 768 维，C 为 1024 维，来自 `extract.py` 的 `clip.hidden_states[-2][:,0]`。沿用得到内容控制支持的 C，**不切换 CLIP 最后一层 CLS，不使用文本投影后的图文 embedding**。区域 token 和 logits 不进入本轮读取器，加载仅要求 V/C 与身份字段；不需要区域分片。

每个源域 fit fold 独立拟合 V/C 的 StandardScaler。定义：

```text
v' = LayerNorm_without_affine(V_std)
c' = LayerNorm_without_affine(Linear(1024,768)(C_std))
z  = alpha*c' + (1-alpha)*v'
fake_logit = Linear(768,1)(z)
```

LayerNorm 的 eps=1e-5，两路对应位置均无可训练缩放/偏置；混合后**不再加归一化或非线性**。投影可训练且有 bias，分类头有 bias。使用单个 fake logit + BCE，分数越高越假，固定零阈值；诊断标签 `1=fake`。

768 是保留 ViT 输入的诊断起点，不宣称最优；程序使用 V 的实际维度作为融合维度以支持合成自检，正式配置应核对 768/1024。维度匹配不等于坐标语义对齐，分类损失只学习任务可读性，没有额外的图像对齐、cosine 拉近或信息去重损失。

V_MATCHED 的 V 预处理和 768→1 头形状完全相同；同种子与 FIXED/LEARNED 分类头初始参数逐元素一致。FIXED/LEARNED 的投影、分类头初始化完全一致，LEARNED 只多一个参数 a。CONCAT 输入维度不同，分类头无法逐元素配平，不虚报容量一致；其用途是简单读取基准。FIXED/LEARNED 对噪声重训才是投影容量匹配对照。

归一化减弱两路尺度混淆，但 alpha 仍然不是“CLIP 信息占比”。投影方向、分类头读出和两路相关性都会影响贡献。保留 alpha 轨迹、两路向量范数、融合范数、投影后 cosine 及分数分量，不能单用 alpha 或 cosine 判断信息重合度。没有非线性归一化时，全局标量加权和线性头可以改写成线性融合；本协议的投影后 LayerNorm 引入非线性，收益仍可能来自参数约束/优化，不自动是新交互信息。

## 4. 训练、选参与唯一变量比较

仅 FF++ train，以真实视频关联分组执行三折 GroupKFold；每折 StandardScaler 仅见 fit，训练模型仅见 fit 标签。关联真实视频与派生伪造需共享 group，运行者须审计上游 video_id，脚本不能凭字符串证明视频真实关联。

默认三个种子 20261030/31/32，200 epochs，lr=0.001，Adam，全批 CPU，weight_decay=[0.0001,0.001,0.01]，每20epoch保存轨迹。分类头和投影使用同一 decay，**alpha_logit 不做 weight decay**，避免正则隐式拉回0.5；没有另加残差惩罚、alpha 先验、早停或目标域阈值调优。

每组每种子仅按源域 CV 最后一 epoch 的 validation BCE 选 decay；固定训练预算，不按目标或源验证最佳 epoch 报成绩。并在全部源域重训，再一次性评估目标。保存完整网格的 CV 训练/验证 BCE；上边界选择照实标注，不自动扩网格。

LEARNED 与 FIXED 的源域所选 decay 可能不同，因此报告两类比较：

- 每组源域选择后的 LEARNED vs FIXED：比较合法部署流程；
- 相同 decay 的 GRID LEARNED vs GRID FIXED：同初始化/预算/正则，仅学习标量不同，隔离门控变量。目标网格不能被用来挑部署模型。

默认训练 108 个 CV 头、24 个正常全源头（6个 V/CONCAT + 18个融合网格）、30个噪声重训头，另有两个 LR 参考。投影约79万参数，明显多于历史线性探针，必须联合噪声对照解释；耗时需在实验机实测。

## 5. 内容控制与可复现随机数

FIXED/LEARNED 每种子各做五次 donor、固定噪声和噪声重训。donor 不按标签挑选、不跨 domain/split、不从同视频抽样。无合法 donor 时逐域剔除并报告有效样本量；主三域覆盖不全，宏平均标为 incomplete，不回退成证据。

噪声在标准化 C 空间按源训练逐维均值/标准差匹配；每 `(seed,domain,split)` 独立生成，目标评估行删增不改变源训练噪声或 donor。噪声重训保持 V 不变，投影/分类头/alpha 都重新训练，初始参数与真实组相同，使用对应真实组**源域选定的 decay**；不独立搜索噪声最优超参。因此这是固定协议容量控制，不是噪声模型能力上界。

保存每次 donor 的行位置（相对于本次 scores.npz 的排列）及有效掩码、原 row_id、噪声均值/标准差/种子和公式。不是反复保存巨大噪声数组；精确重建需保持样本身份/顺序、NumPy版本和 float32 转换顺序。

## 6. 指标和判定逻辑

主宏平均只包含 CD2/DFDCP/Wild 的逐域 AUC 算术平均，按各域内视频配对 bootstrap 1000次；CD1 是 CD2 子集，仅单列，FFIW 不入主指标。逐种子结果是主要证据，AUC 均值/标准差与分数平均 ensemble 单列。区间条件于已训练头，所有比较是探索性描述，不将多个种子或重复当成独立目标集。

三项主要问题及比较键在 summary.primary_questions 中明确：

1. LEARNED vs V_MATCHED：实际 CLIP 增量；
2. LEARNED vs FIXED：是否需要学习权重；
3. LEARNED vs CONCAT：是否优于简单读取方式。

FIXED vs V_MATCHED/FIXED vs CONCAT、真实组 vs 全部自身控制组成补充链。判断：

| 观察 | 可得结论 |
|---|---|
| 真实组超过 donor，但未超过 V_MATCHED | 输入依赖，不是纯 ViT 之上的稳定增量 |
| 真实组超过 V_MATCHED，但不胜噪声重训 | 未排除新增参数/优化解释 |
| 真实组稳定超过 V_MATCHED 与自身内容控制 | 支持当前读取协议的真实 CLIP 增量 |
| FIXED 与 LEARNED 接近 | 未证明学习权重的必要性；优先简单方案 |
| 源选 LEARNED 更好，同 decay 比较不明确 | 不能单独归因于学习 alpha，可能包含选参效应 |
| 不超过 CONCAT 或历史线性拼接 | 没有证明新结构的性能必要性；不能从非显著推出等价 |
| alpha 接近0，或两路 cosine 很高 | 不单独证明 CLIP 无用、冗余或表征已对齐 |

rescue/harm 使用每个比较基线的固定零阈值错误集合，并与总体 AUC 同看；失败子集 AUC有选择偏差。固定 CLIP_OFF 也只是已训练模型干预，不替代 V_MATCHED。

当前 checkpoint 曾用 FFIW 选模，目标集已反复查看；默认 exploratory，不宣告零目标选模或正式确认。新视频留出可用 CSV row_id 清单，经 `--evaluation-manifest` 选择整视频，并声明 `--evaluation-status new-holdout`；脚本不能证明该视频此前从未查看，运行者仍需审计并使用源域选模 checkpoint。

## 7. 运行命令

```powershell
python vit_module/diagnostics/check_vector_fusion.py
python vit_module/diagnostics/vector_fusion.py `
  --input vit_module/_g25/diag_runs/extract_regions `
  --output vit_module/_g29/vector_fusion_run01 `
  --primary-domains cd2 dfdcp wild `
  --seeds 20261030 20261031 20261032 `
  --folds 3 --epochs 200 --lr 0.001 `
  --weight-decays 0.0001 0.001 0.01 `
  --bootstrap 1000 --repeats 5 --threads 1
```

需要真实 clean_*.npz，不能只用汇总 JSON 运行。输出目录必须不存在，中断产物保留但无 COMPLETE，不支持自动续跑。旧 G27/G28 和原模型不修改、不覆盖。

## 8. 存档与读表次序

保存 scores.npz（全部分数/身份）、features.npz（实际 V/C）、controls.npz、vector_diagnostics.npz（逐样本向量与分数分量诊断），config.json（维度/层位/选参/alpha/版本），split.json、scalers.npz 与折内标准化参数、training_trace.json、所有头的 .pt、LR参数、comparisons.json 和 summary.json。向量诊断保存范数/cosine/头分量，不默认保存每个头的巨大768维融合数组，可由特征/标准化/模型重建。

统计前保存全部分数；另保存 code_snapshot/（执行代码与规格）、input_config.json、输入分片SHA256及输出 artifact_hashes.json。哈希清单不包括其自身与最后写入的 COMPLETE。仓库忽略 .npz/.pt，须在实验机另行完整存档。

先读 LEARNED/FIXED 相对 V_MATCHED，再看真实组自身控制，随后读 LEARNED vs FIXED 的源选/同 decay比较和 CONCAT 对照；最后看逐域、alpha轨迹、范数/分数分量、rescue/harm。不得先从目标全网格挑一个最好成绩，再倒推创新结论。
