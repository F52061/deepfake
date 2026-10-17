# 论文骨架 v0.2 — 跨域伪造检测：单轴脆弱性与正交备份（工作文档，非投稿稿）

> 创建 2026-09-09。本文是论文**组织稿**：每节给职责、已有证据（数字+报告出处）、缺口实验（T/G 编号）。
> 状态图例：✅证据已闭环 | ⏳缺口实验待跑 | ✍️正文待写 | ⚠️风险/需裁定
> 铁律：任何数字以 WORKLOG.md §A/§C.1 与 `vit_module/_probe|_phaseA|_tsne|_spectrum/*report*.txt|json` 为唯一来源；改动数字必须同时改三处（本骨架、WORKLOG、report）并注明协议（C=1e-3 vs C=1.0）。

---

## 0. 定位备忘（先读）

- **不追 SOTA 数字**（用户已确认跨域弱于 SOTA 是现状）。卖点 = (a) 对"冻结双编码器检测器为什么跨域会坏"给出**机制级归因**（迄今文献没有对 ViT+CLIP 双编码器做方差谱归因 + 因果截断的）；(b) 由此提出**零训练、零标签、forward-only 的测试时可靠性路由**（T-Route），并在 oracle 上界框架下诚实度量它能收回多少。
- 诚实基线声明（Intro/Limitation 必须写）：我们研究的是 shift 下的失效机制与补救，非端到端 SOTA 比拼；与 SOTA 的差距用一张公开数字对照表显式呈现。
- 证据载体 = 冻结 PDI-ViT(net_050) + 冻结 CLIP ViT-L/14-336 + bridge_v2 检测器（stage-1 权重），**特征级 LR 探针**（video-disjoint 协议）。"探针级 vs 模型级"的边界见 §9 G4。
- 命名：检测器桥接后的完整模型沿用 M2F2-Det-hyy 描述；论文中架构叙述保持通用（"a frozen forgery-pretrained ViT + a frozen CLIP ViT-L, fused by an adapter"）。

## 1. 标题候选 与 摘要草稿（✍️）

标题候选（各附风险注释）：
- **A. The One-Axis Oracle: Where Cross-Domain Deepfake Detection Breaks in Representation Space**（分析导向，稳妥，审稿人预期容易管理）
- **B. Fragile Axis, Orthogonal Backup: Unsupervised Test-Time Channel Routing for Cross-Domain Deepfake Detection**（方法导向，最大卖点=路由，风险=E2 显示合并域每个距离段 V≥C，路由收益存疑 → 见 §6 Q3 与 ⚠️3）
- **C. Spectral Anatomy of Dual-Encoder Deepfake Detectors under Domain Shift**（中性，分析+方法都装得下）

摘要草稿 v1（约 200 词，**加粗段数字待 Phase F 回填**）：
> "Deepfake detectors built on frozen dual encoders — a forgery-pretrained ViT and CLIP — reach near-perfect in-domain accuracy but degrade sharply on unseen domains, and score-level fusion of the two branches is nearly redundant. We ask where this cross-domain fragility lives in representation space and whether the second, near-orthogonal branch can be exploited at test time without labels or retraining. Under a video-disjoint probing protocol across five target domains we find three mechanism-level facts. First, in-domain score fusion of the branches is decision-redundant (concat gains <0.4%) although the branches are representation-near-orthogonal (linear CKA≈0.14). Second, the ViT branch's real-vs-fake evidence is concentrated on a single dominant variance direction: one principal direction alone reproduces 0.983 in-domain AUC, the retained top-50 directions reach 0.989, and the remaining 718 low-variance directions are at chance (0.489); the ViT is a one-axis oracle whose accuracy decays monotonically with per-sample distance from the source manifold (0.92→0.72 in distance terciles). Third, the CLIP branch distributes its evidence over hundreds of near-orthogonal directions and outperforms the ViT only where that single ViT axis is broken (lowest-overlap domain FFIW, in-distribution half: CLIP 0.937 vs ViT 0.863). We turn these findings into T-Route, a zero-parameter label-free test-time router that scores per-sample trust in the ViT channel via source-manifold distance, channel disagreement and spectral-concentration conformity, then abstains or falls back to CLIP when trust is low. **At 90%/75%/50% coverage across five domains, T-Route reduces error by X/X/X% over always-ViT and beats confidence-based abstention baselines, recovering Y% of the oracle-switch ceiling** (numbers pending Phase F)."

## 2. Introduction 计划（✍️）

段落职责：
1. 问题：跨域泛化是深伪检测的第一瓶颈；主流对策 SBI/数据堆叠，成本高、无机制理解。
2. 研究载体：双编码器（语义 CLIP + 取证 ViT）是近年主流结构（M2F2-Det 等），但文献没有回答"失效在表示空间哪里、互补通道如何被利用"。
3. 三个机制发现 F1–F3（一段一个，见 §5）→ 提出 T-Route（一段）。
4. Contributions 列表（英文，投稿时直接抄）：
   - **(C1) Redundancy–orthogonality dissociation.** In-domain score fusion of a ViT/CLIP dual encoder is decision-redundant (<+0.4% AUC) although the channels are representation-near-orthogonal (CKA≈0.14–0.17); score agreement (logit corr +0.67) coexists with decorrelated evidence axes (top-direction corr −0.35). [§5.1]
   - **(C2) One-axis fragility.** With causal truncation tests, projection nulls and model-weight spectral attribution, we show the ViT channel's deepfake evidence sits on its single largest-variance direction (>97% of Fisher discriminability; tail-718-d ≈ chance), and this axis breaks monotonically with source-manifold distance (AUC 0.92→0.72). [§5.2, §5.4]
   - **(C3) Regime-specific orthogonal backup.** CLIP evidence is diffuse (≈274 dirs for 90% of discriminability) and near-orthogonal to the ViT axis; it outperforms the ViT only on the lowest-overlap domain (FFIW) and specifically in its in-distribution half (0.937 vs 0.863), with the advantage carried by directions beyond the top-50 that are *harmful* under other shifts. [§5.3]
   - **(C4) T-Route**: zero-training, label-free, forward-only per-sample trust routing for frozen dual encoders (abstain/fallback), with oracle-ceiling quantification. [§6, ⏳Phase F]
5. 负结果显式声明（审稿人管理）：全局"远→信 CLIP"门控被证伪；域内融合冗余；SBI 只作对照不作贡献。

## 3. Related Work 提纲（✍️，引用待补）
- 深伪检测与泛化：FF++/Celeb-DF/DFDC/FFIW/WildDeepfake 协议差异；SBI（自监督动态增强）及其代价。
- 多模态/双编码器检测（M2F2-Det, CLIP-based forensic, 语义-取证互补）— 现有工作只报融合收益，**不做谱级归因**（本文空白点）。
- 表示几何分析：线性探针、CKA、PCA/Fisher 谱、秩/参与率（连接 R-test 的 V 低秩观察）。
- 测试时鲁棒性对照：selective prediction（弃权/风险覆盖曲线）、OOD 检测（MSP/energy 作为弃权基线）、TTA（TENT 等：需反向/需标签族）与 source-free DA — 本文 T-Route 定位：**零训练、无标签、单次前向**的实例级通道路由，作为二者的"最简端点"。
- 诚实差异点：大多数 OOD 方法在语义分类上验证；**深伪检测的伪造特征是学习过的单一轴**（我们的发现）使"轴支持度"成为天然信任信号。

## 4. Setup 与协议铁律（✅ 数字已冻结；正文写作时直接抄录）

架构/通道（探针特征来自冻结 bridge_v2 前向，视频级切分）：
| 记号 | 定义 | 维度 |
|---|---|---|
| V | PDI-ViT CLS 原始特征 | 768 |
| C | CLIP ViT-L/14-336 CLS 原始特征 | 1024 |
| R | 残差 V−Ridge(V\|C)（训练集拟合） | 768 |
| V_proj / C_proj | 检测器 head 输入投影 | 768 |
| F | 融合后 head 输入（组成待 stage-1 代码核对：见 §11） | 1664 |

协议（任何实验不许偏离）：
- 单一固定探针：`StandardScaler` 只 fit 源训练集 → `LogisticRegression(C=1e-3, lbfgs)`；AUC 只在 video-clean test（FF++ 800，与 train 2200 零视频重叠）或目标域子集上评估。**锚点**：V=0.9852 / C=0.9108 / R=0.7034（C=1e-3；C=1.0 时的 0.9881/0.8781/0.6721 只作附录敏感性，C-grid 见 spectrum_report AUDIT 节）。
- 标签：y=1 真实 / 0 伪造；AUC 在标签交换下不变。
- PCA 约定：原始协方差（center-only），特征空间载荷 = **右奇异向量** `E=Vt.T`，投影 `(X−μ)@E`；'corr' = StandardScaler 后 PCA（仅敏感性）。
- 全部离线 npz；禁 GPU 前向；CPU 单线程（OMP/MKL/OPENBLAS/NUMEXPR=1）。

数据文件（证据台账唯一入口）：
| 文件 | 内容 | keys |
|---|---|---|
| `vit_module/_probe/probe_feats.npz` | FF++ 3000 视频级（2200/800） | V C R V_proj C_proj y paths vids train_mask |
| `vit_module/_tsne/feats_multi.npz` | 6 域 2300（cd1/cd2/dfdcp/ffiw/wild 各 300 + ffpp 800） | V C F y domain vid path |
| 报告 | `_probe/residual_probe_report.txt`（R-test, C=1.0 协议）、`_phaseA/phaseA_report.txt`、`_spectrum/spectrum_report.txt` | — |

## 5. 发现章（Analysis，核心证据章）— claim 级组织

### 5.1 F1 冗余–正交解耦（✅ 证据闭环）→ 支撑 C1
| 量 | 值 | 出处 |
|---|---|---|
| concat(V_proj,C_proj) 域内 | 0.9874 vs V_proj 0.9850（+0.24%）；MLP +0.35% | Phase A E0 |
| probe(V)/probe(C)/probe(R) 域内（C=1e-3） | 0.9852 / 0.9108 / 0.7034 | spectrum E3c |
| CKA(V,C) | ≈0.14（全量）~0.169（test） | R-test / E3c |
| V/C 顶判别轴投影相关（test） | −0.35 | E3c |
| V/C probe logit 相关（test） | +0.67 | E3c |
| V 有效秩 | 参与率≈30，90%方差能量≈5 维（E3a 能量阈值 1/1/5） | R-test / E3a |
一句话（论文用）：*分支表示近乎正交，判决却高度一致 → 域内"分数层融合"无可加性，冗余在决策层而非表示层。*

### 5.2 F2 V 通道 = 单轴 oracle；判别与方差近乎单调（✅ 证据闭环）→ 支撑 C2
| 模块 | 数字 | 读法 |
|---|---|---|
| E3a Fisher 谱 | max-F = 方差 rank 0；Spearman(logλ,F)=+0.9992；PC0 含 >97% 总 Fisher 量、占 62% 方差；90% 判别累计=1 方向 | 判别≈全在最大方差方向 |
| E3b 因果截断（raw≈corr，各 K 差≤0.004） | top1=0.9827≈full 0.9852；top50=0.9891；**tail50(718维)=0.4885≈随机**；top200=0.9877；tail200=0.4607 | 保留头部=保全部；只留尾部=归零 |
| E3b-null | 标签置乱 30 seeds：0.477±0.101 / 0.514±0.043；Gaussian 尾替代=0.4924；随机 718/768 子空间=**0.9861** | 非噪声伪影；信号只在大方差子空间 |
| E3e 真实权重（V_proj 探针头） | 99.9% 权重能在 top 带；PC0=52.6%、top5=91.6%；Spearman(logλ,g)=+0.9998 | 学习到的分类器**从未压**小方差方向 |
| 对应 R-test 旧观察 | V/R 顶判别特征向量 |cos|=0.9928（残差几乎共线 → R 的弱判别是同一根轴的弱回声） | E3c |
一句话：*该通道是"单轴 oracle"——一个 ~PC0 方向承载几乎全部取证证据；域移打废这根轴 = 通道失效。*

### 5.3 F3 C 通道 = 弥散正交备份，优势有 regime 特异性（✅ 证据闭环）→ 支撑 C3
| 量 | V | C | R |
|---|---|---|---|
| full 域内 AUC | 0.9852 | 0.9108 | 0.7034 |
| Spearman(logλ,F) | +0.9992 | +0.9994 | +0.9989 |
| 90% 判别所需方向数（能量占比） | 1 (0.62E) | **274 (0.74E)**；50%→34 方向 | 91 (0.83E) |
| 域内 top50 vs full | 0.9891≈full（头部足够） | **0.8197<full（头部不够，需>50 方向）** | 0.6918<full |
跨域带迁移（E3d；与 Phase A E1 逐域对齐 = 协议互证）：
| 域 | V full | V top50 | V tail50 | C full | C top50 | C tail50 |
|---|---|---|---|---|---|---|
| cd1 | 0.8286 | 0.8164 | 0.6296 | 0.6559 | **0.8028** | 0.5114 |
| cd2 | 0.8633 | 0.8392 | 0.6238 | 0.7227 | 0.7321 | 0.5284 |
| dfdcp | 0.8261 | 0.7907 | 0.5356 | 0.7068 | 0.6963 | 0.4812 |
| ffiw | 0.8244 | 0.8061 | 0.6399 | **0.8344** | 0.7272 | 0.4579 |
| wild | 0.8090 | 0.7611 | 0.6070 | 0.7170 | 0.6520 | 0.5430 |
读法：
- **ffiw 是唯一 C>V 域**（也是 E1 中 V 子空间重叠最低的域，ovl_V10=0.602）；C 的 FFIW 优势由 top-50 **之外**的弥散方向承载（full>top50 差 +0.107）→ 尾部在 ffiw 是承重墙。
- **cd1 相反**：C 的弥散尾部是**有害噪声**（top50 比 full 高 +0.147）→ 朴素 concat/平均会把负资产也掺进来 → 融合需要**路由而非平均**。
一句话：*备份通道的价值是域特定的；"一刀切融合"把它稀释掉——这正是路由方法的动机，也是本论文方法章的第 0 步证据。*

### 5.4 F4 实例级脆弱性可预测 + 距离门控证伪（✅ 证据闭环）→ 支撑 C2/C4
- E2（合并 5 域三分位，k=10）：V 0.921/0.759/0.720，C 0.736/0.678/0.625 → **每个距离段 V≥C："远→信 C"全局门控被证伪**；V 随源距单调掉点（域内近/远对照成立，排除域混淆）→ 源距 = 实例级 V 可靠性信号。
- ⚠️ 但注意：V 最远段 0.720 仍 > C 最远段 0.625 → **"低信任 → 换 C"的朴素路由在合并域不成立**（C 只在 ffiw 近源半 0.937 vs V 0.863 反超）→ 路由对象必须是有条件的（见 §6 Q3 与 ⚠️3）。
- E1 序数证据（n=5 弱，仅作动机不作结论）：V 空间重叠/对齐 vs V 域间 AUC Spearman +0.74~0.94；wild 破单调（重叠高但 AUC 低）→ "本身难分"≠"分布远"，两者都要建模。

### 5.5 负结果汇总（显式写出，审稿人管理）→ 支撑论文可信度
1. 域内融合决策冗余（E0）——不是融合方法论文。
2. 全局距离门控证伪（E2 合并三分位）——不是 OOD-gating 方法论文。
3. 跨域整体弱于 SOTA（用户确认）——Intro/Limitation 显式对照。

## 6. 方法章草案：选择性预测 + 失败预测（T-Route 已按 Phase F 判定收缩，2026-09-09）

> **Phase F 已跑完（WORKLOG §C.3）**：Q3 路由转 C 证伪（oracle ceiling +16.1pt 但固定比例转 C 全≤+0.5pt，根因 C 跨域整体弱 + 信号只预测"V 错"不预测"C 更好"）→ 预注册出口触发，方法形态从"路由"收缩为 **失败预测 + 选择性弃权**。Q2 正结果：免费 MSP 弃权在 50% coverage 把合并域错误率 −41%(0.269→0.159)，源距弃权不占优。Q1：轴支持度域内 0.86 / 跨域 0.695（<0.70 门槛但机制自洽），MSP 0.68 打平。G5（WORKLOG §C.4）：真实头 C 块权重 62%>V 块 38% → 反直觉：问题不是忽略 CLIP 而是权重锁死源分布；V 侧块 top-5 93.8% 复现探针集中结论于真实头。

**问题形式化**：给定 x 与冻结双通道输出 (s_V(x), s_C(x), z_V(x))，求 (i) 失败预测器 τ̂(x)（预测 V 会判错）、(ii) 决策策略 π(τ̂)：弃权（coverage 可调）或转 C / 混合。评价 = 失败预测 AUROC、risk@coverage 曲线、域级固定覆盖率 AUC；上界 = oracle π*（逐样本知道 V 对错）→ 度量代理能收回上界几成。

**三个无标签信任信号族**（全部单次前向可算）：
| 信号 | 定义 | 依据 | 风险 |
|---|---|---|---|
| τ_dist | 到源训练流形的逐样本距离（E2 已证单调） | F4 | 抓不到 ffiw 近源半的轴断裂 |
| τ_spec | 轴支持度/谱集中度合身性：|z_V·PC0| 相对类典型值 | F2：单轴机制的天然信任量 | 需设计标准化，防标量折叠 |
| τ_disc | V–C 判决分歧 | 5.1：logit 相关 0.67 → 分歧=证据冲突信号 | 分歧也包含"都错"区 |
（可选 τ_ovl：域级批次统计，不作实例级。）

**三个可证伪问题（Phase F 一次跑完，预注册判定规则）**：
- **Q1 失败预测**：τ 族（单用/加权合）预测 V 错误的 AUROC，逐域 + 合并，vs 0.5 与 MSP(概率置信) 基线。成立门槛：合并域 AUROC ≥0.70（E2 三分位提示有量）。
- **Q2 选择性弃权**：按 τ 弃权后 risk@coverage 曲线；对照 = 基于 softmax 置信的弃权（MSP/energy，在探针概率上实现）。成立门槛：同覆盖率下错误率显著更低（DeLong/bootstrap CI）。
- **Q3 通道路由**：oracle 先算"V 错且 C 对"实例占比与 τ 的相关；再评估"τ 低于 θ → 转 C"是否胜过同覆盖率弃权。**预注册诚实出口**：若 C 无可救回实例（合并域 E2 提示可能如此），Q3 降级为负结果并入讨论——路由只在轴断裂 regime（如 ffiw 近源半）可用，方法章收缩为"选择性预测 + 失败预测 + regime 检测"。

## 7. 实验表册（表号全局唯一；状态 = DONE/⏳）
| 表 | 内容 | 状态 | 数字来源 |
|---|---|---|---|
| T1 | 域内：V/C/R/V_proj/C_proj/concat 探针 + 真实 head 对照 | ✅（head 对照待 G5） | phaseA/spectrum |
| T2 | 跨域迁移：V vs C 五域 | ✅ | E1/E3d |
| T3 | 距离三分位 V/C | ✅ | E2 |
| T4 | 截断因果 + null | ✅ | E3a/b |
| T5 | 谱集中度三通道对比 + CKA 矩阵 | ✅ | E3c/e |
| T6 | Q1 失败预测 AUROC（τ 族 × 域） | ⏳ Phase F1 | — |
| T7 | Q2 risk@coverage（τ vs MSP/energy） | ⏳ Phase F1/F2 | — |
| T8 | Q3 oracle 上界 + 代理回收 | ⏳ Phase F2 | — |
| T9 | 与 SOTA 公开数字的诚实对照表 | ✍️ 待收集 | 论文引用 |
| T10 | （可选）模型级一致性（bridge_v2 9 域） | ⏳ G4 GPU | — |
| T11 | （可选）SBI/数据量对照 baseline | ✍️ 仅当审稿预期 | — |
| T12 | E4 融合头对比（跨域 concat 首次测量 + transformer 交互证伪 + 容量-泛化单调律） | ✅ 2026-09-09 | `_fusion/fusion_report.txt` |
| T13 | Phase F：Q1 失败预测 AUROC × 信号、Q2 risk@coverage(MSP vs 源距)、Q3 oracle ceiling +16.1pt + 转 C 证伪 | ✅ 2026-09-09 | `_selective/selective_report.txt` |
| T14 | G5 真实 head 权重三块占比(C62/V38/bridge0.03) + 谱集中度 + G3 bootstrap CI（跨域 concat 增益不显著） | ✅ 2026-09-09 | `_head/head_report.txt` |

## 8. 图清单
| 图 | 内容 | 状态 |
|---|---|---|
| Fig1 | 架构/特征通道示意（V/C/R/proj→head） | ✍️ |
| Fig2 | logλ–F 谱（V vs C 双面板）+ 累计判别曲线 | ✅ 数据在手（spectrum_curves.npz） |
| Fig3 | topK/tailK 截断曲线（"因果刀"） | ✅ 数据在手 |
| Fig4 | 域地图：重叠/源距 × (V,C) AUC | ✅ phaseA 数据在手 |
| Fig5 | V 脆弱性：距离段箱线 + 逐样本（t-SNE 底色已有） | ✅ 部分 |
| Fig6 | risk@coverage / 路由回收曲线 | ⏳ Phase F |
| Fig7 | FFIW case study（近源半 C 反超区可视化） | ⏳ 小跑即得 |

## 9. 缺口清单（按优先级；决定下一步的入口）
| 编号 | 内容 | 支撑 | 成本 | 类型 |
|---|---|---|---|---|
| G1=Phase F1 | oracle 上界 + Q1/Q2 逐域（feats_multi 现有 2300 样本即可；需补：FF++ test 800 上 τ 的实例级评估也在 probe npz 内） | §6 | ~1h CPU | 离线 |
| G2=Phase F2 | 代理组合 + MSP/energy 基线 + bootstrap CI + Q3 诚实判定 | §6 | ~2h CPU | 离线 |
| G3 | 统计加固：AUC bootstrap CI 全表统一跑一遍（key 数字）+ seed 固定声明 | §4/§7 | ~1h CPU | 离线 |
| G4 | 端到端模型级一致性（bridge_v2 + 9 域评测重跑 bprsida7g；曾在 GPU1 被 kill） | §5 边界 | 数小时 GPU | GPU 可选 |
| G5 | bridge_v2 **真实 head 权重**的谱归因（读 pth 线性层，无前向；E3e 目前是探针头） | C2 增强 | ~0.5h CPU | 离线 |
| G6 | 数据集实名核对（cd1/cd2/dfdcp/ffiw/wild → 论文实名 + 引用）+ F(1664) 通道组成核对 | §4/§7 | 0.5h | 查证 |
| G7 | SBI/data-scale baseline（只在审稿预期需要时做；需训练，成本高） | §2 | 天级 GPU | 暂缓 |

> **2026-09-09 状态更新**：G1(Phase F1)✅、G2(Phase F2)✅、G3✅、G5✅ 均已执行（数字见上表 T13/T14 与 WORKLOG §C.3/C.4）。剩 G4(端到端 9 域 GPU，可选)、G6(实名核对)、G7(SBI baseline，暂缓)。G4 因 G5 已证真实头复现探针结论而优先级下调，待用户裁定。

## 10. 复现索引（数字 → 脚本 → 报告）
- R-test：`_probe/`（C=1.0 协议）→ `residual_probe_report.txt`
- Phase A E0/E1/E2：`_phaseA/run_phaseA.py` → `phaseA_report.txt` + `phaseA_summary.json`
- E3 全套：`_spectrum/e3_abc.py && e3_audit.py && e3_d.py && e3_e.py`（置顶 CPU 限制，顺序执行；e3_abc 先 reset report）→ `spectrum_report.txt` + `spectrum_curves.npz`
- Python：`C:/Users/Supor2/.conda/envs/M2F2_Det/python.exe`

## 11. 待你裁定 / 待核对清单
1. 标题方向 A / B / C（§1）→ 决定方法章强弱措辞。
2. 若选 B（路由方法论文）：接受 Q3 可能降级为负结果的风险，核心贡献落在 Q1+Q2+机制；若选 A：分析为主、T-Route 收尾于 discussion 一段。
3. 核对：真实分类头输入 F(1664) 的组成（`vit_module/vit_m2f2_detector_unified.py`）、五个目标域在数据集清单中的实名（`dataset/data_2023/*_test.txt`）。
4. 补测优先级建议：**G1(G2)+G3+G5 全离线共 ~4h CPU，可一次批准**；G4 待 G1 判定后再定。
