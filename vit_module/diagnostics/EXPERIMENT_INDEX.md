# 本地实验索引与存档口径

本索引整理 G23–G29 的入口，不移动历史结果目录、不改运行路径、不删除分数或权重。更早的架构排查仍以根目录 `WORKLOG.md` 和 `FINDINGS_问题验证数据.md` 为入口。

## 先读什么

1. G27 历史复核：[G27_REVIEW.md](G27_REVIEW.md)，包含 JSON 核对结果与解释边界。
2. 研究证据：根目录 [FINDINGS_问题验证数据.md](../../FINDINGS_问题验证数据.md)，最新运行结果为 G28 / P16；非显著不等于等价，多种控制不等于独立目标验证。
3. 完整过程：根目录 [WORKLOG.md](../../WORKLOG.md)，G28 为 D.70–D.72，G29 设计为 D.73。
4. 运行条件：[README.md](README.md)。实验规格记录设计，不自动代表实验已经成功。
5. 下一轮设计：[VECTOR_FUSION_EXPERIMENT.md](VECTOR_FUSION_EXPERIMENT.md)，G29 按六组控制验证标量向量组合；尚无真实成绩。G28 规格仍保留在 [CLIP_READOUT_EXPERIMENT.md](CLIP_READOUT_EXPERIMENT.md)。

## 实验地图

| 编号 | 验证问题 / 主要基线 | 脚本和规格 | 正式结果目录 | 当前状态 |
|---|---|---|---|---|
| G23 | 单轴受控脆弱性、固定文本提示的增量；区分检测器重放与 V-only 探针 | `extract.py`、`analyze.py`；`README.md` | `_g23/diag_runs/extract_full/`、`_g23/diag_runs/analysis_full/` | 已运行；文本候选不等于全部 CLIP 语义；见 P10–P12、D.51–D.55 |
| G24 | `[V,Cres]` 相对 V 的增量及噪声控制 | `residual_clip.py`；`RESIDUAL_EXPERIMENT.md` | `_g24/residual_run01/` | 已运行；残差是重新参数化，不是纯独立信息；见 P13.1、D.56 |
| G25 | CLIP 九区域和关系特征的可读性 | `extract.py --save-regions`、`local_analyze.py`；`LOCAL_EXPERIMENT.md` | `_g25/diag_runs/extract_regions/`、`_g25/local_run01/` | 已运行；为 G26/G27 提供同一冻结输入；见 P13.2、D.57 |
| G26 run01 | 条件区域读取能否改善原完整检测器 | `conditional_clip.py`；`CONDITIONAL_CLIP_EXPERIMENT.md` | `_g26/conditional_clip_run01/` | 历史版本，选参标准化泄漏已登记；不作当前确认依据 |
| G26 run02 | 修复选参隔离后，复核域条件性及跨域 donor | 同上 | `_g26/conditional_clip_run02_domain/` | 当前 G26 版本；仍以完整检测器为基线，不是纯 ViT；见 P14.8、D.64–D.65 |
| G27 run01 | 相对纯 V 线性探针，自适应区域读取能否提供增量 | `pure_vit_clip.py`；`PURE_VIT_CLIP_EXPERIMENT.md` | `_g27/pure_vit_clip_run01/` | ADAPTIVE 未获支持，POOLED 有探索性正线索；见 P15、D.67–D.69 |
| G28 run01 | POOLED 自身内容控制、统一超参比较、冻结模型缩放/权重干预 | `clip_readout.py`、`check_clip_readout.py`；`CLIP_READOUT_EXPERIMENT.md` | `_g28/clip_readout_run01/` | 已运行；全局内容控制较强，源选纯 V 增量仍未确认；见 P16、D.71–D.72 |
| G29（待运行） | 同归一化纯 V、拼接、固定/学习标量组合与噪声/donor 对照 | `vector_fusion.py`、`check_vector_fusion.py`；`VECTOR_FUSION_EXPERIMENT.md` | 预定 `_g29/vector_fusion_run01/` | 独立实现；保留旧模型/结果；真实实验需完整 clean V/C 特征；见 D.73 |
| G30（待运行） | 同一改进 ViT 权重下，V 原生/配平、全局 CLIP、有效 Bridge 的增量 | `incremental_fusion.py`、`check_incremental_fusion.py`；`INCREMENTAL_FUSION_EXPERIMENT.md` | 预定 `_g30/incremental_run01/` | 当前核心验证；需要逐样本 native score 和修复后 Bridge 特征；见 D.74 |

结果路径均相对于 `vit_module/`。`smoke`、`extract_smoke`、`analysis_smoke`、`local_smoke` 是冒烟产物，不能合并到正式结果。`self_check.py`、`check_residual.py`、`check_conditional.py`、`check_pure_vit_clip.py`、`check_clip_readout.py`、`check_vector_fusion.py` 使用合成数据，只验证程序。

## 三种基线不可混用

- G26 的 `s_V` 实为原完整融合检测器的 fake-logit 差，包含 CLIP；名字不能作为纯 ViT 的证据。
- G27 的 `V_BASE` 为仅 FF++ train 拟合的 `StandardScaler + LogisticRegression(C=0.001)`，输入仅 ViT CLS；它不是单独训练的 ViT 原生分类头。
- G27 的 `DETECTOR_REFERENCE` 是原检测器参考，不参与残差训练和选参。不同头、不同基线不能直接用两轮增益相减。

诊断文件中 `1=fake`，历史 `_g16` 标签中 `1=real`。G27 主宏平均只含 CD2/DFDCP/Wild，CD1 单列且是 CD2 子集；单种子 AUC 均值与 ensemble 分数平均后的 AUC 分开报告。

## 本 checkout 的存档完整性

本次核查：G23–G27 上表正式目录在当前 checkout 主要保留 JSON 汇总、配置和完成标记；G26/G27 另有 `split.json`。G27 run01 当前只有 `COMPLETE`、`config.json`、`split.json`、`summary.json` 四个文件。

运行记录声称保存的 `scores.npz`、`features.npz`、`controls.npz`、区域权重、读取器参数及 `.pt` 权重，在当前 G27 目录不存在。仓库 `.gitignore` 排除了 `.npz`/`.pt`/`.csv`/`.log` 等；这解释了它们不受版本管理，但不能证明它们仍在其他机器上。`COMPLETE` 证明运行结束，不证明同步后的目录包含全部原始产物。

因此本次能核对汇总值、宏平均算术、选参表和控制次数，不能从逐样本分数独立重算 AUC/置信区间，也不能新增真实验证实验。补齐原始产物时应从实验机核对并复制，不覆盖或删除原文件；至少保留逐样本分数与身份、参数、配置、代码快照及校验清单。旧 G26 的大噪声文件仍按原样保留，不能因 G27 改用种子复现就将它删掉。

G27 的输入配置 SHA256 与本地 G25 配置一致；运行登记的三个代码 SHA256 与当前代码字节不一致，统一 CRLF→LF 后仍不一致，原因未确认。正式复现前需取回运行时代码快照，不能自行改写配置哈希来宣称一致。
