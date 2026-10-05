# G28：CLIP 内容增量与局部读取失效的受控验证

## 1. 问题与证据边界

G27 给出了值得继续验证的证据：POOLED 在 DFDCP/Wild 三个种子方向为正，DFDCP ensemble 的增量区间排除零；但三域宏平均区间仍含零，且没有 POOLED 自身的输入控制。因此当前只能说存在域条件性正线索，不能预设“CLIP 的独立增量已确证”或“注意力已被证明是失败原因”。

本轮不训练编码器、不接入原架构、不使用 CLIP text。先回答两个问题：

1. **内容增量**：POOLED 相对纯 V 的收益是否来自当前图像的 CLIP image，而不是新增容量、校准或噪声？
2. **失效定位**：局部读取不佳更符合修正过强、源域选参差异，还是当前区域聚合方式无法稳定迁移？

实现为 `clip_readout.py`，自检为 `check_clip_readout.py`。独立输出到 `_g28/`，G27 代码和结果保持不变。所有条件在查看本轮目标成绩前固定，不从完整目标网格中挑最佳模型。

## 2. 实验组及严格基线

复用带 `--save-regions` 的完整 clean 提取。V 是冻结 ViT CLS，C 是 CLIP 全局 CLS，R 是 CLIP 九区域的逐区域 L2 归一化表示。POOLED 读 C，**不是 R 的均值**。

| 组 | 输入和操作 | 回答的问题 |
|---|---|---|
| V_BASE | 源域 StandardScaler + LR(C=0.001)，仅 V | 纯 ViT 表征能达到多少 |
| CALIBRATED | 源域 OOF 拟合的正仿射校准 | 阈值收益是否只是校准；AUC 理论不变 |
| V_C_LINEAR | 源域 `[V,C]` 线性读取 | 简单全局融合基准 |
| V_ONLY | 容量匹配 V 残差头 | 更多非线性容量是否足够解释 |
| POOLED | V 查询/修正头 + 全局 C | 全局内容增量候选 |
| MEAN | 同结构、九区域固定 1/9 权重 | 区域投影/聚合与局部表征 |
| ADAPTIVE | 同结构、V 查询产生九区域权重 | 自适应权重是否超过固定读取 |

四种非线性头均 `score=V_BASE+delta`，零初始化修正头。POOLED/MEAN/ADAPTIVE 参数结构和同种子初始化一致，V_ONLY 参数量不低于它们。仅 MEAN/ADAPTIVE 的权重规则不同；POOLED 与区域组还改变了表示来源，不能视作注意力的单因素比较。

原检测器 fake-logit 差仅保存为 DETECTOR_REFERENCE，不训练残差、不参与选参。诊断标签 `1=fake`。CD2/DFDCP/Wild 是主三域；CD1 单列且与 CD2 不独立。

## 3. 更正残差训练中的基线分数口径

G27 的修正头训练时看到训练内基线分数，验证时看到样本外分数。本轮采用**源域视频分组的嵌套 cross-fitting**：

1. 外层源域 GroupKFold 分出 fit 与 validation。
2. 仅在 fit 视频训练该折 V_BASE 和各输入标准化。
3. fit 内再做 GroupKFold；每个 fit 样本的修正训练基线分数来自未见其视频的内层 V_BASE，形成 OOF 分数。内层 V_BASE 自己的 StandardScaler 也只见内层 fit。
4. 修正头只在外层 fit 上训练；外层 validation 使用第 2 步的 V_BASE。外层 validation 的标签、特征不进入基线、标准化或修正头训练。
5. 按外层 validation BCE 选择超参；固定 epochs，不用目标域或验证集选择最后一个训练 epoch。
6. 最终 V_BASE 在全部源域 train 上拟合；最终修正头用全部源域分组 OOF 基线分数训练，推理使用最终 V_BASE。校准仅用源域 OOF 分数和标签。

这减少了训练内/样本外分数失配，但内层 OOF 基线训练集更小，仍有训练规模差异，不声称完全消除了分布失配。V 特征已经由冻结 checkpoint 学习过，也不等于编码器从未见过源训练视频。

V_BASE 定义及数据不变，但非线性训练协议、种子、网格不同，G28 与 G27 非线性成绩不直接相减归因。先核对本轮 V_BASE；若变化，应查输入/子集而不是调整基线到历史数字。

关联真实视频与派生伪造必须共用分组身份。脚本能核对字符串分组隔离，但无法从字符串证明实际视频关联，运行前须审计 manifest。新增目标集也需满足至少两个视频、二分类标签和无跨 split 视频重叠。

## 4. 实验 A：给 POOLED 补内容控制

对每个源域选定的 POOLED 和 ADAPTIVE，各执行 5 次：

- **DONOR**：保持 V/V_BASE 不变，将 C 和整组 R 替换为同 domain/split、不同视频的 donor。POOLED 实际只读替换后的 C，ADAPTIVE 实际只读替换后的 R。不按标签、姿态或质量挑 donor；没有合法 donor 的样本从逐域比较剔除，主宏平均覆盖不全时标为未计算，不悄悄回退为证据。
- **NOISE_FIXED**：固定真实训练的修正头，C/R 换为源域逐维均值/标准差匹配高斯噪声。
- **NOISE_RETRAIN**：同结构、初始化、预算，用噪声重新训练修正头；使用该真实组的源域选定超参，不重新按目标成绩调节。

控制随机数按 `(seed,domain,split)` 独立生成。增加或删除目标评估行不会改变源训练的 donor/噪声，也不影响源域选参。保存源均值/标准差、生成种子、公式和 donor 行索引；索引对应本次 `scores.npz` 的行位置，原提取身份是 `row_id`。噪声不再保存五份巨型特征数组。

**判读**：

- POOLED 不超过 V_BASE：本协议未建立实际增量，不能因它胜 V_ONLY 就宣布成功。
- POOLED 超过 V_BASE 但不稳定超过噪声重训：没有排除容量/优化解释。
- 真实 POOLED 稳定超过基线及内容控制：支持当前读取协议的真实内容增量；不能证明 CLIP 表征与 V 没有重合，也不能据此给区域命名。
- 仅 donor 导致下降：只支持输入依赖；donor 和高斯替换可能离开自然特征流形，下降并不自动证明条件互补。

全部种子和重复报告，**不要求 15 个区间同时排除零，也不挑其中一次正区间当确认**。噪声重训未独立选参，是固定协议对照，不是噪声模型能力上界。

## 5. 实验 B：定位读取失败，而不是直接换模块

### B1. 相同训练条件比较

固定 λ=[0.01,0.1,1]、weight_decay=[0.001]、200 epochs、lr=0.001、宽度/投影各 128，三个种子 20261020/21/22。

每个条件都训练 V_ONLY/POOLED/MEAN/ADAPTIVE；每个候选/折在构造模型前设置同种子。默认共 36 个全源训练头 + 108 个外层 CV 头 + 30 个噪声重训头，另有基线读取器；这不是低成本推理任务，正式耗时需实测，不能保证比 G27 更快。

报告两套结果：

1. **SELECTED**：每组仅按源域 CV 选参，作为部署候选；
2. **GRID**：相同 λ/decay 下的组间比较，回答选参差异是否足以解释局部/全局差异；只是诊断，不按目标网格选模型。

同 λ 不保证同修正幅度，所以同时保存各域修正 RMS、p50/p90、注意力熵与最大权重。记录每折训练与验证 BCE、基线 BCE、修正量随 epoch 的轨迹，才有材料判断训练过拟合或基线失配。λ=1 如果仍被选中上边界，照实登记，不能自动继续扩网格。

### B2. 冻结模型，只缩放修正

在源域选定的 POOLED/ADAPTIVE 上，计算 `V_BASE + alpha*delta`，alpha 固定为 [0,0.25,0.5,1]，不重训。alpha=0 必须逐元素还原 V_BASE，alpha=1 必须还原候选。

若 ADAPTIVE 缩小修正后稳定改善，支持当前修正在目标域造成干扰的解释；若 POOLED 也有同样曲线，问题可能是残差读出的一般限制。缩小至零回到基线不算 CLIP 正增量。不能按目标 AUC 选 alpha 部署。

### B3. 冻结模型，只改变区域权重

固定同一个已训练 ADAPTIVE 的投影、查询、修正头和 baseline，推理时执行：

- softmax 温度 T=2/4；
- 强制九区域均匀权重。

这与“单独训练一个 MEAN”不同：前者只干预当前模型权重规则，后者可重新适应均匀聚合。保存实际权重与逐样本分数，干预结束恢复模型状态，不能污染后续正常预测和控制。

如果均匀/平滑干预稳定改善，同时训练的 MEAN 在同条件下更好，才更支持当前区域选择策略的迁移限制；如果没有改善，不能归咎于注意力尖锐。权重改变还会改变输入修正头的表示及修正幅度，因此仍不是区域语义或空间定位的因果证明。区域顺序置换对本模型数学上不变，不作为有效破坏控制。

## 6. 统计与确认纪律

主要问题是每个 SELECTED POOLED 相对 V_BASE 的主宏平均增量，以及 SELECTED ADAPTIVE 相对 MEAN 的增量；容量和输入控制组成解释证据链。逐域、所有种子和控制重复均保存；ensemble 只作次级结果，与单模型 AUC 均值分开。

主宏平均按域内抽视频、再平均逐域 AUC；不是 pooled AUC。区间条件于已训练模型，不覆盖重采样源训练集或大量模型筛选的全部不确定性。所有零阈值 rescue/harm 与 AUC 同时报；失败子集选择偏差仍在，不单用失败子集 AUC 下结论。

当前目标数据反复被查看，默认 `--evaluation-status exploratory`。正式确认需独立视频/新数据集和预先登记的比较与统计门槛，不能把探索网格里最好的结果选出来充当确认。

可选 `--evaluation-manifest` 为 CSV，仅必需 `row_id` 列，指定提取目录中的 test 行；必须完整选入这些视频在提取中的所有帧。脚本保留全部源训练行、仅保留指定测试行；拒绝未知/重复/训练行 ID 和不完整的视频选择，复制并哈希该清单。`--evaluation-status new-holdout` 必须提供清单，但标记本身不证明这些视频从未被查看，运行者须确保未与此前报告重叠。全新图像须先另行提取，不能只给脚本一个没有特征的路径。

继承的 checkpoint 有 FFIW 选模历史；当前新读取器只使用源域选参，不等于整个流程为零目标域选模。正式确认应使用源域选模 checkpoint。

## 7. 运行

```powershell
python vit_module/diagnostics/check_clip_readout.py
python vit_module/diagnostics/clip_readout.py `
  --input vit_module/_g25/diag_runs/extract_regions `
  --output vit_module/_g28/clip_readout_run01 `
  --primary-domains cd2 dfdcp wild `
  --seeds 20261020 20261021 20261022 `
  --folds 3 --inner-folds 3 --epochs 200 `
  --lambdas 0.01 0.1 1 --weight-decays 0.001 `
  --gains 0 0.25 0.5 1 --temperatures 2 4 `
  --bootstrap 1000 --repeats 5 --threads 1
```

输出目录必须不存在；输入需完整 `clean_*.npz`，当前仅有 JSON 的 checkout 无法执行真实实验。先合成自检，再在完整实验机跑；不覆盖 G27，不自动运行数小时任务。

## 8. 产物及读表次序

- `scores.npz`：身份/标签及所有 GRID/SELECTED/干预/控制/ensemble 的逐样本分数；统计之前保存。
- `features.npz`：本轮实际使用的 V/C/原始 CLIP 区域与身份。
- `split.json`、各 `*_oof.npz` / `*_baseline.npz` / `*_scalers.npz`：外层/内层行身份、OOF 分数、基线与标准化参数。
- `config.json`：版本、网格、每组源域选择、参数量、校准、代码哈希与确认状态。
- `training_trace.json`：全部源域 CV/最终/噪声训练轨迹。
- `controls.npz`：所有重复 donor 索引、有效掩码、噪声源域统计及种子；`region_weights.npz` 保存真实网格、选中模型、平滑/均匀干预权重。
- `adapter_*.pt` / `noise_*.pt`：全部已训练头；选中头由 config 的 grid_index 定位，无需复制同一权重。
- `summary.json`：per_domain、primary_macro、single_seed_auc、primary_single_seed_auc、score_diagnostics、comparison_roles、primary_questions；`comparisons.json` 标注比较双方与角色。primary_questions 直接列出两项主要问题的比较键；primary_single_seed_auc 明确区分宏平均单模型均值与 ensemble。
- `code_snapshot/`、`input_config.json`、`input_hashes.json`、`artifact_hashes.json`：运行时代码/规格、输入分片及输出校验；输出哈希清单不包含其自身和最后写入的 COMPLETE。

先看 SELECTED POOLED vs V_BASE，随后看该组自身所有内容控制；再看固定 GRID 组间比较和两种冻结干预，最后看域条件性、训练轨迹和救回/误伤。不要倒过来先挑目标域最好的一行。

仓库忽略 `.npz`/`.pt`，这些文件不会随 Git 自动同步；须在实验机另行完整存档并校验。只有所有统计和输出校验成功后生成 COMPLETE；中断目录保留，不支持自动续跑，不应被解释成完整负结果。
