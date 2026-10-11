# G29-A：donor 标签控制（事前预注册）

本文件是**运行前**写定的设计书。规格中所有种子、口径、判定阈值在见到结果前固定；运行后不得回改，只能在结果记录里追加。

## 0. 运行前修订（2026-10-11，见到任何结果之前）

本实验**尚未运行**。运行时用到的视频身份口径在原稿里写错了，此处按"运行前可修订、运行后不得回改"的规矩修正，并逐条说明**改了什么、没改什么**。

### 0.1 发现：`video_id` 列不是视频身份

原稿 §3 假定"同一 `video_id` 只有一个标签，所以异类必然异视频"。用 `_g29/vector_fusion_run01/features.npz` 的 `features.npz` 直接核对，该假定**为假**：

| 数据集 | `video_id` 实际是什么 | 证据（同一 `video_id` 下的真实路径） |
|---|---|---|
| cd1 / cd2 | **人物 id** | `id1` 同时对应 `Celeb-DF-v2/faces23/id1_0007/…`（y=0）与 `Celeb-DF-v2/faces23/id1_id0_0007/…`（y=1） |
| dfdcp | **人物 id** | `1290777` 同时对应 `original_videos/1290777_A_002/…`（y=0）与 `method_A/1290777_2060058_A_002/…`（y=1） |
| ffpp | **人物 id** | `003` 同时对应 `original_sequences/c23/faces23/003/…`（y=0）与 `…/Deepfakes/c23/faces23/003_000/…`（y=1） |
| wild | **帧号** | `0` 同时对应 `real_test/27/real/…`（y=0）与 `fake_test/33/fake/…`（y=1） |

按 `(domain, split)` 单元统计：cd1 8/41、cd2 49/124、dfdcp 18/47、ffpp 42/42 个 `video_id` 携带两个标签；wild 一个"id"跨不同视频。因此 `video_id` 既不能保证"异视频"，在 wild 上更**不保证**（同一视频的不同帧号算不同 id）。这与项目一贯口径一致：视频/重叠身份必须由**路径派生**，不能用裸 `video_id` 字符串。

### 0.2 改了什么

- **视频身份改为路径派生**：取图片所在目录，去掉抽取标记目录 `faces23`；若该目录位于 `real`/`fake` 之下（wild 的排布），再去掉最后一层帧目录。即：
  - ffpp `…/manipulated_sequences/Deepfakes/c23/faces23/114_102/167.png` → `…/Deepfakes/c23/114_102`
  - cd2 `…/Celeb-DF-v2/faces23/id10_0007/461.png` → `…/Celeb-DF-v2/id10_0007`
  - dfdcp `…/DFDC-P/test_set_23/original_videos/2005778_C_003/175.png` → `…/original_videos/2005778_C_003`
  - wild `…/WildDeepfake/real_test/79/real/180/3525.png` → `…/WildDeepfake/real_test/79/real`
- **断言改为作用于该派生身份**（"同一视频只有一个标签"）。实测在真实数据上**成立**：五个单元 `keys_with_both_labels = 0`；视频数 cd1 49、cd2 173、dfdcp 70、wild 96、ffpp test 199 / train 471。
- 抽取时**同时**要求"异标签"与"异视频"（CROSS 不再依赖"异标签⇒异视频"这条不成立的推理）。
- 新增运行期断言：任何 donor 不得与所在行同视频。

### 0.3 没改什么

种子与偏移（`seed + 400000 + repeat`、`cell_seed(seed, kind, f"{domain}|{split}")`）、两组控制的定义（SAME/CROSS）、统计口径（test 行、逐域 AUC 算术平均、视频块配对 bootstrap 1000、`delta = 真实 − 控制`）、评审表（§5 的 9/9 与 ±0.005 门槛）、重复次数（3 次，见 §4）。**没有任何与结果相关的参数因这次修订而改变**——修订只修正"谁算一个视频"。

### 0.4 连带登记：G29 自身 donor 控制的 wild 暴露度

G29 的 `DONOR`（`clip_controls` → `split_donors`）用的正是 `video_id`。按上面的口径复核其候选池：全部单元里，旧口径候选行中被抽到的行有 **0.05%** 落在该行**自己的视频**上；唯一有暴露的是 wild/test 的 SAME 池（2.97%），其余五个单元为 0.00%。所以 G29 的 donor 控制在这条口径上**基本成立**，wild 上有约 3% 的稀释（会把真实内容效应略微拉向零），不改变 §D.78 的更正结论。该暴露度在本实验里为 0（见 §0.2 的运行期断言）。

## 1. 为什么做

G29 的 `DONOR` 控制（`vector_fusion.py` 的 `clip_controls`）把一行的 `C` 换成**同一 `(domain, split)` 单元内、不同视频**的 `C`。它**不约束标签**：实测 donor 与所在行的标签一致率仅 **0.5043**（G28 为 0.496），即随机水平。

2026-10-11 的复核（`WORKLOG.md` §D.78、`FINDINGS_问题验证数据.md` P17.4）把该控制的 15 组结果按 donor 标签事后分层，得到：同标签 donor 无损（+0.0094/+0.0092），全部损失来自异标签 donor（−0.2454/−0.2294）。因此原控制**只能**证明"`C` 的类别级信息被读取"，**不能**证明"当前图像的内容被使用"。

**事后分层有两个弱点**：(a) 评测集缩小到约一半（n≈450），只能作诊断；(b) 分层是按结果选的，不属预注册。本实验把这件事做成**事前、全样本**的两组控制，直接回答"模型是否用到超出类别级的图像特有内容"。

## 2. 输入（只读，不训练）

只用 G29 已跑完的产物（`vit_module/_g29/vector_fusion_run01/`）：

| 文件 | 用途 |
|---|---|
| `features.npz` | 原始 `V`(768) / `C`(1024) 与身份列 |
| `scalers.npz` | 全源 `StandardScaler` 参数（本实验会从 `features.npz` + 训练行**重算**并与它逐位比对） |
| `model_{FIXED,LEARNED}_{seed}.pt` | 12 个**已训练**的全源模型权重 |
| `scores.npz` | 参考列（用于校验本实验复算的"真实组"分数逐位一致） |
| `controls.npz` | G29 自己的 donor 行号（仅用于 §6 的分解核对，不参与本实验的抽取） |
| `artifact_hashes.json` | 上述每个文件的 sha256 校验 |

**不训练任何模型**：本实验是"固定已训练模型、只换输入"的干预，与 G29 的 `DONOR` / `NOISE_FIXED` 同型。因此它**不消耗**训练预算，也不回答任何关于容量或优化的问题。

## 3. 两组控制

对每个 `(domain, split)` 单元（`domain` 含 `cd1/cd2/dfdcp/ffpp/wild`，`split` 含 `train/test`）：

- **SAME（主问题组）**：把一行的 `C` 换成**同单元内、同类（`y` 相同）、异视频**的另一行的 `C`。
- **CROSS（正对照组）**：换成**同单元内、异类（`y` 不同）、异视频**的另一行的 `C`。

"异视频"按 §0 修订后的**路径派生视频身份**判定（原稿此处写的"异类必然异视频"不成立，已作废）；抽取时同时要求两个条件，并在运行期断言 donor 与所在行不同视频。

抽取规则：

- 按行所属的 `(video_id, y)` 分组，对每组从候选集**有放回**均匀抽取，组内每行得到一个 donor 行号。
- 候选集为空（该组没有符合条件的行）时记 `-1`，**记录在案而非静默回退**；比较时按 `usable` 掩码剔除，并报告剔除行数。
- 随机种子：`default_rng(cell_seed(seed + 400000 + repeat, kind, f"{domain}|{split}"))`，`kind ∈ {SAME, CROSS}`，`repeat ∈ {0,1,2}`。
  - 该偏移与 G29 自身 donor 的 `seed + 200000 + repeat`（其内部噪声再 `+100000`）**正交**，两组抽取互不重叠。
  - 种子只依赖 `(seed, kind, domain, split)`，**不依赖目标域行的增减**（沿用 G29 的噪声/ donor 纪律）。
- `V` 始终是**该行自己的 `V`**，只换 `C`；标准化用与 G29 相同的全源 `StandardScaler`。

## 4. 统计口径（与 G29 完全一致）

- 评测行：`split == "test"`（ffpp 只取 test 800 行，train 2200 行不参与评测）。
- 逐域 AUC（`roc_auc_score`），主宏平均 = `cd2/dfdcp/wild` 三域 AUC 的**算术平均**（不是 pooled AUC）。
- 配对区间：视频块重采样 1000 次，取 [0.025, 0.975] 分位。
- `delta = 真实组 − 控制组`（**正 = 真实组更好 = 控制组丢了信息**）。
- 重复 3 次（本控制不训练，重复只用于平均 donor 抽取的不确定性；取 3 次以控制纯 CPU 统计耗时，同时记录跨重复的点估计离散度）。**规格偏离**：G29 用 5 次，本实验用 3 次，理由如上，运行前固定。

## 5. 判定表（**运行前固定**）

| 问题 | 判据 | 结论 |
|---|---|---|
| **Q-A 主问题**：同标签异视频 donor 是否降低主宏？ | 三 seed × 三重复共 **9** 个 CI95 | 9/9 排除 0 且 delta 均值 ≥ **+0.005** → 支持"模型用到超出类别级的图像特有内容"，**推翻 §D.78 的更正**；9/9 含 0 → "类别级"读法成立；其余 → 未定论，如实报告 |
| **Q-B 正对照**：异标签 donor 是否显著降低主宏？ | 同上 9 个 CI | 9/9 排除 0 且 delta 均值 ≫ 0 → 控制实现有效；**若不成立，本轮整体作废**（说明抽取或评分链有问题） |
| **Q-C 分解**：G29 记录的 `DONOR` 是否 ≈ `(SAME + CROSS)/2`？ | 对每个 `(variant, seed, repeat)` 比较三点估计 | 若近似（差 ≤ 0.02）→ 证明 §D.78 的事后分层确实只是"标签错配占一半"的算术结果；若有系统偏差 → 记录偏差 |

阈值 +0.005 的选择：G29 主宏的比较量级在 0.01–0.03 之间，+0.005 是"小于该量级的一半"的保守门槛；同时要求 9/9 全部排除 0，避免单点显著。

## 6. 禁止事项

- **不得**按目标域结果回改 donor 口径、种子偏移或阈值。
- **不得**把 `delta ≠ 0` 单独归因于"内容"：donor **未匹配姿态、质量、时长**（沿用 G29 的 donor 口径），这些差异与内容混淆。
- **不得**用本实验谈容量、优化或"参数多少"——本轮不涉及训练。
- **不得**把 SAME 的不显著读成"两者等价"：不显著只表示未检出差异，且 CROSS 组的结果限定了本实验的检出灵敏度下限。
- **不得**覆盖 `_g29/vector_fusion_run01/`：本实验写入独立目录 `_g29/donor_label_run01/`。

## 7. 运行

```powershell
python vit_module/diagnostics/check_donor_label.py
python vit_module/diagnostics/donor_label_control.py `
  --reference vit_module/_g29/vector_fusion_run01 `
  --output vit_module/_g29/donor_label_run01 `
  --primary-domains cd2 dfdcp wild `
  --seeds 20261030 20261031 20261032 --repeats 3 --bootstrap 1000 --threads 1
```

先跑 `check_donor_label.py`（合成数据自检：同类/异类关系、异视频、同单元、确定性、`-1` 记录、零替换恒等），通过后才运行正式脚本。

## 8. 产物与读取顺序

`summary.json`（`primary_macro` → `per_domain` → `decomposition`）→ `controls.npz`（`donors|kind|seed|repeat`、`usable|...`、`label_match|...`）→ `scores.npz`（逐样本分数）→ `config.json`（输入哈希与设计参数）→ `artifact_hashes.json`。

`code_snapshot/` 保存运行时的本脚本、本规格与所依赖模块（`vector_fusion.py`、`analyze.py`、`clip_readout.py`、`pure_vit_clip.py`、`conditional_clip.py`），用于复现比对。
