# G30：改进 ViT 基线上的融合增量验证

## 目的

验证问题不是“CLIP 是否有信息”，而是：在**同一份改进 ViT 权重、同一批图像、同一标签和同一源域选参规则**下，加入 CLIP 或有效 Bridge 是否产生相对于 ViT 的稳定增量。

`acc.txt` 只作为独立改进 ViT 的历史结果参考，不能与本实验目标域 AUC 直接相减。正式比较必须保存逐样本分数，并使用同一评测行做视频分组配对 bootstrap。

## 四组对照

| 组 | 输入 | 作用 |
|---|---|---|
| `V_NATIVE` | 改进 ViT 原生分类头分数（若 checkpoint 提供） | 真实工程基线；不能用汇总日志代替 |
| `V_MATCHED` | 同一 ViT CLS，源域重新训练的配平分类头 | 排除分类头/尺度差异 |
| `V_CLIP` | 同一 ViT CLS + CLIP image CLS，全局投影融合 | 测量全局 CLIP 的增量 |
| `V_BRIDGE` | 同一 ViT + CLIP patch tokens，经修复且重新训练的有效 Bridge | 测量多层交互是否超过简单融合 |

主比较顺序固定为：`V_MATCHED → V_CLIP → V_BRIDGE`；`V_NATIVE` 作为工程锚点单列。不得用目标域结果挑选权重、alpha、正则或阈值。

## 关键口径

1. `V` 必须来自与 `acc.txt` 相同的改进 ViT 权重；不能把原 Bridge checkpoint 的 ViT CLS 探针称为原生 ViT 成绩。
2. `C` 是同一图像的 CLIP image CLS，不使用文本 token；文本分支不进入本轮。
3. `B` 必须来自**修复后的 Bridge**，修复 `LayerNorm(1)` 后重新训练。原 checkpoint 的 128 维 Bridge 是常数，不能作为 `V_BRIDGE`。
4. 四组共用同一视频级 source-only folds、三个随机种子和正则网格；目标域只做一次最终评估。
5. 所有组保存 `row_id/path/domain/split/video_id/y` 与逐样本分数，保证配对比较和 rescue/harm 分析。

## 判定逻辑

- `V_CLIP > V_MATCHED` 且配对区间排除 0：支持全局 CLIP 在改进 ViT 之上的增量。
- `V_CLIP` 不超过 `V_MATCHED`：不能声称全局 CLIP 有稳定增量；不要直接转向更复杂 Bridge。
- `V_BRIDGE > V_CLIP`：支持多层交互的额外价值。
- `V_BRIDGE ≈ V_CLIP`：Bridge 没有被证明必要，简单融合更合适。
- `V_BRIDGE` 只超过 `V_NATIVE` 而不超过 `V_MATCHED`：收益可能只是分类头差异。
- 真实组提升但 donor/noise 重训也提升：不能归因于当前图像的 CLIP 内容。

同时报告 ViT 错误集合上的 rescue、harm、flip rate 和逐域 AUC。平均 AUC 提升但 harm 同步增加时，优先研究“受约束修正”，而不是继续扩大 Bridge 容量。

## 数据准备

离线脚本要求 clean 分片包含 `V`、`C`、身份字段；正式 `V_BRIDGE` 还必须包含由修复 Bridge 重新训练/提取的 `B`。如果只有 G25 的 `clean_*.npz`，只能运行 `V_MATCHED` 与 `V_CLIP`，不能伪造 `V_BRIDGE`。

原生 ViT 头分数通过 `--native-score-npz` 提供，数组必须与 `row_id` 逐行一致。只有 `acc.txt` 而没有逐样本 native score 时，`V_NATIVE` 只能在报告中作为历史参考。

## 运行

```powershell
python vit_module/diagnostics/check_incremental_fusion.py
python vit_module/diagnostics/incremental_fusion.py `
  --input vit_module/_g30/features `
  --output vit_module/_g30/incremental_run01 `
  --primary-domains cd2 dfdcp wild `
  --seeds 20261040 20261041 20261042 `
  --folds 3 --epochs 200 --lr 0.001 `
  --weight-decays 0.0001 0.001 0.01 --bootstrap 1000
```

输出目录必须不存在。当前 checkout 没有服务器上的 clean 特征、native 逐样本分数或有效 Bridge 特征，因此这里只完成规格和离线统计器，不能宣称真实结果。
