# 新集成工作区

此目录是重新实现集成的唯一开发入口。当前已实现显式 `vit_native` 接入：完整原生 ViT 检测器 → 检测 token → LLaVA。检测器不组合 CLIP 或 Bridge，尚未启动训练。

服务器运行顺序与命令见 [SERVER_TEST.md](SERVER_TEST.md)。配置 `configs/vit_native.json` 指向此前核对过哈希的服务器原生权重；预处理与原生类别顺序仍须按 PDI 训练配置确认。

重新开始的职责划分、最小实验和停止条件见 [RESTART_PLAN.md](RESTART_PLAN.md)。先做完整检测器接入和独立互补性检查，再根据证据决定是否进行检测特征融合。

| 目录 | 后续用途 |
|---|---|
| `models/` | 独立分支接口与新的融合实现 |
| `training/` | 源域训练、独立验证及 checkpoint 管理 |
| `evaluation/` | 同图配对预测、分支互补与融合增量评估 |
| `configs/` | 经确认的模型、数据与实验配置 |

先复现完整原生 ViT 检测函数和独立 CLIP-only 读取器，再验证简单融合；有支持时才评估正确 Bridge。此目录不导入 `legacy_integration` 或旧 `vit_m2f2_detector*`，不默认加载历史融合权重，不把旧实验的 COMPLETE 或汇总文件作为新实验成功证据。

可复用的基础代码在 `vit_module/vit_adaptive_mattn_aps.py`、`vit_module/flash_attn_shim/`；现有 `llava/`、`sequence/`、`utils/` 等保持原样。这些是现存基础，不代表已经完成独立验证。

服务器后续使用原项目 `M2F2_Det`：`C:/Users/Supor2/.conda/envs/M2F2_Det/python.exe`。本次只整理本地，服务器目录与权重未移动。当前本地原本就没有 `dataset/`、`checkpoints/`，不能把归档视为服务器数据和权重的备份。

历史资料统一见 [归档说明](../legacy_integration/archive_20261017/README.md)。新结果使用本目录下的新 `runs/` 或 `outputs/`，并单独保存来源与配置；二者已排除出 Git。
