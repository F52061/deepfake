# Deepfake：重新集成工作区

2026-10-17 已将上一轮融合实现、启动脚本、实验产物与研究日志移入专用归档。下一轮集成从 [integration/](integration/README.md) 开始；当前尚未实现新融合模型，也没有启动训练。

目前已完成第一步 **ViT-only 接入**，保留原生分类头并接通 LLaVA 检测 token；本地检查已通过，服务器真实权重/整模型运行待用户测试。命令见 [服务器验证说明](integration/SERVER_TEST.md)。

| 位置 | 当前用途 |
|---|---|
| [integration/](integration/README.md) | 干净的新集成目录，包含模型、训练、评测和配置位置 |
| [vit_module/](vit_module/README.md) | 原改进 ViT 基础实现、注意力兼容实现、三个旧接口转发文件 |
| `llava/`、`sequence/`、`utils/`、`eval/`、`asset/` | 保持原样的现有结构；保留不等于已验证无缺陷 |
| `stage_1_detection*.py`、`stage_1_*.sh` | 保留的原检测入口，不代表新集成已经可运行 |
| `scripts/` | 保留原有 ZeRO 配置，旧合并与微调脚本已归档 |
| [documents/](documents/README.md) | 原根目录 Word 报告，内容原样保留 |
| [legacy_integration/archive_20261017/](legacy_integration/archive_20261017/README.md) | 上一轮完整本地快照、移动清单、文件哈希与恢复脚本 |
| [WORKLOG.md](WORKLOG.md) | 本轮整理记录与后续工作入口 |

旧融合模型的三个兼容入口仅保证原导入路径可用，仍指向有已知缺陷的历史实现。新集成目录不依赖它们。旧实验与训练请在归档说明所述的独立恢复目录中复现，避免与新工作混用。

先完成原生 ViT、CLIP-only 的独立基线和配对评估，再开展融合；训练与验证的数据划分、标签方向、预处理、checkpoint 来源须明确。后续服务器环境固定为 `M2F2_Det`，本次未操作服务器。

原项目介绍、引用信息和历史使用方法保存在 [原 README](legacy_integration/archive_20261017/snapshot/README.md)，许可证保留在 [LICENSE](LICENSE)。
