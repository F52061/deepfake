# 旧集成专用归档

归档日期：2026-10-17。它保存重新集成前的本地状态，不作为当前模型或训练入口。

## 内容

- `snapshot/vit_module/`：旧三个融合模型、训练/推理/合并脚本、全部实验、diagnostics、exp_1 和审计数据；保留原相对目录。
- `snapshot/llava/`、`sequence/`、`utils/`、`eval/`、`asset/` 等：未改动基础代码的副本，供恢复历史上下文。
- `snapshot/scripts/` 及根目录的旧启动文件：旧微调与合并入口。
- `snapshot/WORKLOG.md`、`FINDINGS_问题验证数据.md`、`describe.md`、`acc.txt`、`paper/`、`reports/`：原始日志、设计、实验总结与论文/报告材料。
- `snapshot/*.docx`：原报告的精确副本；用户日常文档移到了工作区 `documents/`。
- `snapshot/README.md`、环境文件、LICENSE、pyproject 和原 `.gitignore`：原项目说明与配置。

不包含 `.git/`、`.claude/`，二者在工作区原位置保留。原本未在本地的服务器 dataset、checkpoints、环境不在此归档。历史脚本中的服务器绝对路径和某些已缺失产物路径仍然按原样保留，不能承诺所有旧实验可直接重跑。

## 完整性与工作区边界

`inventory.json` 记录整理前 501 个文件的原路径、字节数和 SHA256；`verification.json` 确认全部归档副本一致；`operations.json` 区分实际移动与保留文件的复制。原 Git HEAD 与未提交状态分别保存在 `git_head_before.txt`、`git_status_before.txt`。用户未提交修改已包含在快照中，整理前已删除的文件未恢复。

`workspace_verification.json` 还记录保留文件字节核对、兼容入口转发校验与一次实际恢复测试。恢复至系统临时新目录后，501 个原文件校验一致，临时恢复副本保留。转发校验使用隔离模块验证，不构建模型；本地训练依赖不全，未声称已完成模型推理。

该目录是当前代码保留旧接口所需的本地依赖，不可直接删除或改名。旧 `vit_module.vit_m2f2_detector*` 入口只把模块转发到本归档，因此现有 LLaVA 无需改代码；模块级 monkeypatch 仍作用于实际实现。新 `integration/` 不导入这些旧接口。

归档中的代码和文本可由 Git 跟踪；权重、特征、图片等继续按原忽略规则留在磁盘。忽略文件不会因为归档就进入 Git。归档在同一工作盘上，提供整理回退，不提供磁盘故障保护。本次没有提交或推送。

## 恢复历史工作区

在当前工作区根目录执行下列脚本，将完整快照复制到一个**新的、不存在的**目录；不会覆盖当前新工作区：

```powershell
& ./legacy_integration/archive_20261017/restore.ps1 -Destination 'E:/明确选择的历史恢复目录'
```

脚本先校验归档 SHA256，再复制并校验目标。恢复后从恢复目录作为项目根运行历史代码，使用所需的依赖、外部数据和权重。`organize_workspace.ps1` 仅记录本次整理实现，不能当作反复执行的清理工具。
