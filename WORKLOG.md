# 新工作区日志

## 2026-10-17：归档旧集成，建立干净工作区

用户要求将未完善的集成文件移入专有文件夹，为重新集成整理本地工作区，同时保留现有架构。

- 旧融合源码、训练/合并/启动脚本、G 系列实验、diagnostics、exp_1、审计产物、论文草稿和历史日志统一归档至 `legacy_integration/archive_20261017/snapshot/`，原相对目录结构保持。
- 归档也复制保留了 `llava/`、`sequence/`、`utils/`、`eval/`、基础源码和整理前其他文件，形成整理前本地状态的完整快照；`.git/` 与 `.claude/` 不在快照内，二者在原位置保持。
- 整理前 **501 个文件、40,035,764 字节**逐个计算 SHA256；归档后逐个校验一致。清单与结果在归档根目录 `inventory.json`、`operations.json`、`verification.json`。
- `vit_module/` 保留改进 ViT 和注意力 shim 的原始字节；三个旧模型文件改为模块转发入口，防止 LLaVA 的现有导入断裂。未把隔离修复偷偷合入旧架构，旧模型行为仍属历史实现。
- Word 文件移至 `documents/`；原有未提交修改保留。整理前已删除的初稿未恢复。所有原文件均有归档副本，无实验或文档被删除。
- 新增 `integration/{models,training,evaluation,configs}/`，没有实现新模型，没有默认历史权重，没有启动训练。
- 仅本地整理；服务器文件、环境、数据和权重不变。后续服务器环境使用 `M2F2_Det`。

整理后核对：501 个原文件的归档哈希一致；64 个保留基础文件字节一致；四份报告字节一致；三个旧模块的转发与模块级修改行为通过隔离校验；保留代码对旧模型的直接导入路径均存在。恢复脚本已实际复制到系统临时新目录，501 个原文件再次校验一致，验证副本保留在临时目录。结果见归档 `workspace_verification.json`。

本地 Python 缺少完整训练依赖，本次没有声称完成完整模型加载或推理；兼容入口校验不替代服务器 `M2F2_Det` 环境的运行验证。没有提交或推送。

历史工作日志见 [旧 WORKLOG](legacy_integration/archive_20261017/snapshot/WORKLOG.md)；研究核查见 [2026-10-16 报告](legacy_integration/archive_20261017/snapshot/reports/RESEARCH_STATUS_20261016.md)。

接下来先在 `integration/` 重建独立基线与可验证接口，再实现融合。归档不是已修复模型，也不包含原本不在本地的服务器权重和数据。

## 2026-10-17：重新集成的研究问题

新增 [integration/RESTART_PLAN.md](integration/RESTART_PLAN.md)，区分“完整 ViT 检测器接入可解释系统”与“CLIP/ViT 检测特征融合”。已有实验不能证明两种表达相同，也不能预设复杂融合必有增量。建议保留 ViT 原生头，先补独立基线和配对错误；CLIP 先承担视觉编码职责，检测增量另行验证。解释任务可研究检测证据与视觉 token 的对齐，但二类检测 token 只是接口基线，不作为定位/忠实性证据。

核对原论文与本地两份 CNN 实现后，记录了结构差异及未注册 Bridge 列表等审计事项，未把当前本地文件直接视为可信的原始基线。方案包含阶段产物、数据隔离、控制和停止规则。本次仅形成研究方案，没有新增融合模型、没有启动训练或服务器实验。

## 2026-10-17：第一步 ViT-only 接入实现

用户决定先接入 ViT，由用户在服务器测试运行，再逐步研究融合。本次仅修改本地，不上传或运行服务器。

- 新增 `integration/models/vit_native.py`：完整 ViT-B/16（91,388,690 参数，164 个 state key），保留原生二类头。严格加载 PDI `model.*`/纯原生 state，拒绝缺头、旧融合、多余键、错误形状、非有限张量与配置哈希不符；未验证权重时拒绝前向。
- 原生检测器始终冻结并保持 eval；独立 PIL RGB 输入管线，不构造检测用 CLIP、文本编码器、Bridge 或随机替代头。
- `llava_llama.py` 新增显式 `deepfake_model_name=vit_native` 分支，旧分支延迟导入并保留非 native 配置原行为。原生 logits 顺序保持，语言 token 明确采用 `[fake,real]`。支持无 CLIP 的检测 token 路径；完整系统视觉 CLIP 只向语言模块提供视觉 token。
- 新增原生配置、真实 checkpoint smoke、可选完整 7B 运行脚本与 `integration/SERVER_TEST.md`。先加载 LLaVA，再加载原生头，避免 HF 初始化覆盖原生权重。旧融合 LoRA 不作为新原生模型的对齐权重。
- 本地 8 项合成检查通过，包含真实 LLaVA 类的 token/语言前向、视觉输入变化不影响检测、冻结/梯度、HF 保存重载等。
- 本地还以完整 ViT 拓扑生成合成 PDI 格式 checkpoint 与两张随机图片，端到端运行 smoke CLI，164 键完整加载；裸 ViT/包装/接入 LLaVA 的同批 logits 差均为 0，token `[2,1,64]`、语言前向 `[2,3,64]`，约 23.125 秒。产物在 `integration/runs/local_fullshape_fixture_1792239086058548900/`，明确 `input_is_test_fixture=true`，临时合成权重已清理，不作性能证据。
- 真实 PDI 预处理和原生类别顺序尚需核对；配置中的两个 verified 字段保持 false，当前脚本不算 AUC、不宣称复现 acc.txt。完整 7B/GPU 路径尚未实测，留待用户在 M2F2_Det 环境验证。

当前没有训练、没有融合增量结果、没有服务器修改或 Git 提交/推送。

服务器手动同步包已生成：`integration/outputs/vit_native_server_bundle_20261017.zip`，41,270 字节、14 个成员；全部成员解压字节与源文件一致，ZIP 校验通过。包内仅源码、说明、配置、原 ViT 参考文件和 SHA256 清单，不含权重、图片、实验结果。包 SHA256：`2f4fef67b979b7c2a9a20f863d59721302ef218e285b9b989d79dea7e37350b4`。另通过 Python 3.10 语法解析及两个运行入口的 `--help` 检查。
