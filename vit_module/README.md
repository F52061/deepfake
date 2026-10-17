# 保留的 ViT 基础代码与旧接口

- `vit_adaptive_mattn_aps.py`：原有改进 ViT 基础实现，字节保持不变；其正确性和预处理仍须在新集成中独立核对。
- `flash_attn_shim/`：原有注意力兼容实现，字节保持不变。
- 三个 `vit_m2f2_detector*.py`：仅转发到归档实现，维持现有 LLaVA 的导入路径与模块级 monkeypatch 行为；它们不是新模型，也不包含已经验证的三处修复。

旧完整实现、训练入口、实验与日志见 [专用归档](../legacy_integration/archive_20261017/README.md)。
重新集成只在 [integration/](../integration/README.md) 开发。不得通过在此处复制旧融合代码来开始新版本。
