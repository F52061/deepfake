# 纯 PyTorch 实现的 flash_attn 兼容 shim。
# 背景: M2F2_Det conda 环境被删除后重建, 找不到 cu118+torch2.2+Windows 的
# flash_attn 预编译轮子 (woct0rdho/flash-attn-prebuilt 仓库已删除)。
# 本项目代码只用到 flash_attn.modules.mha.MHA, 且以默认 use_flash_attn=False
# 调用 (MHA 构造未传 use_flash_attn=True), 走的是纯 PyTorch 的非 flash 路径
# (SelfAttention + einsum), 不需要任何 CUDA 融合 kernel, 因此可用等价的
# PyTorch 实现替代, 数学与 flash_attn 2.5.9 非 flash 路径逐字一致, 支持 Pascal。
