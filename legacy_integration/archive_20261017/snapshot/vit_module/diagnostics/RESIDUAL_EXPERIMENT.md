# CLIP image 残差实验：开设详述

## 研究问题与纠正

V 是 ViT 末层图像 CLS，C 是 CLIP image CLS；本实验不使用 CLIP text。
研究问题：源域训练的线性读取器是否能从 CLIP 中读出对 V-only 的跨域增量，以及残差化是否改变这种读出。

设源域拟合 Cpred=AV+b，Cres=C-Cpred。给定 V 和 Cres 可以完整恢复 C：
C=Cres+AV+b。因此 [V,C] 与 [V,Cres] 信息完全相同，线性分类器假设空间也相同。
先前将“残差融合不提升”解释为“没有独立信息”，或将“提升”解释为条件互信息大于零，均不严谨。
线性 Ridge 残差只表示未被该预测器充分解释的部分，包含非线性共享信息、拟合误差和噪声；不是纯独立信息。
本实验的科学结论限定于源域训练读取器和冻结表示。

## 固定协议

仅 FF++ train 拟合预处理、Ridge 和真假读取器。目标域标签仅用于最终统计。
Ridge alpha=[0.1,1,10,100,1000]，源域最多五折 GroupKFold 按 video_id 切分，以预测 C 的均方误差选择 alpha。
每折独立拟合 V 的 StandardScaler，不使用验证折统计；选定 alpha 后在全部源训练集重拟合。
真假读取器统一 StandardScaler+LogisticRegression(C=0.001)，不根据目标域选择超参数。
主结果按域分别报告；不将 CD1、CD2 当作独立重复证据，不默认合并平均。
FFIW 单视频不满足置换与视频区间条件，必须排除或重新采样；不要用帧号伪造 video_id。

## 条件与各自用途

1. V-only：固定主要基线，定义失败子集及救回/误伤。
2. C-only：CLIP image 自身可读性。
3. Cres-only：该源域线性残差的可读性，不能称为独立信息量。
4. [V,C]：原始线性融合基线。
5. [V,Cres]：相同信息的重新参数化，用同样正则化训练新读取器。
6. 等价分类头：把已训练的 [V,C] 头精确转换到 [V,Cres] 空间，不重新训练，要求逐样本分数误差<1e-7。
7. 评估时跨视频置换 Cres：保持读取器不变，破坏样本对应。
8. 置换后重训读取器：容量对照。
9. 源域残差均值/方差匹配的 Gaussian 特征：同维数噪声对照。

使用五个种子生成置换/噪声，不使用标签选择 donor，且不跨域、不跨 split、不取同视频。
等价头检验是必要控制：[V,Cres] 重训收益若存在，必须解释为读出/正则化变化，而非创造了信息。
残差化在完整源 train 上拟合，因此读取器训练使用的是 in-sample 残差；它可能比目标域残差小，属于跨域失配因素，需在解释中保留。

## 预先指定的统计和判定

逐域 AUC、相对 V-only 的 AUC 差、相对 [V,C] 的差，视频分组配对 bootstrap 1000 次。
V-only 读取器零分数阈值定义错误集合：报告 rescue、harm、净救回。
错误集合 AUC 只作描述，存在选择偏差；不可用该集合调模型。
主要判据：[V,Cres] 相对 V 的配对 AUC 区间、净救回及与置换/噪声对照的跨种子一致性。
多域/多控制比较是探索性结果，不挑一个正区间就声称整体显著；确认实验应使用新的视频留出集。
不显著不等于等价。负结果描述为“未检测到稳定增量”，不要描述为“信息不存在”。
若 [V,Cres] 优于 [V,C]：后续研究读出和正则化；若两者相近：保留简单融合；若 Cres-only 强但联合无收益：可能重复或源域训练偏置，仍需进一步干预。
该实验不能定位局部区域，也不能直接支持局部门控架构。

## 运行与数据保留

优先读已有提取分片，不重新前向：

```powershell
python vit_module/diagnostics/residual_clip.py --input vit_module/_g23/diag_runs/extract_full --label-one fake --output vit_module/_g24/residual_run01
```

也支持一个 NPZ，要求 V/C/y/path/domain/split/video_id/row_id，row_id=0..n-1。
历史 npz 字段名、split 和标签约定不同，不能直接传入；必须转换并审计，--label-one real 会显式翻转标签。
源域 video_id 应将关联真实/派生伪造视频纳入同一组，现成 vids 是否满足这点需核查。

输出目录不得存在。features.npz 保存 V/C/Cres/Cpred 和样本身份；ridge.npz 保存预测参数；reader_*.npz 保存所有读取器；scores.npz 保存逐样本分数；controls.npz 保存 donor 和噪声；config.json 保存参数/校验；summary.json 保存统计。
逐样本分数在 bootstrap 前落盘，统计中断不会丢失核心数据。COMPLETE 标记表示统计也完成。

## 本地执行状态

当前 checkout 的 extract_full 仅有 config.json 和 COMPLETE，没有 clean_*.npz；
analysis_full 仅有 config.json 和 summary.json，没有逐样本特征。
文档中的 E:/Cross-domain_authentication_verification/... 原始工程路径当前不可用。
因此本地汇总足以复核原来的 V 与 [V,C]，但不能拟合新的 Ridge 或计算真实 Cres。
真实实验必须以实际可读的逐样本特征为输入；不会根据汇总 AUC 伪造残差结果。
