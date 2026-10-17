# CLIP image 局部/结构互补实验

## 目的

前一轮只证明全局 CLIP image 的增量有限。本实验检验更具体的假设：CLIP 的补充可能存在于空间局部和区域关系中，而不是 pooled CLS。

实验不会把固定网格直接命名为眼睛、嘴或边界；网格只是空间位置。只有后续定位实验结合人脸关键点后，才能给出器官语义。

## 提取

在原有提取命令增加 --save-regions，保存每张图的 ViT 最后一层 14x14 patch、CLIP 最后一层 24x24 patch，以及两者在各自网格内池化到 3x3 的九个相对位置区域。

当前代码不把 24x24 强行插值成 14x14，因此不制造假一一对应关系。两边用同样的 3x3 空间分区，可比较相同相对位置。

## 读取器

仅用 FF++ train 拟合 StandardScaler + LogisticRegression(C=1e-3)，目标域标签只在最后计算：

1. V、C、[V,C]：全局基线。
2. V_regions、C_regions、[V_regions,C_regions]：九区域局部信息。
3. V_relations、C_relations、[V,C,V_relations,C_relations]：区域间 cosine 和区域到全局 cosine。

所有局部 token 在进入读取器前单独 L2 归一化，避免 CLIP/ViT 原始尺度主导结果。此实验是表示可读性诊断，不等于模型内部融合。

## 空间对应控制

每个重复在同一 domain/split 内打乱 CLIP 九区域的样本来源。该控制保留局部特征分布，破坏图像对应关系；读取器固定，不重新拟合。另加入同维度均值/标准差匹配随机区域。

如果正常 [V_regions,C_regions] 明显优于全局 [V,C]、局部 ViT、空间打乱和随机区域，并且增量主要出现在 ViT 错误样本上，才支持 CLIP 局部空间信息具有条件互补性。

如果空间打乱后相近，收益来自局部特征整体统计而非空间关系。如果局部特征本身不如全局，但关系特征有效，说明可能是区域间结构关系。如果只有单一目标域增益，结论应写成域条件现象。

## 统计与保存

逐域报告 AUC、相对 ViT 和 [V,C] 的增量、视频分组 bootstrap 区间、原始 V 错误集合的 rescue/harm。FFIW 单视频不纳入主要聚合。

输出保存原始九区域、关系描述、读取器参数、置换区域和随机区域，支持复算统计。

运行：

    python vit_module/diagnostics/extract.py --manifest vit_module/_g23/manifest_full.csv --checkpoint checkpoints/stage_1/bridge_v2_phase1.pth --clip checkpoints/clip-vit-large-patch14-336 --device cuda:0 --batch-size 4 --fake-logit 0 --save-regions --variants clean --output vit_module/_g25/diag_runs/extract_regions
    python vit_module/diagnostics/local_analyze.py --input vit_module/_g25/diag_runs/extract_regions --output vit_module/_g25/local_run01 --bootstrap 1000

运行前确保输出目录不存在。--save-regions 会增加存储量；正式运行前先用 smoke manifest 检查形状和设备。
