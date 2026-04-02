# 技术路线：基于3D窗口注意力熵的残差感知混合精度量化 (Video Swin PTQ 范式)

本技术路线针对视频动作识别中量化误差传播问题，提出一种适配 Video Swin Transformer 层次化与 3D 局部窗口特性的**训练后量化（Post-Training Quantization, PTQ）**解决方案。

---

## 第一阶段：层次化特征定义与 3D 窗口分组 (Hierarchical Sequence & 3D Window Grouping)

**目标**：适配 Video Swin 的多阶段 (Multi-Stage) 特征金字塔，以 3D 窗口为基本单元进行局部特征提取。

1. **层次化特征处理**：
   不存在 CLS Token。输入特征激活值张量根据当前网络阶段 (Stage $i$) 动态变化，形状为 $X_{in} \in \mathbb{R}^{B \times C_i \times T_i \times H_i \times W_i}$。

2. **3D 窗口划分 (3D Window Grouping)**：
   将全局特征按照时间窗口大小 $P$ 和空间窗口大小 $M \times M$ 划分为多个不重叠的局部 3D 窗口。
   每个窗口特征记为 $X_{win} \in \mathbb{R}^{N_{win} \times (P \cdot M \cdot M) \times C_i}$（其中 $N_{win}$ 为窗口总数）。
   量化敏感度的统计将以此 3D 窗口为最小粒度进行。

---

## 第二阶段：校准阶段 (Calibration) —— 阶段敏感分布统计与阈值标定

**目标**：利用少量校准数据集，提取组级/窗口级度量指标，**按 Stage 拟合**归一化统计区间与决策阈值 $\tau$。

### 1. 窗口内残差幅度 ($R_{win}$) 与 3D 注意力熵 ($E_{3D, win}$)
衡量每个 3D 窗口内的特征剧烈程度与信息冗余度：
- **残差幅度**：计算窗口内所有 Token 相对于该窗口平均特征锚点 $\bar{x}_{win}$ 的残差 L2 范数期望：
  $$ R_{win} = \mathbb{E} \left[ \frac{1}{P M^2} \sum_{i=1}^{P M^2} \| x_{win, i} - \bar{x}_{win} \|_2 \right] $$
- **3D 窗口注意力熵**：提取 3D W-MSA / 3D SW-MSA 的局部注意力概率矩阵 $A_{3D}$ 计算香农熵（低熵表示强烈关注窗口内某个关键时空点）：
  $$ E_{3D, win} = \mathbb{E} \left[ - \sum_{j} A_{3D} \log_2 (A_{3D} + \epsilon) \right] $$

### 2. 归一化统计与联合阈值标定 ($\tau$)
在校准集上，**分 Stage** 获取上述指标的极值区间并归一化（记为 $\widetilde{R}_{win}, \widetilde{E}_{3D, win}$）。构建综合敏感度得分：
$$ S_{win} = \alpha \widetilde{R}_{win} + \beta \left( 1 - \widetilde{E}_{3D, win} \right) $$
提取特定层或 Stage 的前 $p\%$ 分位数作为截断阈值 $\tau$，并预估误差前馈的层间衰减系数 $\lambda$。

---

## 第三阶段：快速评估阶段 (Quick Eval) —— 动态门控与层级位宽分配

**目标**：在迷你验证集上推理，统计各 Transformer Block 内的高敏感窗口比例，进行全局层级位宽分配（Block-Level Bit-width Allocation）。

1. **动态门控得分统计**：
   在前向传播中，基于 Phase 2 阈值，计算当前窗口得分 $\hat{S}_{win}$ 并执行二值阶跃函数：
   $$ Z_{win} = \mathbb{I}(\hat{S}_{win} > \tau) $$

2. **生成位宽分配表 (`block_bits`)**：
   统计每个 Video Swin Block 内触发 $Z_{win}=1$ 的窗口占该层总窗口的比例，得到 `block_high_ratio`。
   若该 Block 的高敏感触发率达到阈值（如 $\ge 30\%$），判定为**信息密集型算子**，分配较高位宽（如 INT8）；反之分配更激进的位宽（如 INT4）。
   最终生成按阶段和深度组织的配置字典 `block_bits`。

---

## 第四阶段：全量评估阶段 (Full Eval) —— 模拟量化与层间误差前馈

**目标**：根据分配好的 `block_bits` 对权重和激活执行模拟量化（Fake Quantization），应用误差补偿减轻 3D 窗口内的量化失真。

### 1. 权重参数的模拟量化 (Weight Fake Quantization)
按 `block_bits` 指定位宽，对 W-MSA、SW-MSA 和 MLP 中的全精度权重 $W$ 注入量化噪声：
$$ W_{sim} = \text{Clamp}(\text{Round}(W / s), q_{min}, q_{max}) \cdot s $$

### 2. 混合精度特征与层间/帧间误差前馈 (Layer-wise Error Forwarding)
由于 Video Swin 的 3D 窗口在空间上解耦而在 Shift 层才产生信息交互，我们采用**深度方向（沿网络层）与相同空间位置的帧间**误差补偿：

- 当 Block 为 **INT8 ($b_{block}=8$)**：计算当前 Block 输出特征的量化误差 $e_{block} = X_{out} - \tilde{X}_{out, INT8}$，利用衰减系数 $\lambda$ 将误差叠加至下一个同分辨率 Block 的残差分支输入中：
  $$ X_{next\_in} \leftarrow X_{next\_in} + \lambda \cdot e_{block} $$
- 当 Block 为 **INT4 ($b_{block}=4$)**：失真剧烈，采用**时间维残差复用（Temporal Residual Reuse）**。利用同属一个 3D 感受野下前一层的反量化输出作为稳定锚点进行补偿，缓解深层网络 4-bit 带来的时空语义坍塌问题。