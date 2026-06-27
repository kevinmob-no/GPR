# 基于高斯过程回归的 CO₂ 浓度预测与滚动更新方法研究

> 参考论文：[11] Duvenaud D, Lloyd J R, Grosse R, et al. Structure Discovery in Nonparametric Regression through Compositional Kernel Search. ICML, 2013.

---

## 项目概述

本项目以 Mauna Loa 大气 CO₂ 浓度数据为基础，实现高斯过程回归（GPR）的完整机器学习流程。核心方法包括：数据清洗与预处理、多项式去趋势、复合核函数 GPR 建模、滚动更新策略，以及残差分析与模型对比。

---

## 完整实验流程（6步）

### 步骤1：数据加载与清洗 — [step1_load_and_clean.py](gpr_co2/step1_load_and_clean.py)

- 读取 Mauna Loa 月度 CO₂ 月度数据（`monthly_co2_data.csv`）
- 使用 `decimal_date` 作为时间坐标，`co2_filled` 作为浓度值
- 异常值剔除：CO₂ 浓度在 300-500 ppm 且相邻月差分 < 8 ppm
- 构造相对时间特征：以 1958 年为原点，单位：年
- 训练/测试集划分：1958~2016（693 点）| 2016~2026（123 点）
- 特征与目标值标准化（Z-score）
- 输出：`mauna_loa_cleaned.csv` + `mauna_loa_data.npz`

### 步骤2：探索性数据分析 — [step2_exploratory_analysis.py](gpr_co2/step2_exploratory_analysis.py)

- 原始数据可视化（含异常值标注）
- 清洗后数据时序图
- CO₂ 浓度分布直方图
- 快速傅里叶变换（FFT）分析周期成分，验证年度周期性

### 步骤3：多项式去趋势 + 复合核 GPR — [step_co2_detrend_predict.py](gpr_co2/step_co2_detrend_predict.py)

**去趋势**：
- 使用 1990~2015 年训练数据拟合三阶多项式（从 1/2/3 阶中选残差标准差最小者）
- 趋势表达式：`p(t) = 7.133×10⁶ − 1.065×10⁴·t + 5.299·t² − 8.787×10⁻⁴·t³`
- 残差标准差：2.240 ppm

**核函数设计**：

| 成分 | 核函数 | 作用 |
|------|--------|------|
| 长期趋势核 | `C × RBF(length_scale=10)` | 捕捉残差中缓变趋势 |
| 季节周期核 | `C × RBF × ExpSineSquared(period=1)` | 固定 1 年周期，幅度慢变 |
| 噪声核 | `WhiteKernel` | 捕获观测白噪声 |

**结果**：

| 模型 | RMSE | R² | MAE | MAPE | 95% CI 覆盖率 |
|------|------|----|-----|------|--------------|
| **RBF + 周期核** | **0.5765 ppm** | **0.9944** | **0.4486 ppm** | **0.11%** | **99.2%** |
| 仅 RBF（无周期） | 17.3823 ppm | -4.1273 | 16.8025 ppm | 4.03% | — |

> 输出：`gpr_composite_kernel_prediction.png` + `gpr_results.npz`

### 步骤4：滚动更新 GPR — [step5_r.py](gpr_co2/step5_r.py)

针对标准 GPR 外推能力弱的缺陷，引入滚动更新（在线学习）策略：

1. 基于初始训练集（1958~1989）训练 GPR 模型
2. 按时间顺序预测测试集的下一个点（仅一步）
3. 将该点的真实值加入训练集，重新训练
4. 重复步骤 2~3 直至覆盖整个测试期

**优势**：
- 动态吸收最新观测数据，自适应跟踪趋势变化
- 核超参数随数据更新，提升外推精度
- RMSE 降至约 0.42 ppm，95% CI 覆盖率约 94.4%

> 输出（运行后）：`rolling_update_optimized.png`

### 步骤5：残差分析与不确定性评估 — [step6_error_analysis.py](gpr_co2/step6_error_analysis.py)

- 残差分布直方图 + Q-Q 图（正态性检验）
- 残差 vs 预测值散点图
- 预测区间覆盖率计算及可视化

### 步骤6：模型对比 — [step7_model_comparison.py](gpr_co2/step7_model_comparison.py)

- 基准模型：基础 GPR（仅 RBF 核）
- 对比模型：去趋势 + 复合核 GPR（RBF + ExpSineSquared 周期核）
- 输出对比表格与可视化图

> 输出（运行后）：`model_comparison.png`

---

## 运行顺序

```bash
python step1_load_and_clean.py      # 数据清洗
python step2_exploratory_analysis.py # 探索性分析
python step_co2_detrend_predict.py  # 去趋势 + 复合核 GPR（主模型）
python step5_r.py                   # 滚动更新 GPR（改进模型）
python step6_error_analysis.py      # 残差分析
python step7_model_comparison.py    # 模型对比
```

---

## 数据集

- **来源**：Scripps CO₂ Program, Mauna Loa Observatory
- **文件**：`monthly_in_situ_co2_mlo.csv` / `monthly_co2_data.csv`
- **时间范围**：1958 年 3 月 ~ 2026 年 2 月
- **观测频率**：月度平均

---

## 依赖环境

- Python ≥ 3.8
- numpy, pandas, matplotlib, scikit-learn, scipy

---

## 参考文献

- [11] Duvenaud D, Lloyd J R, Grosse R, et al. Structure Discovery in Nonparametric Regression through Compositional Kernel Search. ICML, 2013.
- [4] Rasmussen C E, Williams C K I. Gaussian Processes for Machine Learning. MIT Press, 2006.
- [2] Keeling C D, Whorf T P. Atmospheric CO2 records from sites in the SIO air sampling network. CDIAC, 2005.
