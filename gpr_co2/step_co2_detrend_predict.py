# -*- coding: utf-8 -*-
"""
优化后的高斯过程回归模型（趋势分解 + 复合核GPR）
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial as P
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C, ExpSineSquared
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ----------------------------- 环境设置 -----------------------------
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import set_chinese_font
set_chinese_font()
warnings.filterwarnings("ignore", category=UserWarning)

print("=" * 60)
print("高斯过程回归模型：多项式趋势 + 复合核GPR")
print("=" * 60)

# ----------------------------- 1. 加载和划分数据 -----------------------------
df = pd.read_csv("mauna_loa_cleaned.csv").sort_values("time").reset_index(drop=True)
t_all, c_all = df["time"].values, df["co2"].values

split_year = 2016.0
train_mask = t_all < split_year
test_mask = t_all >= split_year

t_train, c_train = t_all[train_mask], c_all[train_mask]
t_test, c_test = t_all[test_mask], c_all[test_mask]

print(f"训练集: {t_train.shape[0]} 点 (年份 {t_train.min():.1f} - {t_train.max():.1f})")
print(f"测试集: {t_test.shape[0]} 点 (年份 {t_test.min():.1f} - {t_test.max():.1f})")

# ----------------------------- 2. 多项式趋势拟合 -----------------------------
# 只使用1990年后的训练数据来确定多项式阶数（避免早期数据波动影响）
poly_start = 1990.0
poly_mask = (t_train >= poly_start)

best_degree = 3
best_std = np.inf
polynomials = {}

for degree in [1, 2, 3]:
    p = P.fit(t_train[poly_mask], c_train[poly_mask], degree)
    polynomials[degree] = p
    residual_std = np.std(c_train[poly_mask] - p(t_train[poly_mask]))
    if residual_std < best_std:
        best_std, best_degree = residual_std, degree

trend_poly = polynomials[best_degree]
print(f"\n最佳多项式阶数: {best_degree}, 残差标准差: {best_std:.3f} ppm")

# 打印多项式表达式
coefs = trend_poly.convert().coef
poly_expr = " + ".join(f"{c:.6e}" + (f"*t^{i}" if i > 0 else "") for i, c in enumerate(coefs))
print(f"趋势多项式: p(t) = {poly_expr}")

# 计算全部时间点上的趋势值和残差
trend_all = trend_poly(t_all)
residual_all = c_all - trend_all
residual_mean = residual_all.mean()           # 用于中心化

residual_train = c_train - trend_poly(t_train)
residual_test  = c_test  - trend_poly(t_test)

# ----------------------------- 3. 准备GPR输入（中心化残差） -----------------------------
time_offset = t_all.min()                     # 平移时间，提高数值稳定性
X_train = (t_train - time_offset).reshape(-1, 1)
y_train_centered = residual_train - residual_mean

X_test = (t_test - time_offset).reshape(-1, 1)

print(f"\nGPR训练输入形状: {X_train.shape}, 中心化残差均值为 {y_train_centered.mean():.3e}")

# ----------------------------- 4. 定义复合核函数 -----------------------------
# (a) 长期趋势核：大尺度RBF
k_long = C(1.0, (1e-3, 1e3)) * RBF(10.0, (1e-1, 1e2))

# (b) 季节性周期核：周期=1年，并允许幅度随时间缓慢变化
k_season = C(1.0) * RBF(5.0, (1e-1, 1e2)) * ExpSineSquared(length_scale=1.0, periodicity=1.0,
                                                              length_scale_bounds=(1e-2, 1e1),
                                                              periodicity_bounds=(1.0, 1.0))

# (c) 噪声核：白噪声 + 局部波动
k_noise = WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1e1))

# 复合核（长期+季节+噪声）
composite_kernel = k_long + k_season + k_noise

print("\n复合核函数结构:")
print(composite_kernel)

# ----------------------------- 5. 训练高斯过程回归模型 -----------------------------
gpr = GaussianProcessRegressor(
    kernel=composite_kernel,
    n_restarts_optimizer=5,
    random_state=42,
    normalize_y=False
)
print("正在训练GPR模型...")
gpr.fit(X_train, y_train_centered)

print("\n优化后的核函数参数:")
print(gpr.kernel_)
print(f"对数边际似然: {gpr.log_marginal_likelihood_value_:.3f}")

# ----------------------------- 6. 预测测试集 -----------------------------
y_pred_centered, y_std = gpr.predict(X_test, return_std=True)
y_pred_residual = y_pred_centered + residual_mean          # 加上之前减去的残差均值
y_pred = trend_poly(t_test) + y_pred_residual              # 加上趋势项，得到最终CO2预测

# ----------------------------- 7. 评估指标 -----------------------------
rmse = np.sqrt(mean_squared_error(c_test, y_pred))
mae = mean_absolute_error(c_test, y_pred)
r2 = r2_score(c_test, y_pred)
mape = np.mean(np.abs((c_test - y_pred) / c_test)) * 100

# 95% 置信区间
ci = 1.96
lower = y_pred - ci * y_std
upper = y_pred + ci * y_std
coverage = np.mean((c_test >= lower) & (c_test <= upper))

print("\n测试集性能指标:")
print(f"  RMSE = {rmse:.4f} ppm")
print(f"  MAE  = {mae:.4f} ppm")
print(f"  R²   = {r2:.4f}")
print(f"  MAPE = {mape:.2f}%")
print(f"  95% 置信区间覆盖率: {coverage * 100:.1f}%")

# ----------------------------- 8. 可视化结果 -----------------------------
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(t_test, c_test, 'k-', alpha=0.8, label='真实 CO₂', lw=0.8)
ax.plot(t_test, y_pred, 'r--', alpha=0.8, lw=1.5, label=f'复合核GPR预测 (RMSE={rmse:.3f} ppm)')
ax.fill_between(t_test, lower, upper, color='red', alpha=0.1, label='95% 置信区间')
ax.set_xlabel('年份')
ax.set_ylabel('CO₂ 浓度 (ppm)')
ax.set_xlim(t_test[0], t_test[-1])
ax.set_title('多项式趋势 + 复合核高斯过程回归 - 测试集预测')
ax.legend(fontsize=10, loc='upper left')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("gpr_composite_kernel_prediction.png", dpi=200)
plt.close()

print("\n图表已保存为 gpr_composite_kernel_prediction.png")

# ----------------------------- 9. 对比简单RBF核（可选，与原代码保持对比） -----------------------------
# 为了与原两个模型对比，增加一个仅RBF核的模型
print("\n" + "=" * 60)
print("对比：仅RBF核（无周期成分）")
print("=" * 60)

simple_kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-1, 1e2)) + WhiteKernel(1e-3, (1e-5, 1e1))
gpr_simple = GaussianProcessRegressor(kernel=simple_kernel, n_restarts_optimizer=3, random_state=42)
gpr_simple.fit(X_train, y_train_centered)
y_pred_simple_centered, y_std_simple = gpr_simple.predict(X_test, return_std=True)
y_pred_simple = trend_poly(t_test) + (y_pred_simple_centered + residual_mean)

rmse_simple = np.sqrt(mean_squared_error(c_test, y_pred_simple))
r2_simple = r2_score(c_test, y_pred_simple)

print(f"RBF核模型测试集 RMSE = {rmse_simple:.4f} ppm, R² = {r2_simple:.4f}")
print(f"复合核模型测试集 RMSE = {rmse:.4f} ppm, R² = {r2:.4f}")

# 保存最终结果供后续使用
np.savez('gpr_results.npz',
         y_test=c_test,
         y_pred_composite=y_pred,
         y_std_composite=y_std,
         y_pred_rbf=y_pred_simple,
         rmse_composite=rmse,
         rmse_rbf=rmse_simple,
         r2_composite=r2,
         r2_rbf=r2_simple,
         coverage=coverage)
print("\n结果已保存至 gpr_results.npz")
