# step6_error_analysis.py
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from utils import set_chinese_font

set_chinese_font()

print("="*60)
print("第六步：残差分析与不确定性评估")
print("="*60)

# ----------------------------- 1. 加载数据 -----------------------------
# 加载第三步（去趋势+复合核GPR）的预测结果
pred_data = np.load('gpr_results.npz')
y_test = pred_data['y_test']
y_pred = pred_data['y_pred_composite']
y_std = pred_data['y_std_composite']

# ----------------------------- 2. 计算残差 -----------------------------
residuals = y_test - y_pred

# 基本统计
print(f"残差均值: {np.mean(residuals):.4f} ppm")
print(f"残差标准差: {np.std(residuals):.4f} ppm")
print(f"残差范围: [{np.min(residuals):.4f}, {np.max(residuals):.4f}] ppm")

# ----------------------------- 3. 残差分布直方图与Q-Q图 -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 直方图
axes[0].hist(residuals, bins=30, density=True, alpha=0.7, color='steelblue', edgecolor='black')
axes[0].set_xlabel('残差 (ppm)')
axes[0].set_ylabel('密度')
axes[0].set_title('残差分布直方图')
axes[0].grid(alpha=0.3)

# Q-Q 图
stats.probplot(residuals, dist="norm", plot=axes[1])
axes[1].set_title('残差 Q-Q 图')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('residual_analysis.png', dpi=150)
plt.show()

# ----------------------------- 4. 残差 vs 预测值 -----------------------------
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.6, edgecolor='k')
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
plt.xlabel('预测值 (ppm)')
plt.ylabel('残差 (ppm)')
plt.title('残差与预测值的关系')
plt.grid(alpha=0.3)
plt.savefig('residuals_vs_pred.png', dpi=150)
plt.show()

# ----------------------------- 5. 预测区间覆盖率 -----------------------------
# 95% 置信区间
lower = y_pred - 1.96 * y_std
upper = y_pred + 1.96 * y_std
coverage = np.mean((y_test >= lower) & (y_test <= upper))
print(f"95% 置信区间实际覆盖率: {coverage*100:.2f}%")

# 绘制预测区间（前200个测试点）
plt.figure(figsize=(12, 6))
idx = np.arange(200)
plt.plot(idx, y_test[:200], 'k.', label='真实值')
plt.plot(idx, y_pred[:200], 'b-', label='预测均值')
plt.fill_between(idx, lower[:200], upper[:200], alpha=0.3, color='blue', label='95% 置信区间')
plt.xlabel('测试样本索引')
plt.ylabel('CO2 浓度 (ppm)')
plt.title('预测不确定性展示（前200个测试点）')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('prediction_intervals.png', dpi=150)
plt.show()

print("第六步完成！")
