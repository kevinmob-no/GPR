# step5_advanced_kernel.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, WhiteKernel, ConstantKernel as C,
                                              RationalQuadratic, ExpSineSquared)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from utils import set_chinese_font

set_chinese_font()

print("="*60)
print("第五步：复合核函数高斯过程回归模型")
print("="*60)

# ----------------------------- 1. 加载数据 -----------------------------
data = np.load('mauna_loa_data.npz')
X_train = data['X_train']
y_train_centered = data['y_train_centered']
y_mean = data['y_mean']
X_test = data['X_test']
y_test = data['y_test']

print(f"训练集样本数: {X_train.shape[0]}, 测试集样本数: {X_test.shape[0]}")

# ----------------------------- 2. 构建复合核函数 -----------------------------
# 长期趋势核：大尺度 RBF
k_long = 66.0**2 * RBF(length_scale=67.0)

# 季节性核：周期核（周期=1年）乘以 RBF，允许季节性幅度随时间变化
k_season = 2.4**2 * RBF(length_scale=90.0) * ExpSineSquared(length_scale=1.3, periodicity=1.0)

# 中期不规则波动核：有理二次核
k_irreg = 0.66**2 * RationalQuadratic(alpha=0.78, length_scale=1.2)

# 噪声核：局部相关 + 白噪声
k_noise = 0.18**2 * RBF(length_scale=0.134) + WhiteKernel(noise_level=0.19**2)

# 总核
co2_kernel = k_long + k_long*k_season + k_noise

print("复合核函数结构:")
print(co2_kernel)

# ----------------------------- 3. 训练GPR模型 -----------------------------
gpr = GaussianProcessRegressor(kernel=co2_kernel, n_restarts_optimizer=5, random_state=42)
print("正在训练复合核GPR模型...")
gpr.fit(X_train, y_train_centered)

print("\n优化后的核函数参数:")
print(gpr.kernel_)
print(f"对数边际似然: {gpr.log_marginal_likelihood_value_:.3f}")

# ----------------------------- 4. 预测与评估 -----------------------------
y_pred_centered, y_std = gpr.predict(X_test, return_std=True)
y_pred = y_pred_centered + y_mean

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n测试集性能指标:")
print(f"  RMSE = {rmse:.4f} ppm")
print(f"  MAE  = {mae:.4f} ppm")
print(f"  R²   = {r2:.4f}")

# 95% 置信区间覆盖率
lower = y_pred - 1.96 * y_std
upper = y_pred + 1.96 * y_std
coverage = np.mean((y_test >= lower) & (y_test <= upper))
print(f"  95% 置信区间覆盖率: {coverage*100:.1f}%")

# ----------------------------- 5. 可视化预测结果 -----------------------------
# 需要将 X_test 转换回实际年份（如果第一步保存了 year_test）
try:
    year_test = data['year_test']   # 假设第一步保存了 year_test
except KeyError:
    # 若没有 year_test，则使用 X_test 作为横坐标（相对时间）
    year_test = X_test.flatten() + 1958  # 近似，X_test 是相对于 1958 的年数

plt.figure(figsize=(12, 6))
plt.plot(year_test, y_test, 'k.', markersize=4, label='真实值')
plt.plot(year_test, y_pred, 'r-', linewidth=2, label='复合核GPR预测')
plt.fill_between(year_test, lower, upper, alpha=0.2, color='red', label='95% 置信区间')
plt.xlabel('年份')
plt.ylabel('CO2 浓度 (ppm)')
plt.title(f'复合核GPR预测效果 (RMSE={rmse:.2f} ppm, R²={r2:.3f})')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('advanced_gpr_prediction.png', dpi=150)
plt.show()

# 保存预测结果供后续步骤使用
np.savez('advanced_predictions.npz',
         y_test=y_test, y_pred=y_pred, y_std=y_std,
         rmse=rmse, mae=mae, r2=r2, coverage=coverage)
print("预测结果已保存至 advanced_predictions.npz")