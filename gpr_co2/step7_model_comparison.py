# step7_model_comparison.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.metrics import mean_squared_error, r2_score
from utils import set_chinese_font

set_chinese_font()

print("="*60)
print("第七步：模型对比（基础GPR vs 复合核GPR）")
print("="*60)

# 加载数据
data = np.load('mauna_loa_data.npz')
X_train = data['X_train']
y_train_centered = data['y_train_centered']
y_mean = data['y_mean']
X_test = data['X_test']
y_test = data['y_test']

# 获取测试集的年份（用于绘图）
try:
    year_test = data['year_test']
except KeyError:
    year_test = X_test.flatten() + 1958

# 训练基础GPR（RBF核）
kernel_base = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(1e-3, (1e-5, 1e1))
gpr_base = GaussianProcessRegressor(kernel=kernel_base, n_restarts_optimizer=5, random_state=42)
gpr_base.fit(X_train, y_train_centered)
y_pred_base_centered = gpr_base.predict(X_test)
y_pred_base = y_pred_base_centered + y_mean
rmse_base = np.sqrt(mean_squared_error(y_test, y_pred_base))
r2_base = r2_score(y_test, y_pred_base)

# 加载或训练复合核GPR
try:
    pred_data = np.load('advanced_predictions.npz')
    y_pred_adv = pred_data['y_pred']
    rmse_adv = pred_data['rmse']
    r2_adv = pred_data['r2']
    print("已加载复合核预测结果 (advanced_predictions.npz)")
except FileNotFoundError:
    print("未找到 advanced_predictions.npz，重新训练复合核模型...")
    from sklearn.gaussian_process.kernels import RationalQuadratic, ExpSineSquared
    k_long = 66.0**2 * RBF(length_scale=67.0)
    k_season = 2.4**2 * RBF(length_scale=90.0) * ExpSineSquared(length_scale=1.3, periodicity=1.0)
    k_irreg = 0.66**2 * RationalQuadratic(alpha=0.78, length_scale=1.2)
    k_noise = 0.18**2 * RBF(length_scale=0.134) + WhiteKernel(noise_level=0.19**2)
    co2_kernel = k_long + k_season + k_irreg + k_noise
    gpr_adv = GaussianProcessRegressor(kernel=co2_kernel, n_restarts_optimizer=5, random_state=42)
    gpr_adv.fit(X_train, y_train_centered)
    y_pred_adv_centered = gpr_adv.predict(X_test)
    y_pred_adv = y_pred_adv_centered + y_mean
    rmse_adv = np.sqrt(mean_squared_error(y_test, y_pred_adv))
    r2_adv = r2_score(y_test, y_pred_adv)

# 输出对比表格
print("\n测试集性能对比:")
print(f"{'模型':<20} {'RMSE (ppm)':<15} {'R²':<10}")
print("-" * 45)
print(f"{'基础GPR (RBF核)':<20} {rmse_base:<15.4f} {r2_base:<10.4f}")
print(f"{'复合核GPR (论文复现)':<20} {rmse_adv:<15.4f} {r2_adv:<10.4f}")
print(f"\n复合核相比基础模型 RMSE 降低: {rmse_base - rmse_adv:.4f} ppm")

# 可视化对比
plt.figure(figsize=(12, 6))
plt.plot(year_test, y_test, 'k.', markersize=4, label='真实值')
plt.plot(year_test, y_pred_base, 'b-', linewidth=1.5, label=f'基础GPR (RMSE={rmse_base:.2f})')
plt.plot(year_test, y_pred_adv, 'r-', linewidth=2, label=f'复合核GPR (RMSE={rmse_adv:.2f})')
plt.xlabel('年份')
plt.ylabel('CO2 浓度 (ppm)')
plt.title('模型预测效果对比')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150)
plt.show()

print("第七步完成！对比图已保存为 model_comparison.png")