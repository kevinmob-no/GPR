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
    pred_data = np.load('gpr_results.npz')
    y_pred_adv = pred_data['y_pred_composite']
    rmse_adv = pred_data['rmse_composite']
    r2_adv = pred_data['r2_composite']
    print("已加载复合核预测结果 (gpr_results.npz)")
except FileNotFoundError:
    print("未找到 gpr_results.npz，正在重新训练...")
    # 重新运行 step_co2_detrend_predict.py 或用简单复合核训练
    from sklearn.gaussian_process.kernels import ExpSineSquared
    k_long = C(1.0, (1e-3, 1e3)) * RBF(10.0, (1e-1, 1e2))
    k_season = C(1.0) * RBF(5.0, (1e-1, 1e2)) * ExpSineSquared(length_scale=1.0, periodicity=1.0,
                                                                  length_scale_bounds=(1e-2, 1e1),
                                                                  periodicity_bounds=(1.0, 1.0))
    k_noise = WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1e1))
    co2_kernel = k_long + k_season + k_noise
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
print(f"{'复合核GPR (去趋势+周期核)':<20} {rmse_adv:<15.4f} {r2_adv:<10.4f}")
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
