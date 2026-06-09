import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from utils import set_chinese_font

set_chinese_font()

print("="*60)
print("滚动更新高斯过程回归 - 开始运行")
print("="*60)

# ----------------------------- 1. 加载数据 -----------------------------
print("步骤1: 加载预处理数据...")
data = np.load('mauna_loa_data.npz')
X_train = data['X_train'].flatten()          # 相对时间 (0~32)
y_train = data['y_train_centered'] + data['y_mean']  # 原始浓度
X_test = data['X_test'].flatten()            # 相对时间 (32~44)
y_test = data['y_test']
print(f"  训练集样本数: {len(X_train)} (1958-1989)")
print(f"  测试集样本数: {len(X_test)} (1990-2001)")

# ----------------------------- 2. 标准化输入特征 X -----------------------------
print("步骤2: 标准化输入特征...")
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train.reshape(-1, 1)).flatten()
X_test_scaled = scaler_X.transform(X_test.reshape(-1, 1)).flatten()

# ----------------------------- 3. 标准化目标值 y -----------------------------
print("步骤3: 标准化目标值...")
y_mean_train = y_train.mean()
y_std_train = y_train.std()
y_train_scaled = (y_train - y_mean_train) / y_std_train
print(f"  目标值均值: {y_mean_train:.2f} ppm, 标准差: {y_std_train:.2f} ppm")

# 初始化训练集（使用标准化后的 X 和 y）
X_current = X_train_scaled.reshape(-1, 1)
y_current_scaled = y_train_scaled

# ----------------------------- 4. 核函数 -----------------------------
print("步骤4: 定义核函数...")
kernel = C(1.0, (1e-2, 1e2)) * RBF(1.0, (1e-1, 5e1)) + WhiteKernel(1e-2, (1e-4, 1e0))
print(f"  核函数: {kernel}")

rolling_pred = []   # 存储预测值（原始尺度）
rolling_std = []    # 存储预测标准差（原始尺度）

total_points = len(X_test_scaled)
print(f"步骤5: 开始滚动更新预测（共 {total_points} 个点，约 {total_points/12:.1f} 年）...")

# ----------------------------- 5. 滚动更新循环 -----------------------------
for i in range(total_points):
    # 当前要预测的点（标准化后的）
    x_point = np.array([[X_test_scaled[i]]])

    # 训练 GPR：每次重新优化超参数，增加重启次数以增强稳定性
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=10,   # 从 5 增加到 10
        random_state=42,
        alpha=1e-6                 # 小噪声增强数值稳定性
    )
    gpr.fit(X_current, y_current_scaled)

    # 预测并获取标准差
    y_pred_scaled, y_std_scaled = gpr.predict(x_point, return_std=True)

    # 还原到原始尺度
    y_pred = y_pred_scaled[0] * y_std_train + y_mean_train
    y_std = y_std_scaled[0] * y_std_train

    rolling_pred.append(y_pred)
    rolling_std.append(y_std)

    # 将真实值加入训练集（需要先标准化到与 y_current_scaled 同一尺度）
    y_true_scaled = (y_test[i] - y_mean_train) / y_std_train
    X_current = np.vstack([X_current, x_point])
    y_current_scaled = np.append(y_current_scaled, y_true_scaled)

    # 每 12 个月（一年）打印一次进度和当前 RMSE
    if (i + 1) % 12 == 0:
        current_rmse = np.sqrt(mean_squared_error(y_test[:i+1], rolling_pred))
        print(f"  已处理 {i+1}/{total_points} 个点 ({(i+1)/total_points*100:.1f}%)，当前 RMSE = {current_rmse:.4f} ppm")

# ----------------------------- 6. 评估与可视化 -----------------------------
print("步骤6: 计算最终性能指标...")
rmse = np.sqrt(mean_squared_error(y_test, rolling_pred))
r2 = r2_score(y_test, rolling_pred)
# 置信区间覆盖率
lower = np.array(rolling_pred) - 1.96 * np.array(rolling_std)
upper = np.array(rolling_pred) + 1.96 * np.array(rolling_std)
coverage = np.mean((y_test >= lower) & (y_test <= upper))

print("\n" + "="*60)
print("滚动更新 GPR 测试集最终性能:")
print(f"  RMSE = {rmse:.4f} ppm")
print(f"  R²   = {r2:.4f}")
print(f"  95% 置信区间覆盖率 = {coverage*100:.1f}%")
print("="*60)

# 绘图
print("步骤7: 生成预测对比图...")
years_test = X_test + 1958
plt.figure(figsize=(12, 6))
plt.plot(years_test, y_test, 'k.', markersize=4, label='真实值')
plt.plot(years_test, rolling_pred, 'r-', linewidth=2, label='滚动预测')
plt.fill_between(years_test,
                 np.array(rolling_pred) - 1.96 * np.array(rolling_std),
                 np.array(rolling_pred) + 1.96 * np.array(rolling_std),
                 alpha=0.3, color='red', label='95% 置信区间')
plt.xlabel('年份')
plt.ylabel('CO2 浓度 (ppm)')
plt.title(f'滚动更新 GPR 预测效果 (RMSE={rmse:.2f} ppm, R²={r2:.3f})')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('rolling_update_optimized.png', dpi=150)
plt.show()

# 保存结果
print("步骤8: 保存预测结果...")
np.savez('rolling_update_optimized.npz',
         years=years_test, true=y_test, pred=rolling_pred, std=rolling_std,
         rmse=rmse, r2=r2, coverage=coverage)
print("结果已保存至 rolling_update_optimized.npz")

print("\n所有步骤完成！")
print("="*60)