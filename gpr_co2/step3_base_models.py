# step3_base_models.py
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              WhiteKernel, ConstantKernel as C,
                                              ExpSineSquared)
from sklearn.metrics import mean_squared_error
from utils import set_chinese_font, load_co2_data

# 设置中文字体（如果本步骤需要绘图，可以调用；不绘图也可以保留）
set_chinese_font()

print("="*60)
print("第三步：不同核函数的基础GPR对比")
print("="*60)

# 加载数据
X_train, y_train_centered, y_mean, X_test, y_test = load_co2_data()
print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")

# 定义候选核函数
kernels = {
    'RBF': C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(1e-3, (1e-5, 1e1)),
    'Matern (ν=1.5)': C(1.0) * Matern(1.0, nu=1.5) + WhiteKernel(1e-3),
    'RationalQuadratic': C(1.0) * RationalQuadratic(alpha=1.0, length_scale=1.0) + WhiteKernel(1e-3),
    'RBF+周期': C(1.0) * RBF(1.0) + C(0.5) * RBF(10.0) * ExpSineSquared(1.0, periodicity=1.0) + WhiteKernel(1e-3)
}

results = {}
for name, kernel in kernels.items():
    print(f"训练 {name} ...")
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)
    gpr.fit(X_train, y_train_centered)
    y_pred_centered = gpr.predict(X_test)
    y_pred = y_pred_centered + y_mean
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    results[name] = {'rmse': rmse, 'kernel': gpr.kernel_}
    print(f"  RMSE = {rmse:.4f} ppm")

# 输出对比表格
print("\n核函数对比结果:")
print(f"{'核函数':<20} {'RMSE (ppm)':<15}")
print("-" * 35)
for name, res in results.items():
    print(f"{name:<20} {res['rmse']:<15.4f}")