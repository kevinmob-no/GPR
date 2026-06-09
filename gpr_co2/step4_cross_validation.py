# step4_cross_validation.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.metrics import mean_squared_error
from utils import set_chinese_font

set_chinese_font()

print("="*60)
print("第四步：时间序列交叉验证 (Expanding Window)")
print("="*60)

df = pd.read_csv('mauna_loa_cleaned.csv')
X = df['time_norm'].values.reshape(-1, 1)
y = df['co2'].values
years = df['time'].values

print(f"数据年份范围: {years.min():.2f} - {years.max():.2f}, 总样本数: {len(years)}")

test_years = np.arange(1990, 2002, 1)
rmse_list = []

plt.figure(figsize=(12, 6))

for test_year in test_years:
    train_mask = years < test_year
    test_mask = (years >= test_year) & (years < test_year + 1)
    if test_mask.sum() == 0:
        print(f"年份 {test_year}: 无测试数据，跳过")
        continue

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]

    y_mean = y_train.mean()
    y_std = y_train.std()
    if y_std == 0:
        y_std = 1
    y_train_scaled = (y_train - y_mean) / y_std

    kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(1e-3, (1e-5, 1e1))
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)
    gpr.fit(X_train, y_train_scaled)
    y_pred_scaled = gpr.predict(X_test)
    y_pred = y_pred_scaled * y_std + y_mean
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmse_list.append(rmse)

    label_real = f'{test_year}真实' if test_year == 1990 else ""
    label_pred = f'{test_year}预测' if test_year == 1990 else ""
    year_vals = years[test_mask]
    plt.plot(year_vals, y_test, 'go', markersize=4, label=label_real)
    plt.plot(year_vals, y_pred, 'r-', alpha=0.7, label=label_pred)

if rmse_list:
    print(f"逐年预测 RMSE 列表: {[f'{x:.2f}' for x in rmse_list]}")
    print(f"平均 RMSE: {np.mean(rmse_list):.4f} ppm")
    plt.xlabel('年份')
    plt.ylabel('CO2 浓度 (ppm)')
    plt.title('逐年滚动预测（RBF核，扩展窗口训练）')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('rolling_forecast.png', dpi=150)
    plt.show()
else:
    print("没有成功预测任何年份，请检查数据。")