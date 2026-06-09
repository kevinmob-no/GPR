# step1_load_and_clean.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import set_chinese_font

set_chinese_font()

print("="*60)
print("第一步：数据加载与预处理（使用 decimal_date 和 co2_filled）")
print("="*60)

# 读取 CSV，跳过前3行注释
col_names = ['year', 'month', 'day', 'decimal_date', 'co2_raw', 'seasonally_adj',
             'fit', 'seasonally_fit', 'co2_filled', 'seasonally_filled', 'station']
df = pd.read_csv('monthly_co2_data.csv', skiprows=3, names=col_names, encoding='utf-8')
print(f"原始数据形状: {df.shape}")

# 使用 decimal_date 作为时间坐标（已经是年份加小数，最准确）
df['time'] = df['decimal_date']

# 使用 co2_filled 列（已由数据提供者插值填补，不含缺失值）
df['co2'] = pd.to_numeric(df['co2_filled'], errors='coerce')

# 删除时间或浓度无效的行
df = df.dropna(subset=['time', 'co2'])
print(f"删除无效行后样本数: {len(df)}")

# 异常值剔除（基于常识：CO2 浓度应在 300-500 ppm 之间，且年增长不应超过 10 ppm）
mask_normal = (df['co2'] > 300) & (df['co2'] < 500)
# 计算相邻月差分，剔除突变（> 8 ppm）
df['diff'] = df['co2'].diff().abs()
mask_diff = df['diff'] < 8
df_clean = df[mask_normal & mask_diff].copy()
df_clean.drop(columns='diff', inplace=True)
print(f"异常值剔除后样本数: {len(df_clean)}")

# 排序（按时间）
df_clean = df_clean.sort_values('time').reset_index(drop=True)

# 可选：将时间标准化（从 0 开始，便于数值稳定性）
df_clean['time_norm'] = (df_clean['time'] - df_clean['time'].min())

# 保存清洗后的数据
df_clean.to_csv('mauna_loa_cleaned.csv', index=False)

# 保存数组
X = df_clean['time_norm'].values.reshape(-1, 1)
y = df_clean['co2'].values
np.savez('co2_cleaned.npz', X=X, y=y, time=df_clean['time'].values, year=df_clean['time'].values)

# 划分训练/测试集（1990年之前训练，之后测试）
cutoff = 1990.0
train_mask = df_clean['time'] < cutoff
X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[~train_mask], y[~train_mask]
y_mean = y_train.mean()
y_train_centered = y_train - y_mean

np.savez('mauna_loa_data.npz',
         X_train=X_train, y_train_centered=y_train_centered, y_mean=y_mean,
         X_test=X_test, y_test=y_test,
         year_train=df_clean['time'].values[train_mask],
         year_test=df_clean['time'].values[~train_mask])
print(f"训练集样本数: {len(X_train)} (时间 < 1990)")
print(f"测试集样本数: {len(X_test)} (时间 ≥ 1990)")

# 可视化最终数据
plt.figure(figsize=(12, 5))
plt.plot(df_clean['time'], df_clean['co2'], 'b-', linewidth=0.5)
plt.xlabel('年份')
plt.ylabel('CO2 浓度 (ppm)')
plt.title('Mauna Loa CO2 数据（经异常剔除后）')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('raw_data.png', dpi=150)
plt.show()

print("第一步完成！")