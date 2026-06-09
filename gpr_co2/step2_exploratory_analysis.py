# step2_exploratory_analysis.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
from utils import set_chinese_font

set_chinese_font()

print("=" * 60)
print("第二步：探索性数据分析")
print("=" * 60)

# 加载清洗后的数据（由第一步生成）
df = pd.read_csv('mauna_loa_cleaned.csv')

# 确保数据按时间排序
df = df.sort_values('time').reset_index(drop=True)
years = df['time'].values  # 已经是年份（小数）
co2 = df['co2'].values

# ----------------------------- 1. 长期趋势分解 -----------------------------
window = 12  # 12个月移动平均（假设月度数据）
trend = pd.Series(co2).rolling(window=window, center=True).mean().values

plt.figure(figsize=(12, 6))
plt.plot(years, co2, alpha=0.5, label='原始序列')
plt.plot(years, trend, 'r-', linewidth=2, label='12个月移动平均趋势')
plt.xlabel('年份')
plt.ylabel('CO2 浓度 (ppm)')
plt.title('CO2 长期趋势分解')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('trend_decomposition.png', dpi=150)
plt.show()

# ----------------------------- 2. 季节性分析 -----------------------------
# 使用原始数据中的月份列（如果存在）
if 'month' in df.columns:
    months = df['month'].values
    detrended = co2 - trend
    # 按月计算平均偏差
    df_temp = pd.DataFrame({'month': months, 'detrended': detrended})
    monthly_avg = df_temp.groupby('month')['detrended'].mean()

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(years, detrended, alpha=0.5)
    plt.xlabel('年份')
    plt.ylabel('去趋势后的波动 (ppm)')
    plt.title('去趋势后的波动')
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.bar(monthly_avg.index, monthly_avg.values, color='steelblue')
    plt.xlabel('月份')
    plt.ylabel('平均季节性偏差 (ppm)')
    plt.title('季节性分量')
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('seasonal_pattern.png', dpi=150)
    plt.show()
else:
    print("警告：CSV 中缺少 month 列，跳过季节性分析。")

# ----------------------------- 3. 自相关分析 -----------------------------
plt.figure(figsize=(10, 5))
autocorrelation_plot(co2)
plt.title('CO2 浓度自相关图')
plt.grid(alpha=0.3)
plt.savefig('autocorrelation.png', dpi=150)
plt.show()

print("第二步完成！")