# utils.py
import matplotlib.pyplot as plt
import matplotlib
import platform
import numpy as np


def set_chinese_font():
    """根据操作系统自动设置 matplotlib 中文字体"""
    system = platform.system()
    if system == "Windows":
        font_candidates = ['Microsoft YaHei', 'SimHei', 'SimSun', 'FangSong']
    elif system == "Darwin":  # macOS
        font_candidates = ['PingFang SC', 'STHeiti', 'Arial Unicode MS']
    else:  # Linux
        font_candidates = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']

    for font in font_candidates:
        try:
            matplotlib.rcParams['font.sans-serif'] = [font]
            matplotlib.rcParams['axes.unicode_minus'] = False
            # 测试字体
            plt.text(0.5, 0.5, '测试', fontsize=10)
            plt.close()
            print(f"已使用中文字体: {font}")
            return
        except:
            continue
    print("警告：未找到合适的中文字体，图表中的中文可能显示为方框。")


def load_co2_data():
    """
    加载第一步生成的预处理数据
    返回: (X_train, y_train_centered, y_mean, X_test, y_test)
    """
    data = np.load('mauna_loa_data.npz')
    return (data['X_train'], data['y_train_centered'], data['y_mean'],
            data['X_test'], data['y_test'])