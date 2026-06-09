# -*- coding: utf-8 -*-
"""
step5_compositional_kernel_search.py
基于 Duvenaud et al. (ICML 2013) 的组合核搜索方法,
自动发现 Mauna Loa CO2 数据的最佳核函数结构。

核心思路:
1. 四种基核: SE (RBF), Per (ExpSineSquared), Lin (DotProduct), RQ (RationalQuadratic)
2. 组合运算: 加法 (+) 和乘法 (x)
3. 搜索算子: Add (S->S+B), Multiply (S->SxB), Replace (B->B')
4. 评分: 边缘似然 (Marginal Likelihood) + BIC
5. 策略: 贪心搜索, 逐步扩展最优核

参考原文:
  Section 2 - Expressing structure through kernels (基核与组合)
  Section 3 - Searching over structures (搜索算法)
  Section 5 - Structure discovery in time series (时间序列应用)
  Figure 3-4 - Mauna Loa CO2 结果

使用方法:
  可用系统 Python 直接运行: python step5_compositional_kernel_search.py
  或指定 Anaconda Python: "E:\Anaconda3\python.exe" step5_compositional_kernel_search.py
"""

# ============================================================
# [0] 自动依赖配置 (Auto-dependency setup)
# ============================================================
import subprocess, sys, importlib, os, textwrap

REQUIRED_PACKAGES = {
    "numpy":     "numpy",
    "pandas":    "pandas",
    "sklearn":   "scikit-learn",
    "matplotlib":"matplotlib",
    "scipy":     "scipy",
}

_missing = []
for mod, pkg in REQUIRED_PACKAGES.items():
    try:
        importlib.import_module(mod)
    except ImportError:
        _missing.append(pkg)

if _missing:
    print("=" * 60)
    print("[配置] 检测到缺少依赖包, 尝试自动安装...")
    print(f"      缺少: {', '.join(_missing)}")
    for pkg in _missing:
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pkg, "--quiet"],
                timeout=120)
            print(f"      [+] {pkg} 安装成功")
        except Exception:
            pass
    print("=" * 60)

# 再次验证
for mod, pkg in REQUIRED_PACKAGES.items():
    try:
        importlib.import_module(mod)
    except ImportError:
        print(f"[错误] 无法加载 {mod}({pkg})。请手动安装:")
        print(f"       {sys.executable} -m pip install {pkg}")
        sys.exit(1)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 无 GUI 后端, 避免 Windows 上弹窗
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, WhiteKernel, ConstantKernel as C,
    RationalQuadratic, ExpSineSquared, DotProduct
)
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

# 中文字体设置
_CN_FONTS = ["Microsoft YaHei", "SimHei", "WenQuanYi Micro Hei", "DejaVu Sans"]
for _f in _CN_FONTS:
    try:
        matplotlib.rcParams["font.sans-serif"] = [_f]
        matplotlib.rcParams["axes.unicode_minus"] = False
        plt.figure(); plt.text(0.5,0.5,"测"); plt.close()
        break
    except:
        continue

# ============================================================
# 1. 数据加载与预处理
# ============================================================
print("=" * 60)
print("组合核搜索 (Compositional Kernel Search) for Mauna Loa CO2")
print("参考: Duvenaud et al. ICML 2013, Structure Discovery in")
print("       Nonparametric Regression through Compositional Kernel Search")
print("=" * 60)

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(DATA_DIR)

col_names = ["year","month","day","decimal_date","co2_raw","seasonally_adj",
             "fit","seasonally_fit","co2_filled","seasonally_filled","station"]
df = pd.read_csv("monthly_co2_data.csv", skiprows=3, names=col_names, encoding="utf-8")

df["year"] = df["decimal_date"].astype(int)
df["month_float"] = df["decimal_date"] - df["year"]
df["month"] = (df["month_float"] * 12 + 0.5).astype(int)
df.loc[df["month"] == 0, "month"] = 1
df.loc[df["month"] == 13, "month"] = 12
df["co2"] = pd.to_numeric(df["co2_raw"], errors="coerce")

monthly = df.groupby(["year","month"]).agg({"co2":"mean"}).reset_index()
monthly = monthly.dropna(subset=["co2"])

X_raw = (monthly["year"] + monthly["month"] / 12).values.reshape(-1, 1)
y_raw = monthly["co2"].values

time_min = float(X_raw.min())
X = X_raw - time_min  # 从 1958 年起经过的年数
y = y_raw.copy()

cutoff_time = 1990.0 - time_min
train_mask = X.ravel() < cutoff_time
X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[~train_mask], y[~train_mask]
year_test = X_raw[~train_mask].ravel()

y_mean = float(y_train.mean())
y_train_centered = y_train - y_mean

print(f"训练集: {len(X_train)} 样本 (1958-1989)")
print(f"测试集: {len(X_test)} 样本 (1990-{int(year_test.max())})")

# ============================================================
# 2. 定义基核家族 (Base Kernel Families)
# ============================================================
# 参考原文 Section 2, Figure 1

BASE_KERNELS = {}
BASE_KERNELS["SE"]  = lambda: C(1.0, (1e-3, 1e3)) * RBF(length_scale=10.0, length_scale_bounds=(0.5, 500))
BASE_KERNELS["Per"] = lambda: C(1.0, (1e-3, 1e3)) * ExpSineSquared(
    length_scale=1.0, length_scale_bounds=(0.1, 20),
    periodicity=1.0, periodicity_bounds=(0.5, 2.0))
BASE_KERNELS["Lin"] = lambda: C(1.0, (1e-3, 1e3)) * DotProduct(sigma_0=0.0, sigma_0_bounds="fixed")
BASE_KERNELS["RQ"]  = lambda: C(1.0, (1e-3, 1e3)) * RationalQuadratic(
    alpha=1.0, alpha_bounds=(0.1, 100),
    length_scale=5.0, length_scale_bounds=(0.5, 500))

# ============================================================
# 3. 核搜索算法实现 (参考原文 Section 3)
# ============================================================

class KernelNode:
    """核表达式树节点"""

    def __init__(self, op, left=None, right=None, kernel=None, name=""):
        self.op = op          # "base" | "+" | "x"
        self.left = left
        self.right = right
        self.kernel = kernel
        self.name = name

    def to_sklearn(self):
        if self.op == "base":
            return self.kernel
        elif self.op == "+":
            return self.left.to_sklearn() + self.right.to_sklearn()
        elif self.op == "x":
            return self.left.to_sklearn() * self.right.to_sklearn()

    def to_str(self):
        return self.name

    def copy(self):
        return deepcopy(self)


def make_base_kernel(kernel_type):
    return KernelNode(op="base", kernel=BASE_KERNELS[kernel_type](), name=kernel_type)


def eval_kernel(node, Xt, yt):
    """评估核: 训练 GPR, 返回 (log_marginal_likelihood, optimized_kernel)"""
    try:
        gpr = GaussianProcessRegressor(
            kernel=node.to_sklearn(),
            n_restarts_optimizer=3,
            random_state=42,
            normalize_y=False,
        )
        gpr.fit(Xt, yt)
        return float(gpr.log_marginal_likelihood_value_), gpr.kernel_
    except Exception:
        return -1e10, None


def bic_score(log_ml, n_params, n_samples):
    """BIC = log(ML) - (M/2)*log(N)"""
    return log_ml - 0.5 * n_params * np.log(n_samples)


class KernelSearch:
    """组合核搜索 (参考原文 Section 3: Searching over structures)"""

    def __init__(self, Xt, yt, max_depth=5, verbose=True):
        self.Xt = Xt
        self.yt = yt
        self.max_depth = max_depth
        self.verbose = verbose
        self.n = len(yt)
        self.candidates = []      # (node, bic_score, opt_kernel)
        self.explored = set()

    def initialize(self):
        """Step 1: 所有基核家族应用到输入维度"""
        for name in ["SE", "Per", "Lin", "RQ"]:
            node = make_base_kernel(name)
            log_ml, ok = eval_kernel(node, self.Xt, self.yt)
            score = bic_score(log_ml, 3, self.n)
            self.candidates.append((node, score, ok))
            self.explored.add(node.to_str())
            if self.verbose:
                print(f"  基核 {name:4s}: logML={log_ml:7.2f}, BIC={score:7.2f}")

    def _expand(self, node):
        """应用三个搜索算子生成新候选"""
        results = []
        for name in ["SE", "Per", "Lin", "RQ"]:
            # Add: S -> S + B
            n1 = KernelNode("+", node.copy(), make_base_kernel(name),
                            f"({node.to_str()} + {name})")
            if n1.to_str() not in self.explored:
                results.append(n1)
            # Multiply: S -> S x B
            n2 = KernelNode("x", node.copy(), make_base_kernel(name),
                            f"({node.to_str()} x {name})")
            if n2.to_str() not in self.explored:
                results.append(n2)
            # Replace: B -> B' (only for base nodes)
            if node.op == "base" and name != node.name:
                n3 = make_base_kernel(name)
                if n3.to_str() not in self.explored:
                    results.append(n3)
        return results

    def search(self):
        """执行贪心搜索"""
        print("\n[Step 1] 初始化: 评估所有基核")
        self.initialize()
        best = max(self.candidates, key=lambda x: x[1])
        history = [(0, best[0].to_str(), best[1])]

        for depth in range(1, self.max_depth + 1):
            best_node, best_sc, _ = max(self.candidates, key=lambda x: x[1])
            new_nodes = self._expand(best_node)
            if not new_nodes:
                break
            evaluated = []
            for nn in new_nodes:
                self.explored.add(nn.to_str())
                log_ml, ok = eval_kernel(nn, self.Xt, self.yt)
                n_params = 3 + nn.to_str().count("+") + nn.to_str().count("x") * 2
                n_params = max(n_params, 3)
                sc = bic_score(log_ml, n_params, self.n)
                evaluated.append((sc, log_ml, nn, ok))
                if self.verbose:
                    print(f"  [{depth}] {nn.to_str():42s} logML={log_ml:+7.2f} BIC={sc:+7.2f}")
            for sc, lm, nn, ok in evaluated:
                self.candidates.append((nn, sc, ok))
            cur_best = max(evaluated, key=lambda x: x[0])
            history.append((depth, cur_best[2].to_str(), cur_best[0]))

        overall = max(self.candidates, key=lambda x: x[1])
        return overall, history


# ============================================================
# 4. 执行搜索
# ============================================================
print("\n[启动] 正在执行组合核搜索...")
search = KernelSearch(X_train, y_train_centered, max_depth=6, verbose=True)
(best_node, best_bic, best_sklearn_kernel), history = search.search()

print("\n" + "=" * 60)
print("搜索完成! 最佳核结构 (BIC={:.2f}):".format(best_bic))
print("=" * 60)
print(f"  表达式: {best_node.to_str()}")
if best_sklearn_kernel is not None:
    print(f"  参数:\n{best_sklearn_kernel}")

print("\n搜索历史:")
print(f"  {'深':>3s} | {'表达式':<40s} | {'BIC':>8s}")
print(f"  {'-'*55}")
for d, expr, sc in history:
    print(f"  {d:>3d} | {expr:<40s} | {sc:>+8.2f}")

# ============================================================
# 5. 用最佳核训练最终模型
# ============================================================
print("\n[训练] 最终模型...")
final_kernel = best_node.to_sklearn() if best_sklearn_kernel is None else best_sklearn_kernel
gpr_final = GaussianProcessRegressor(
    kernel=final_kernel, n_restarts_optimizer=10,
    random_state=42, normalize_y=False)
gpr_final.fit(X_train, y_train_centered)

print(f"对数边际似然: {gpr_final.log_marginal_likelihood_value_:.3f}")
print(f"最终核:\n{gpr_final.kernel_}")

# ============================================================
# 6. 预测与评估
# ============================================================
y_pred_centered, y_std = gpr_final.predict(X_test, return_std=True)
y_pred = y_pred_centered + y_mean

rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
r2   = float(r2_score(y_test, y_pred))
lower = y_pred - 1.96 * y_std
upper = y_pred + 1.96 * y_std
coverage = float(np.mean((y_test >= lower) & (y_test <= upper)))

print("\n" + "=" * 60)
print("测试集性能评估 (1990年后)")
print("=" * 60)
print(f"  RMSE          = {rmse:.4f} ppm")
print(f"  R\u00b2           = {r2:.4f}")
print(f"  95% CI 覆盖率  = {coverage*100:.1f}%")

# ============================================================
# 7. 可视化 (参考原文 Figure 3-4)
# ============================================================
plt.figure(figsize=(14, 10))

# 子图1: 预测
plt.subplot(3, 1, 1)
plt.plot(year_test, y_test, "k.", markersize=3, label="真实值")
plt.plot(year_test, y_pred, "r-", linewidth=2, label=f"GPR (RMSE={rmse:.2f} ppm)")
plt.fill_between(year_test, lower, upper, alpha=0.15, color="red", label="95% CI")
plt.axvline(1990, color="blue", ls="--", alpha=0.5, label="Train/Test split")
plt.ylabel("CO2 (ppm)")
plt.title(f"组合核搜索最佳核: {best_node.to_str()}")
plt.legend(fontsize=9); plt.grid(alpha=0.3)

# 子图2: 残差
plt.subplot(3, 1, 2)
res = y_test - y_pred
plt.plot(year_test, res, "b.", markersize=3, alpha=0.6)
plt.axhline(0, color="k", lw=1)
plt.fill_between(year_test, -1.96*y_std, 1.96*y_std, alpha=0.15, color="blue", label="\u00b11.96\u03c3")
plt.ylabel("Residual (ppm)")
plt.title(f"残差 (RMSE={rmse:.2f}, 覆盖率={coverage*100:.1f}%)")
plt.legend(fontsize=9); plt.grid(alpha=0.3)

# 子图3: 搜索历史
plt.subplot(3, 1, 3)
depths = [h[0] for h in history]
scores = [h[2] for h in history]
labels = [(h[1][:32]+"...") if len(h[1])>32 else h[1] for h in history]
plt.plot(depths, scores, "bo-", lw=2, ms=6)
for d, s, lab in zip(depths, scores, labels):
    plt.annotate(lab, (d, s), xytext=(5, 8), fontsize=6.5, alpha=0.7)
plt.xlabel("搜索深度")
plt.ylabel("BIC")
plt.title("搜索历史: 核结构逐步优化")
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("compositional_kernel_search_result.png", dpi=150)
print("[图] 已保存 compositional_kernel_search_result.png")
plt.close()

# ============================================================
# 8. 基线对比 (参考原文 Section 7.1, Figure 7)
# ============================================================
print("\n[对比] 与基线方法对比")
baselines = [
    ("Linear Regression", None),
    ("RBF (SE only)",     C(1.0) * RBF(10.0)),
    ("Periodic (Per)",    C(1.0) * ExpSineSquared(1.0, periodicity=1.0)),
    ("SE + Per",          C(1.0) * RBF(10.0) + C(1.0) * ExpSineSquared(1.0, periodicity=1.0)),
    ("SE x Per",          C(1.0) * RBF(10.0) * ExpSineSquared(1.0, periodicity=1.0)),
]

lr = LinearRegression()
lr.fit(X_train, y_train_centered)
results = [("Linear Regression", float(np.sqrt(mean_squared_error(y_test, lr.predict(X_test) + y_mean))))]

for name, kernel in baselines:
    if kernel is None or name == "Linear Regression":
        continue
    try:
        g = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)
        g.fit(X_train, y_train_centered)
        yp = g.predict(X_test) + y_mean
        r = float(np.sqrt(mean_squared_error(y_test, yp)))
        results.append((name, r))
    except Exception as e:
        results.append((name, float("inf")))

results.append((f"Compositional Search", rmse))

print(f"  {'方法':<35s} {'RMSE':>8s} {'改进':>8s}")
print(f"  {'-'*53}")
base_rmse = results[0][1]
for name, r in results:
    pct = f"{(base_rmse/r-1)*100:+5.1f}%" if r < 1e10 else "  N/A"
    mark = " <<<" if "Search" in name else ""
    print(f"  {name:<35s} {r:>8.4f} {pct:>8s}{mark}")

# ============================================================
# 9. 保存结果
# ============================================================
np.savez("compositional_search_results.npz",
         best_kernel_str=best_node.to_str(),
         y_test=y_test, y_pred=y_pred, y_std=y_std,
         rmse=rmse, r2=r2, coverage=coverage,
         history_depth=[h[0] for h in history],
         history_expr=[h[1] for h in history],
         history_bic=[h[2] for h in history])

print("\n[结果] 已保存至 compositional_search_results.npz")
print("=" * 60)
print("完成! 参考: Duvenaud et al. ICML 2013")
print("  Structure Discovery in Nonparametric Regression")
print("  through Compositional Kernel Search")
print("=" * 60)