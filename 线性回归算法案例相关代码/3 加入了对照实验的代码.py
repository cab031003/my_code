import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 设置随机种子和中文显示
np.random.seed(42)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 数据生成 ====================
disciple_count = np.random.randint(50, 200, size=20)
master_skill = 3 * disciple_count + np.random.randn(20) * 20 + 100

# ==================== 数据预处理 ====================
X = disciple_count.reshape(-1, 1)
y = np.where(master_skill > 600, 600, master_skill)  # 异常值处理

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y_scaled = (y - np.mean(y)) / np.std(y)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.3, random_state=42
)

# ==================== 模型训练 ====================
# 对原始数据进行模型训练
lin_reg_original = LinearRegression()
lin_reg_original.fit(X, y)


# 基准模型（普通线性回归）
lin_reg = LinearRegression(fit_intercept=True)
lin_reg.fit(X_train, y_train)

# 对照模型（岭回归）
ridge_reg = Ridge(alpha=1.0)
ridge_reg.fit(X_train, y_train)

# ==================== 模型评估 ====================
def evaluate_model(model, X, y_true):
    y_pred = model.predict(X)
    return {
        'MSE': mean_squared_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred)
    }

# 反标准化函数
def inverse_scale(y_scaled):
    return y_scaled * np.std(y) + np.mean(y)

print("=== 基准模型（线性回归） ===")
lin_metrics = evaluate_model(lin_reg, X_test, y_test)
print(f"MSE: {lin_metrics['MSE']:.4f}")
print(f"R2: {lin_metrics['R2']:.4f}")

print("\n=== 对照模型（岭回归） ===")
ridge_metrics = evaluate_model(ridge_reg, X_test, y_test)
print(f"MSE: {ridge_metrics['MSE']:.4f}")
print(f"R2: {ridge_metrics['R2']:.4f}")

# ==================== 可视化 ====================
plt.figure(figsize=(15, 5))

# 原始数据与拟合结果
plt.subplot(1, 3, 1)
X_orig = scaler.inverse_transform(X_scaled)
plt.scatter(X_orig, y, color='blue', label='原始数据')
plt.title("原始数据分布")
y_pred_original = lin_reg_original.predict(X)
plt.plot(X, y_pred_original, color='orange', linewidth=2, label='原始回归直线')
plt.xlabel("弟子数量")
plt.ylabel("武功修为")
plt.grid(True)


# 线性回归结果
plt.subplot(1, 3, 2)
y_pred_lin = inverse_scale(lin_reg.predict(X_scaled))
plt.scatter(X_orig, y, color='blue', alpha=0.3, label='真实值')
plt.plot(X_orig, y_pred_lin, 'r-', label='线性回归')
plt.title(f"线性回归 (R2={lin_metrics['R2']:.3f})")
plt.xlabel("弟子数量")
plt.ylabel("武功修为")
plt.grid(True)
plt.legend()

# 岭回归结果
plt.subplot(1, 3, 3)
y_pred_ridge = inverse_scale(ridge_reg.predict(X_scaled))
plt.scatter(X_orig, y, color='blue', alpha=0.3, label='真实值')
plt.plot(X_orig, y_pred_ridge, 'g-', label='岭回归')
plt.title(f"岭回归 (R2={ridge_metrics['R2']:.3f})")
plt.xlabel("弟子数量")
plt.ylabel("武功修为")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

