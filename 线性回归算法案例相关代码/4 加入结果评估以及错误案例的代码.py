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
        'R2': r2_score(y_true, y_pred),
        'MAE': np.mean(np.abs(y_true - y_pred))  # 新增MAE指标
    }

# 反标准化函数
def inverse_scale(y_scaled, y_original):
    return y_scaled * np.std(y_original) + np.mean(y_original)

print("=== 基准模型（线性回归） ===")
lin_metrics = evaluate_model(lin_reg, X_test, y_test)
print(f"截距: {lin_reg.intercept_:.4f}")
print(f"系数: {lin_reg.coef_[0]:.4f}")
print(f"MSE: {lin_metrics['MSE']:.4f}")
print(f"R2: {lin_metrics['R2']:.4f}")
print(f"MAE: {lin_metrics['MAE']:.4f}")


print("\n=== 对照模型（岭回归） ===")
ridge_metrics = evaluate_model(ridge_reg, X_test, y_test)
print(f"截距: {ridge_reg.intercept_:.4f}")
print(f"系数: {ridge_reg.coef_[0]:.4f}")
print(f"MSE: {ridge_metrics['MSE']:.4f}")
print(f"R2: {ridge_metrics['R2']:.4f}")
print(f"MAE: {ridge_metrics['MAE']:.4f}")

# ==================== 可视化 ====================
plt.figure(figsize=(18, 6))

# 原始数据与拟合结果
plt.subplot(1, 3, 1)
X_orig = scaler.inverse_transform(X_scaled)

plt.scatter(X, y, color='blue', label='原始数据')
plt.title("原始数据分布")
y_pred_original = lin_reg_original.predict(X)
plt.plot(X_orig, y_pred_original, color='orange', linewidth=2, label='原始回归直线')
plt.xlabel("弟子数量")
plt.ylabel("武功修为")
plt.grid(True)

# 线性回归结果
plt.subplot(1, 3, 2)
y_pred_lin = inverse_scale(lin_reg.predict(X_scaled), y)
plt.scatter(X_orig, y, color='blue', alpha=0.3, label='真实值')
plt.plot(X_orig, y_pred_lin, 'r-', linewidth=2, label='线性回归')
plt.title(f"线性回归 (R2={lin_metrics['R2']:.3f})")
plt.xlabel("弟子数量")
plt.ylabel("武功修为")
plt.legend()
plt.grid(True)

# 岭回归结果
plt.subplot(1, 3, 3)
y_pred_ridge = inverse_scale(ridge_reg.predict(X_scaled), y)
plt.scatter(X_orig, y, color='blue', alpha=0.3, label='真实值')
plt.plot(X_orig, y_pred_ridge, 'g-', linewidth=2, label='岭回归')
plt.title(f"岭回归 (R2={ridge_metrics['R2']:.3f})")
plt.xlabel("弟子数量")
plt.ylabel("武功修为")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()



# ==================== 错误案例分析 ====================
print("\n=== 错误案例分析 ===")

# 计算所有预测值
y_pred_all_lin = inverse_scale(lin_reg.predict(X_scaled), y)
y_pred_all_ridge = inverse_scale(ridge_reg.predict(X_scaled), y)

# 找出误差最大的样本
max_err_idx_lin = np.argmax(np.abs(y - y_pred_all_lin))
max_err_idx_ridge = np.argmax(np.abs(y - y_pred_all_ridge))

print("\n【线性回归】最大误差样本：")
print(f"- 弟子数量: {X_orig[max_err_idx_lin][0]:.0f}")
print(f"- 真实修为: {y[max_err_idx_lin]:.1f}")
print(f"- 预测修为: {y_pred_all_lin[max_err_idx_lin]:.1f}")
print(f"- 绝对误差: {np.abs(y[max_err_idx_lin] - y_pred_all_lin[max_err_idx_lin]):.1f}")

print("\n【岭回归】最大误差样本：")
print(f"- 弟子数量: {X_orig[max_err_idx_ridge][0]:.0f}")
print(f"- 真实修为: {y[max_err_idx_ridge]:.1f}")
print(f"- 预测修为: {y_pred_all_ridge[max_err_idx_ridge]:.1f}")
print(f"- 绝对误差: {np.abs(y[max_err_idx_ridge] - y_pred_all_ridge[max_err_idx_ridge]):.1f}")


