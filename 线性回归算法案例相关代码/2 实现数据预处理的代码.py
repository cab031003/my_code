import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 设置默认字体为中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ==================== 数据生成 ====================
np.random.seed(42)
disciple_count = np.random.randint(50, 200, size=20)
master_skill = 3 * disciple_count + np.random.randn(20) * 20 + 100

# ==================== 数据预处理 ====================
# 1. 转换为二维数组
X = disciple_count.reshape(-1, 1)
y = master_skill

# 2. 异常值处理（假设武功修为超过600为异常）
y = np.where(y > 600, 600, y)

# 3. 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y_scaled = (y - np.mean(y)) / np.std(y)

# 4. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.3, random_state=42
)

# ==================== 模型训练 ====================
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# ！！！！！（加入的代码）
# 对原始数据进行模型训练
lin_reg_original = LinearRegression()
lin_reg_original.fit(X, y)

# ==================== 结果分析 ====================
# ！！！！！（加入的代码）
print("原始数据的模型参数：")
print(f"截距: {lin_reg_original.intercept_:.4f}")
print(f"系数: {lin_reg_original.coef_[0]:.4f}")

print("标准化后的模型参数：")
print(f"截距: {lin_reg.intercept_:.4f}")
print(f"系数: {lin_reg.coef_[0]:.4f}")



# ==================== 可视化 ====================
# 创建画布
plt.figure(figsize=(12, 5))

# 原始数据散点图
plt.subplot(1, 2, 1)
plt.scatter(X, y, color='blue', label='原始数据')
plt.title("原始数据分布")
# ！！！！！（加入的代码）
y_pred_original = lin_reg_original.predict(X)
plt.plot(X, y_pred_original, color='orange', linewidth=2, label='原始回归直线')

plt.xlabel("弟子数量")
plt.ylabel("掌门武功修为")
plt.legend()

# 标准化后的回归结果
plt.subplot(1, 2, 2)
# 反标准化还原数据
X_plot = scaler.inverse_transform(X_train)
y_plot = y_train * np.std(y) + np.mean(y)
y_pred = lin_reg.predict(X_train) * np.std(y) + np.mean(y)

plt.scatter(X_plot, y_plot, color='green', label='训练数据')
plt.plot(X_plot, y_pred, color='red', linewidth=2, label='回归直线')
plt.title("预处理后的回归分析")
plt.xlabel("弟子数量（标准化后）")
plt.ylabel("武功修为（标准化后）")
plt.legend()

plt.tight_layout()
plt.show()
