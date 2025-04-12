import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 设置默认字体为中文字体，例如SimHei
plt.rcParams['font.sans-serif'] = ['SimHei']

# 生成模拟数据
np.random.seed(42)
disciple_count = np.random.randint(50, 200, size=20)
master_skill = 3 * disciple_count + np.random.randn(20) * 20 + 100

# 数据转换为二维数组, 用于训练.
# 这里我们假设每个样本只有一个特征，即每个样本只有一个属性。
# 如果每个样本有多个特征，则需要将特征转换为二维数组。
# -1: 表示自动计算数组的行数，-1表示自动计算数组的列数。
X = disciple_count.reshape(-1, 1)
y = master_skill

# 创建线性回归模型并训练
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# 指定字体
plt.rcParams['font.family'] = 'Arial'

y_pred = lin_reg.predict(X)

# 计算残差
residuals = y - y_pred

# 绘图
plt.figure(figsize=(12, 10))

# 子图1: 散点图与回归线
plt.subplot(2, 2, 1)
plt.scatter(X, y, color='blue', label='Actual Data', alpha=0.6)
plt.plot(X, y_pred, color='red', linewidth=2, label='Regression Line')
plt.title('Scatter Plot with Regression Line')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()

# 子图2: 残差图
plt.subplot(2, 2, 2)
plt.scatter(X, residuals, color='green', alpha=0.6)
plt.hlines(y=0, xmin=min(X), xmax=max(X), color='red', linestyle='--')
plt.title('Residual Plot')
plt.xlabel('X')
plt.ylabel('Residuals')

# 子图3: 预测值与实际值对比
plt.subplot(2, 2, 3)
plt.scatter(y, y_pred, color='purple', alpha=0.6)
plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--', label='Perfect Fit')
plt.title('Predicted vs Actual')
plt.xlabel('Actual y')
plt.ylabel('Predicted y')
plt.legend()

# 子图4: 残差直方图
plt.subplot(2, 2, 4)
plt.hist(residuals, bins=20, color='orange', edgecolor='black', alpha=0.7)
plt.title('Residuals Distribution')
plt.xlabel('Residuals')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()