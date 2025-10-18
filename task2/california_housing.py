# 导入必要的库
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 加载数据集
california = fetch_california_housing()
X = california.data
y = california.target 

# 查看数据形状
print("特征形状:", X.shape)
print("目标值形状:", y.shape) 

# 拆分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建并训练线性回归模型
model = LinearRegression()
model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = model.predict(X_test)

# 计算评估指标
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\n均方误差(MSE): {mse:.4f}")
print(f"决定系数(R²): {r2:.4f}")

# 绘制预测值与真实值的散点图
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, edgecolors='k')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2) 

plt.xlabel('真实房价 (万美元)')
plt.ylabel('预测房价 (万美元)')
plt.grid(True)
plt.show()