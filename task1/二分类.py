import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

# 修改字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
noise = 0.2

# 创建数据集
X, y = make_moons(n_samples=1000, noise=noise, random_state=42)

# 数据集可视化
plt.figure(figsize=(6, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='k')
plt.title("数据集 (噪声=" + str(noise) + ")")
plt.show()

# 数据预处理
# 将数据转换为PyTorch张量
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)
y_tensor = y_tensor.unsqueeze(1)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_tensor, y_tensor, test_size=0.25, random_state=42
)

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(2, 10)
        self.layer2 = nn.Linear(10, 10)
        self.layer3 = nn.Linear(10, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        x = self.sigmoid(x)
        
        return x

model = SimpleNN()

# 损失函数
# 二元交叉熵损失，用于二分类问题
criterion = nn.BCELoss()

# 优化器
optimizer = optim.SGD(model.parameters(), lr=0.1)

train_losses = []
train_accs = []
test_losses = []
test_accs = []

num_epochs = 1000

for epoch in range(num_epochs):
    # 模型训练
    model.train()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    predictions = outputs > 0.5
    predictions = predictions.float()
    
    # 计算预测正确的数量
    correct = (predictions == y_train).float()
    accuracy = correct.mean().item()
    
    # 记录训练损失和准确率
    train_losses.append(loss.item())
    train_accs.append(accuracy)
    
    # 模型测试
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
        
        test_predictions = test_outputs > 0.5
        test_predictions = test_predictions.float()

        # 计算测试正确的数量
        test_correct = (test_predictions == y_test).float()
        test_accuracy = test_correct.mean().item()
        
        # 记录测试损失和准确率
        test_losses.append(test_loss.item())
        test_accs.append(test_accuracy)

    if (epoch + 1) % 100 == 0:
        print("轮次 [{}/{}], 损失: {:.4f}, 训练准确率: {:.4f}, 测试准确率: {:.4f}".format(
            epoch + 1, num_epochs, loss.item(), accuracy, test_accuracy
        ))

# 绘制训练曲线
plt.figure(figsize=(12, 5))

# 训练损失下降曲线
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="训练损失")
plt.plot(test_losses, label="测试损失")
plt.xlabel("训练轮次")
plt.ylabel("损失值")
plt.title("损失曲线")
plt.legend()

# 绘制准确率曲线
plt.subplot(1, 2, 2)
plt.plot(train_accs, label="训练准确率")
plt.plot(test_accs, label="测试准确率")
plt.xlabel("训练轮次")
plt.ylabel("准确率")
plt.title("准确率曲线")
plt.legend()

plt.tight_layout()
plt.show()

# 绘制决策边界
h = 0.02
x_min = X[:, 0].min() - 0.5
x_max = X[:, 0].max() + 0.5
y_min = X[:, 1].min() - 0.5
y_max = X[:, 1].max() + 0.5

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), 
                     np.arange(y_min, y_max, h))

grid_points = np.c_[xx.ravel(), yy.ravel()]
grid_tensor = torch.tensor(grid_points, dtype=torch.float32)

with torch.no_grad():
    model.eval()
    Z = model(grid_tensor).numpy()
    Z = Z.reshape(xx.shape)


plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
plt.colorbar(label="类别1的概率")
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.RdYlBu)
plt.title("决策边界 (noise=" + str(noise) + ")")
plt.xlabel("特征1")
plt.ylabel("特征2")
plt.show()

final_test_accuracy = test_accs[-1]
print("最终测试准确率: {:.4f}".format(final_test_accuracy))