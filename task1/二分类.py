from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.nn.modules import BCEWithLogitsLoss
import torch.optim as optim
import matplotlib.pyplot as plt

num_samples = 100

# 导入数据集
X, y = make_moons(num_samples, noise=0, random_state=42)

# 将numpy矩阵转化为torch矩阵
X = torch.tensor(X, dtype = torch.float)
y = torch.tensor(y, dtype = torch.float).unsqueeze(1)

# 划分训练集、测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

class BC(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens):
        super().__init__()

        self.layer1 = nn.Linear(num_inputs, num_hiddens)
        self.layer2 = nn.Linear(num_hiddens, num_hiddens)
        self.layer3 = nn.Linear(num_hiddens, num_hiddens)
        self.layer4 = nn.Linear(num_hiddens, num_outputs)

    def relu(self, x):
        M = torch.zeros_like(x)
        return torch.max(x, M)
  
    def forward(self, x):
        z1 = self.relu(self.layer1(x))
        z2 = self.relu(self.layer2(z1))
        z3 = self.relu(self.layer3(z2))
        y = self.layer4(z3)
        return y

model = BC(2, 1, 10)

# 定义损失函数
loss_f = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params = model.parameters(), lr = 0.1)

# 训练循环
train_losses = []
test_losses = []

epochs = 1000
for epoch in range(epochs):
    # 训练阶段
    model.train()
    optimizer.zero_grad()
    
    # 前向传播
    outputs = model(X_train)
    loss = loss_f(outputs, y_train)
    
    # 反向传播和优化
    loss.backward()
    optimizer.step()
    
    # 记录训练损失
    train_losses.append(loss.item())
    
    # 测试阶段
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = loss_f(test_outputs, y_test)
        test_losses.append(test_loss.item())
    
    # 打印进度
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], '
              f'Train Loss: {loss.item():.4f}, '
              f'Test Loss: {test_loss.item():.4f}')

plt.plot(range(epochs), train_losses, label='Train Loss')
plt.plot(range(epochs), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss Over Epochs')
plt.legend()
plt.show()

# 评估模型
model.eval()
with torch.no_grad():
    # 计算训练集准确率
    train_preds = torch.sigmoid(model(X_train)) > 0.5
    train_acc = (train_preds == y_train).float().mean()
    
    # 计算测试集准确率
    test_preds = torch.sigmoid(model(X_test)) > 0.5
    test_acc = (test_preds == y_test).float().mean()

print(f'Final Train Accuracy: {train_acc.item()*100:.2f}%')
print(f'Final Test Accuracy: {test_acc.item()*100:.2f}%')

# 可视化决策边界
def plot_decision_boundary(model, X, y):
    # 设置网格范围
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.01
    
    # 生成网格点
    xx, yy = torch.meshgrid(torch.arange(x_min, x_max, h),
                           torch.arange(y_min, y_max, h))
    
    # 预测每个网格点的类别
    with torch.no_grad():
        Z = model(torch.cat([xx.reshape(-1,1), yy.reshape(-1,1)], dim=1))
        Z = (torch.sigmoid(Z) > 0.5).reshape(xx.shape)
    
    # 绘制结果
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y.squeeze(), edgecolors='k')
    plt.title('Decision Boundary')
    plt.show()

plot_decision_boundary(model, X, y)