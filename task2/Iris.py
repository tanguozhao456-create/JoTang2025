from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import numpy as np

iris = load_iris()

# 导入数据集
X = iris.data
y = iris.target

# # 将numpy矩阵转化为torch矩阵
X = torch.tensor(X, dtype = torch.float)
y = torch.tensor(y, dtype = torch.long)

# print(type(X))
print(f'特征矩阵的形状:\n{X.shape}\n')
print(f'输出矩阵的形状:\n{y.shape}\n')
print(f'输出的值:\n{y}\n')

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

model = BC(4, 3, 5)

# # 定义损失函数
loss_f = nn.CrossEntropyLoss()
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
        train_preds = torch.argmax(model(X_train), dim=1)
        train_acc = (train_preds == y_train).float().mean()
        
        # 计算测试集准确率
        test_preds = torch.argmax(model(X_test), dim=1)
        test_acc = (test_preds == y_test).float().mean()

print(f'训练准确度: {train_acc.item()*100:.2f}%')
print(f'测试准确度: {test_acc.item()*100:.2f}%')



# 转换为numpy数组
y_true = y_test.numpy()
y_pred = test_preds.numpy()

# 计算混淆矩阵
cm = confusion_matrix(y_true, y_pred)

# 打印混淆矩阵
print("\n混淆矩阵:")
print(cm)