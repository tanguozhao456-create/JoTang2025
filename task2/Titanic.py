import pandas as pd
import sklearn.preprocessing as preprocessing
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 导入训练集
train = pd.read_csv('train.csv')
print("缺失值情况：")
missing_values = train.isnull().sum()
print(missing_values)

# 训练集预处理
train = train.dropna(subset=['Embarked'])
train['Age'] = train.groupby('Sex')['Age'].transform(lambda x: x.fillna(x.mode()[0]))
Sex_dummies = pd.get_dummies(train['Sex'], prefix='Sex', dtype=int)
Embarked_dummies = pd.get_dummies(train['Embarked'], prefix='Embarked', dtype=int)
train = train.drop('PassengerId', axis=1)
train = train.drop('Name', axis=1)
train = train.drop('Cabin', axis=1)
train = train.drop('Ticket', axis=1)
train = train.drop('Sex', axis=1)
train = train.drop('Embarked', axis=1)
train = pd.concat([train, Sex_dummies, Embarked_dummies], axis=1)
print(train.info())

# 归一化处理
scaler = preprocessing.StandardScaler()
train[['Age', 'Fare']] = scaler.fit_transform(train[['Age', 'Fare']])
print(train)

# 测试集预处理
test = pd.read_csv('test.csv')
test = test.dropna(subset=['Fare'])
test1 = test.dropna(subset=['Fare'])
test['Age'] = test.groupby('Sex')['Age'].transform(lambda x: x.fillna(x.mode()[0]))
Sex_dummies = pd.get_dummies(test['Sex'], prefix='Sex', dtype=int)
Embarked_dummies = pd.get_dummies(test['Embarked'], prefix='Embarked', dtype=int)
test = test.drop('PassengerId', axis=1)
test = test.drop('Name', axis=1)
test = test.drop('Cabin', axis=1)
test = test.drop('Ticket', axis=1)
test = test.drop('Sex', axis=1)
test = test.drop('Embarked', axis=1)
test = pd.concat([test, Sex_dummies, Embarked_dummies], axis=1)

# 使用训练集的scaler进行转换
test[['Age', 'Fare']] = scaler.transform(test[['Age', 'Fare']])

# 准备训练数据
train_df = train.filter(regex='Survived|Age|SibSp|Parch|Fare|Sex_.*|Embarked_.*|Pclass')
train_np = train_df.values
X = train_np[:, 1:]
y = train_np[:, 0]

# 将数据转换为PyTorch张量
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1) 

# 创建数据集和数据加载器
dataset = TensorDataset(X_tensor, y_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# 定义神经网络模型
class TitanicModel(nn.Module):
    def __init__(self, input_size):
        super(TitanicModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc3(x))
        return x

# 初始化模型
input_size = X.shape[1]
model = TitanicModel(input_size)

# 定义损失函数和优化器
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 100
best_val_loss = float('inf')
best_val_accuracy = 0.0
patience = 10
counter = 0

print("开始训练模型...")
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # 验证
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    # 添加准确率计算
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            # 计算准确率
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    val_accuracy = correct / total 
    
    # 每10个epoch打印一次损失和准确率
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
    
    # 保存最佳模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_val_accuracy = val_accuracy
        counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"保存最佳模型: Val Loss={val_loss:.4f}, Val Accuracy={val_accuracy:.4f}")
    else:
        counter += 1
        if counter >= patience:
            print(f'早停触发于第 {epoch+1} 轮')
            break

print("模型训练完成")

# 加载最佳模型
model.load_state_dict(torch.load('best_model.pth'))
print(f"加载最佳模型: Val Loss={best_val_loss:.4f}, Val Accuracy={best_val_accuracy:.4f}")

# 合并训练集和验证集重新训练
print("\n重新训练模型（使用全部数据）...")
full_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 重置优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(20):  # 训练少量epoch
    model.train()
    for inputs, labels in full_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    # 输出重新训练进度
    if (epoch+1) % 5 == 0:
        print(f'重新训练轮次 [{epoch+1}/20]')

# 保存最终模型
torch.save(model.state_dict(), 'final_model.pth')
print("最终模型 'final_model.pth'")

# 在测试集上进行预测
test_df = test.filter(regex='Age|SibSp|Parch|Fare|Sex_.*|Embarked_.*|Pclass')
test_data = test_df.values
test_tensor = torch.tensor(test_data, dtype=torch.float32)

model.eval()
with torch.no_grad():
    test_outputs = model(test_tensor)
    nn_predictions = (test_outputs > 0.5).float().squeeze().numpy().astype(np.int32)

# 保存神经网络预测结果
nn_result = pd.DataFrame({'PassengerId': test1['PassengerId'].values, 'Survived': nn_predictions})
print("\n神经网络预测结果:")
print(nn_result.head(20))

# 评估模型性能
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in val_loader:
        outputs = model(inputs)
        preds = (outputs > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 计算评估指标
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
conf_matrix = confusion_matrix(all_labels, all_preds)

print(f'\n模型评估结果:')
print(f'准确率: {accuracy:.4f}')
print(f'精确率: {precision:.4f}')
print(f'召回率: {recall:.4f}')
print(f'F1分数: {f1:.4f}')
print(f'混淆矩阵:\n{conf_matrix}')