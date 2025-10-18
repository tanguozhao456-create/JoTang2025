import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import sklearn.preprocessing as preprocessing
import numpy as np
from sympy.series import S
from sklearn import linear_model

# 导入训练集
train = pd.read_csv('F:/大学/A大二上/2025焦糖工作室招新/task2/train.csv')
# print(train)
# print(train.info())
# print(train.describe())

# 查看缺失值情况
print("缺失值情况：")
missing_values = train.isnull().sum()
print(missing_values)

# 数据预处理
# 补充缺失值
# 填充Age，用相同Sex的Age的众数进行填充
train = train.dropna(subset=['Embarked'])
train['Age'] = train.groupby('Sex')['Age'].transform(lambda x: x.fillna(x.mode()[0]))
Sex_dummies = pd.get_dummies(train['Sex'], prefix= 'Sex', dtype=int)
Embarked_dummies = pd.get_dummies(train['Embarked'], prefix= 'Embarked', dtype=int)
train = train.drop('PassengerId', axis=1)
train = train.drop('Name', axis=1)
train = train.drop('Cabin', axis=1)
train = train.drop('Ticket', axis=1)
train = train.drop('Sex', axis=1)
train = train.drop('Embarked', axis=1)
train = pd.concat([train, Sex_dummies, Embarked_dummies], axis=1)
print(train.info())

scaler = preprocessing.StandardScaler()
train['Age'] = scaler.fit_transform(train[['Age']])
train['Fare'] = scaler.fit_transform(train[['Fare']])
print(train)

# 测试集预处理
test = pd.read_csv('F:/大学/A大二上/2025焦糖工作室招新/task2/test.csv')
test = test.dropna(subset=['Fare'])
test1 = test.dropna(subset=['Fare'])
test['Age'] = test.groupby('Sex')['Age'].transform(lambda x: x.fillna(x.mode()[0]))
Sex_dummies = pd.get_dummies(test['Sex'], prefix= 'Sex', dtype=int)
Embarked_dummies = pd.get_dummies(test['Embarked'], prefix= 'Embarked', dtype=int)
test = test.drop('PassengerId', axis=1)
test = test.drop('Name', axis=1)
test = test.drop('Cabin', axis=1)
test = test.drop('Ticket', axis=1)
test = test.drop('Sex', axis=1)
test = test.drop('Embarked', axis=1)
test = pd.concat([test, Sex_dummies, Embarked_dummies], axis=1)

scaler = preprocessing.StandardScaler()
test['Age'] = scaler.fit_transform(test[['Age']])
test['Fare'] = scaler.fit_transform(test[['Fare']])


train_df = train.filter(regex='Survived|Age|SibSp|Parch|Fare|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_np = train_df.values
X = train_np[:, 1:]
y = train_np[:, 0]
print(y)

clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6,solver='liblinear')
clf.fit(X, y)
print(clf)

test_df = test.filter(regex='Survived|Age|SibSp|Parch|Fare|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
predictions = clf.predict(test_df)
result = pd.DataFrame({'PassengerId':test1['PassengerId'].values, 'Survived':predictions.astype(np.int32)})
print(result[0:20])

# 还缺优化器、损失函数、评估模型的性能