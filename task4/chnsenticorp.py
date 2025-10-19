import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
import random
import json                     

# 关键超参数
hyperparams = {
    "seed": 42,
    "learning_rate": 3e-5,
    "epochs": 5,
    "batch_size": 16,
    "max_length": 128,
    "weight_decay": 0.01
}

# 随机种子
seed = hyperparams["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# 加载并预处理数据
file_path = 'data/ChnSentiCorp_htl_all.csv'
data = pd.read_csv(file_path, encoding='utf-8')
data['review'] = data['review'].fillna('')
non_empty_mask = data['review'].str.strip() != ''
data = data[non_empty_mask].reset_index(drop=True)
print(f"处理后样本数量: {len(data)}")

# 划分训练集和验证集
train_data, val_data = train_test_split(
    data, 
    test_size=0.2, 
    random_state=seed,
    stratify=data['label']
)
print(f"训练集大小: {len(train_data)}")
print(f"验证集大小: {len(val_data)}")

# 创建自定义数据集类
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 加载分词器和模型
model_path = 'models' 
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(
    model_path,
    num_labels=2
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"模型已加载到: {device}")

# 创建数据集对象
train_dataset = SentimentDataset(
    train_data['review'].tolist(),
    train_data['label'].tolist(),
    tokenizer,
    max_length=hyperparams["max_length"]
)
val_dataset = SentimentDataset(
    val_data['review'].tolist(),
    val_data['label'].tolist(),
    tokenizer,
    max_length=hyperparams["max_length"]
)

# 设置训练参数 
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=hyperparams["epochs"],
    per_device_train_batch_size=hyperparams["batch_size"],
    per_device_eval_batch_size=hyperparams["batch_size"],
    learning_rate=hyperparams["learning_rate"],
    weight_decay=hyperparams["weight_decay"],
    evaluation_strategy='epoch',  
    save_strategy='epoch',       
    logging_dir='./logs',
    logging_steps=50,
    seed=seed,
    load_best_model_at_end=True,
    metric_for_best_model='f1_macro',
    greater_is_better=True,
    report_to='none'
)

# 定义评估函数
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # 计算准确率
    accuracy = accuracy_score(labels, predictions)
    
    # 计算F1分数
    f1_macro = f1_score(labels, predictions, average='macro')
    
    # 计算F1分数
    f1_binary = f1_score(labels, predictions, average='binary')
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_binary': f1_binary
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)
train_results = trainer.train()

# 保存最佳模型
best_model_path = './best_model'
trainer.save_model(best_model_path)
tokenizer.save_pretrained(best_model_path)

# 在验证集上评估最终模型
eval_results = trainer.evaluate(val_dataset)

print(f"准确率: {eval_results['eval_accuracy']:.4f}")
print(f"宏平均F1: {eval_results['eval_f1_macro']:.4f}")
print(f"二分类F1: {eval_results['eval_f1_binary']:.4f}")

# 保存评估结果
with open('evaluation_results.json', 'w', encoding='utf-8') as f:
    json.dump(eval_results, f, ensure_ascii=False, indent=2)