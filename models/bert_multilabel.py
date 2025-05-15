from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, hamming_loss
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os

# ==== 配置 ====
data_path = "../data/split_encoded_train/mapped_goemotions_1.csv"
output_dir = "./bert_finetuned_output"

# ==== 加载数据并预处理 ====
df = pd.read_csv(data_path)
label_cols = [col for col in df.columns if col not in ["id", "text", "sentiment", "ekman_emotion"]]
df = df.dropna(subset=["text"])
df["text"] = df["text"].astype(str)

texts = df["text"].tolist()
labels = df[label_cols].values

train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# ==== 自定义数据集 ====
class GoEmotionsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_len)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)

# ==== 加载模型和分词器 ====
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(label_cols), problem_type="multi_label_classification")

# ==== 构造数据集 ====
train_dataset = GoEmotionsDataset(train_texts, train_labels, tokenizer)
val_dataset = GoEmotionsDataset(val_texts, val_labels, tokenizer)

# ==== 训练参数配置 ====
training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch",  # 旧版参数名（v3.x 或更早）
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir=f"{output_dir}/logs",
    logging_steps=10,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)

# ==== 评估指标 ====
def compute_metrics(pred):
    logits, labels = pred
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    y_pred = (probs >= 0.5).astype(int)
    report = classification_report(labels, y_pred, output_dict=True, zero_division=0)
    return {
        "micro_f1": report["micro avg"]["f1-score"],
        "macro_f1": report["macro avg"]["f1-score"],
        "samples_f1": report["samples avg"]["f1-score"],
        "hamming_loss": hamming_loss(labels, y_pred)
    }

# ==== 开始训练 ====
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
