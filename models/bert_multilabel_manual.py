import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast, BertModel, AdamW
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, hamming_loss
import numpy as np

# ==== 数据加载与准备 ====
df = pd.read_csv("../data/split_encoded_train/mapped_goemotions_1.csv")
label_cols = [col for col in df.columns if col not in ["id", "text", "sentiment", "ekman_emotion"]]
df = df.dropna(subset=["text"])
texts = df["text"].astype(str).tolist()
labels = df[label_cols].values.astype(np.float32)

train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# ==== 分词器 ====
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

class GoEmotionsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_len, return_tensors='pt')
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.labels[idx]
        }

train_dataset = GoEmotionsDataset(train_texts, train_labels, tokenizer)
val_dataset = GoEmotionsDataset(val_texts, val_labels, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# ==== 模型定义 ====
class BertForMultiLabel(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.classifier(pooled_output)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForMultiLabel(len(label_cols)).to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)
criterion = nn.BCEWithLogitsLoss()

# ==== 训练循环 ====
for epoch in range(3):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Train Loss: {total_loss:.4f}")

# ==== 验证 ====
model.eval()
all_preds, all_targets = [], []
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].cpu().numpy()

        logits = model(input_ids, attention_mask).cpu().numpy()
        probs = torch.sigmoid(torch.tensor(logits)).numpy()
        preds = (probs >= 0.5).astype(int)

        all_preds.append(preds)
        all_targets.append(labels)

y_pred = np.vstack(all_preds)
y_true = np.vstack(all_targets)

print("Micro F1:", f1_score(y_true, y_pred, average='micro'))
print("Macro F1:", f1_score(y_true, y_pred, average='macro'))
print("Hamming Loss:", hamming_loss(y_true, y_pred))
