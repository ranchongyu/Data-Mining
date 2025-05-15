import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import BertTokenizer, BertModel
import torch
from tqdm import tqdm

# ==== 路径与参数 ====
data_path = "../data/split_encoded_train/mapped_goemotions_1.csv"
max_tfidf_features = 3000
max_bert_len = 64
bert_model_name = "bert-base-uncased"

# ==== 加载数据 ====
df = pd.read_csv(data_path)
label_cols = [col for col in df.columns if col not in ["id", "text", "sentiment", "ekman_emotion"]]
df.dropna(subset=["text"], inplace=True)
df["text"] = df["text"].astype(str)

X_train, X_test, y_train, y_test = train_test_split(df["text"], df[label_cols], test_size=0.2, random_state=42)

# ==== 提取 BERT [CLS] 向量 ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
bert = BertModel.from_pretrained(bert_model_name)
bert.to(device)
bert.eval()

def extract_bert_embeddings(texts):
    cls_embeddings = []
    with torch.no_grad():
        for text in tqdm(texts, desc="BERT encoding"):
            encoded = tokenizer(text, padding="max_length", truncation=True, max_length=max_bert_len, return_tensors="pt")
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            output = bert(input_ids=input_ids, attention_mask=attention_mask)
            cls_vec = output.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
            cls_embeddings.append(cls_vec)
    return np.array(cls_embeddings)

X_train_bert = extract_bert_embeddings(X_train)
X_test_bert = extract_bert_embeddings(X_test)

# ==== 提取 TF-IDF 特征 ====
vectorizer = TfidfVectorizer(max_features=max_tfidf_features)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ==== 拼接 BERT + TF-IDF ====
from scipy.sparse import hstack
X_train_combined = hstack([X_train_tfidf, X_train_bert])
X_test_combined = hstack([X_test_tfidf, X_test_bert])

# ==== 模型训练 ====
model = OneVsRestClassifier(LinearSVC())
model.fit(X_train_combined, y_train)

# ==== 评估 ====
y_pred = model.predict(X_test_combined)
report = classification_report(y_test, y_pred, target_names=label_cols, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv("../results/fusion_bert_tfidf_report.csv")

# ==== 可视化 ====
import matplotlib.pyplot as plt
f1_scores = report_df.loc[label_cols, "f1-score"].sort_values(ascending=False)
plt.figure(figsize=(12, 6))
f1_scores.plot(kind="bar", color="lightseagreen")
plt.title("Fusion Model F1 Scores (TF-IDF + BERT + SVM)")
plt.ylabel("F1 Score")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("../results/fusion_f1_scores.png", dpi=300)

from sklearn.metrics import hamming_loss
print("Hamming Loss:", hamming_loss(y_test, y_pred))  # 二值化后的预测
