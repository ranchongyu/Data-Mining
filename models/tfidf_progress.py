import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, hamming_loss
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ==== 数据加载与预处理 ====
df = pd.read_csv("../data/split_encoded_train/mapped_goemotions_1.csv")
label_cols = [col for col in df.columns if col not in ["id", "text", "sentiment", "ekman_emotion"]]
df["text"] = df["text"].fillna("").astype(str)

# ==== 拆分训练与测试集 ====
X_train, X_test, y_train, y_test = train_test_split(df["text"], df[label_cols], test_size=0.2, random_state=42)

# ==== TF-IDF 向量化 ====
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ==== 模型训练 ====
model = OneVsRestClassifier(
    LogisticRegression(class_weight="balanced", solver="liblinear", max_iter=1000, random_state=42)
)
model.fit(X_train_vec, y_train)

# ==== 模型评估 ====
y_pred = model.predict(X_test_vec)

# === 分类报告保存为 CSV ===
report = classification_report(y_test, y_pred, target_names=label_cols, output_dict=True, zero_division=0)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv("../results/logistic_report.csv")

# === Hamming Loss 输出并加入报告 ===
hloss = hamming_loss(y_test, y_pred)
print("Hamming Loss:", hloss)
report_df.loc["Hamming Loss", "f1-score"] = hloss
report_df.to_csv("../results/logistic_report.csv")  # 更新版本包含 HL

# === 可视化 F1 Score 柱状图 ===
f1_scores = report_df.loc[label_cols, "f1-score"].sort_values(ascending=False)
plt.figure(figsize=(12, 6))
f1_scores.plot(kind="bar", color="darkorange")
plt.title("TF-IDF + Logistic Regression - F1 Scores by Label")
plt.ylabel("F1 Score")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("../results/logistic_f1.png", dpi=300)
