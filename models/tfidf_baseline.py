import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# 读取数据
df = pd.read_csv("../data/split_encoded_train/mapped_goemotions_1.csv")
label_cols = [col for col in df.columns if col not in ["id", "text", "sentiment", "ekman_emotion"]]

# 清理 NaN 文本
df["text"] = df["text"].fillna("").astype(str)

# 拆分训练与测试集
X_train, X_test, y_train, y_test = train_test_split(df["text"], df[label_cols], test_size=0.2, random_state=42)

# 向量化文本
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 训练 SVM 多标签模型
model = OneVsRestClassifier(LinearSVC())
model.fit(X_train_vec, y_train)

# 评估模型
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred, target_names=label_cols))
from sklearn.metrics import hamming_loss

hloss = hamming_loss(y_test, y_pred)
print("Hamming Loss:", hloss)
report = classification_report(y_test, y_pred, target_names=label_cols, output_dict=True)
# pd.DataFrame(report).transpose().to_csv("../results/tfidf_svm_report.csv")
report_df = pd.DataFrame(report).transpose()
report_df.loc["Hamming Loss", "f1-score"] = hloss
report_df.to_csv("../results/tfidf_svm_report.csv")

import matplotlib.pyplot as plt
f1_scores = pd.DataFrame(report).transpose().loc[label_cols, "f1-score"].sort_values(ascending=False)
plt.figure(figsize=(12, 6))
f1_scores.plot(kind="bar", color="cornflowerblue")
plt.title("TF-IDF + SVM Model - F1 Scores by Label")
plt.ylabel("F1 Score")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("../results/tfidf_svm_f1.png", dpi=300)
