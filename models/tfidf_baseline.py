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
