import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
# 数据加载与预处理
df = pd.read_csv("../data/split_encoded_train/mapped_goemotions_1.csv")
label_cols = [col for col in df.columns if col not in ["id", "text", "sentiment", "ekman_emotion"]]
df["text"] = df["text"].fillna("").astype(str)

# 拆分数据
X_train, X_test, y_train, y_test = train_test_split(df["text"], df[label_cols], test_size=0.2, random_state=42)

# TF-IDF向量化
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 直接使用类别权重（跳过过采样）
model = OneVsRestClassifier(
    LogisticRegression(class_weight="balanced", solver="liblinear", max_iter=1000, random_state=42)
)
model.fit(X_train_vec, y_train)

# 评估
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred, target_names=label_cols, zero_division=0))