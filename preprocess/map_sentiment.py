import pandas as pd
import json
import os

# ==== 配置路径 ====
input_path = "../data/split_encoded_test/goemotions_test_1.csv"
output_path = "../data/split_encoded_test/mapped_goemotions_test_1.csv"

# ==== 加载映射文件 ====
with open("../data/sentiment_mapping.json", "r", encoding="utf-8") as f:
    sentiment_map = json.load(f)
with open("../data/ekman_mapping.json", "r", encoding="utf-8") as f:
    ekman_map = json.load(f)

# ==== 构建反向映射 ====
fine_to_sentiment = {emo: group for group, emos in sentiment_map.items() for emo in emos}
fine_to_ekman = {emo: group for group, emos in ekman_map.items() for emo in emos}

# ==== 加载数据 ====
df = pd.read_csv(input_path)

# ==== 获取标签列（自动筛选 one-hot）====
label_cols = [col for col in df.columns if df[col].dropna().apply(lambda x: str(x).isdigit()).all()]

# ==== 映射每一行的标签 → sentiment / ekman 主类 ====
def map_categories(row):
    active_labels = [label for label in label_cols if row[label] == 1]
    sentiments = {fine_to_sentiment[label] for label in active_labels if label in fine_to_sentiment}
    ekmans = {fine_to_ekman[label] for label in active_labels if label in fine_to_ekman}
    return pd.Series({
        "sentiment": ",".join(sorted(sentiments)) if sentiments else "none",
        "ekman_emotion": ",".join(sorted(ekmans)) if ekmans else "none"
    })

df[["sentiment", "ekman_emotion"]] = df.apply(map_categories, axis=1)

# ==== 保存新文件 ====
df.to_csv(output_path, index=False)
output_path
