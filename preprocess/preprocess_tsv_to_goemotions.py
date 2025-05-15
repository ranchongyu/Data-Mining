import pandas as pd
import os
import re
import math
import json

# ==== 路径配置 ====
input_file = "../data/test.tsv"
output_dir = "../data/split_encoded_test"
os.makedirs(output_dir, exist_ok=True)

# ==== 读取情绪标签 ====
with open("../data/emotions.txt", "r", encoding="utf-8") as f:
    emotions = [line.strip() for line in f.readlines()]

# ==== 文本清洗函数 ====
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|@\S+", "", text)
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ==== 加载并预处理 TSV ====
df = pd.read_csv(input_file, sep="\t", names=["text", "labels", "id"], header=None)
df.dropna(subset=["text", "labels"], inplace=True)
df["text"] = df["text"].apply(clean_text)

# ==== 初始化标签列 ====
for emo in emotions:
    df[emo] = 0

# ==== 多标签编码 ====
for i, row in df.iterrows():
    for label_id in str(row["labels"]).split(","):
        if label_id.isdigit():
            idx = int(label_id)
            if 0 <= idx < len(emotions):
                df.at[i, emotions[idx]] = 1

# ==== 移除无标签样本 ====
df["label_sum"] = df[emotions].sum(axis=1)
df = df[df["label_sum"] > 0].drop(columns=["label_sum", "labels"])

# ==== 按 70000 条划分成多个文件 ====
chunk_size = 70000
total_rows = len(df)
num_chunks = math.ceil(total_rows / chunk_size)

for i in range(num_chunks):
    chunk = df.iloc[i * chunk_size : (i + 1) * chunk_size]
    out_path = os.path.join(output_dir, f"goemotions_test_{i+1}.csv")
    chunk.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[✓] Saved {out_path} with {len(chunk)} rows")
