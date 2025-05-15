# Now that emotions.txt is uploaded, rerun the top words + wordcloud extraction
import pandas as pd
import os
from collections import Counter, defaultdict
import math
import operator
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
# ==== 文件路径配置 ====
data_path = "../data/split_encoded_train/mapped_goemotions_1.csv"
emotion_file = "../data/emotions.txt"
output_csv = "../data/emotion_top_words.csv"
wordcloud_dir = "../results/wordclouds"
os.makedirs(wordcloud_dir, exist_ok=True)

# ==== 清洗函数 ====
import re
import string
punct_chars = list(set(string.punctuation) | {"’", "‘", "–", "—", "~", "|", "“", "”", "…", "'", "`", "_"})
punct_chars.sort()
replace = re.compile("[%s]" % re.escape("".join(punct_chars)))

def clean_text(text):
    if isinstance(text, float):
        return []
    text = text.lower()
    text = re.sub(r"http\S*|\S*\.com\S*|\S*www\S*", " ", text)
    text = re.sub(r"\s@\S+", " ", text)
    text = replace.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return [w for w in text.split() if len(w) > 2]

# ==== 读取数据 ====
df = pd.read_csv(data_path)
with open(emotion_file, "r") as f:
    all_emotions = [line.strip() for line in f.readlines()]

# ==== 清洗文本 ====
df["text_tokens"] = df["text"].apply(clean_text)

# ==== 统计词频 ====
def get_counts(text_series):
    words = []
    for tokens in text_series:
        words.extend(tokens)
    return Counter(words)

# ==== Log-Odds 计算 ====
def log_odds(counts1, counts2, prior, zscore=True):
    sigmasquared = defaultdict(float)
    sigma = defaultdict(float)
    delta = defaultdict(float)
    n1 = sum(counts1.values())
    n2 = sum(counts2.values())
    nprior = sum(prior.values())
    for word in prior:
        l1 = (counts1[word] + prior[word]) / ((n1 + nprior) - (counts1[word] + prior[word]))
        l2 = (counts2[word] + prior[word]) / ((n2 + nprior) - (counts2[word] + prior[word]))
        sigmasquared[word] = 1 / (counts1[word] + prior[word]) + 1 / (counts2[word] + prior[word])
        sigma[word] = math.sqrt(sigmasquared[word])
        delta[word] = math.log(l1) - math.log(l2)
        if zscore:
            delta[word] /= sigma[word]
    return delta

# ==== 分情绪提取关键词 ====
top_words_records = []
for emotion in all_emotions:
    contains = df[df[emotion] == 1]
    not_contains = df[df[emotion] == 0]
    counts_emotion = get_counts(contains["text_tokens"])
    counts_other = get_counts(not_contains["text_tokens"])
    prior = counts_emotion + counts_other
    delta = log_odds(counts_emotion, counts_other, prior, zscore=True)
    sorted_words = sorted(delta.items(), key=operator.itemgetter(1), reverse=True)
    top_words = []
    wordcloud_input = {}
    for word, score in sorted_words:
        if score < 2: continue
        freq = counts_emotion[word] / sum(counts_emotion.values())
        top_words_records.append({
            "emotion": emotion,
            "word": word,
            "log_odds": round(score, 2),
            "frequency": round(freq, 4)
        })
        wordcloud_input[word] = score
        if len(top_words) < 10:
            top_words.append(word)

    # ==== 词云图 ====
    if wordcloud_input:
        wc = WordCloud(width=600, height=400, background_color='white').generate_from_frequencies(wordcloud_input)
        wc.to_file(os.path.join(wordcloud_dir, f"{emotion}_wordcloud.png"))

# ==== 输出关键词表格 ====
top_words_df = pd.DataFrame(top_words_records)
top_words_df.to_csv(output_csv, index=False)

# import ace_tools as tools;
# tools.display_dataframe_to_user(name="Top 10 Emotion Words", dataframe=top_words_df)
barplot_dir = "../results/barplots"
os.makedirs(barplot_dir, exist_ok=True)

# 每个情绪画前10个词的柱状图
for emotion in all_emotions:
    sub_df = top_words_df[top_words_df["emotion"] == emotion].sort_values(by="log_odds", ascending=False).head(10)
    if sub_df.empty:
        continue
    plt.figure(figsize=(8, 4))
    sns.barplot(x="log_odds", y="word", data=sub_df, palette="viridis")
    plt.title(f"Top Keywords for {emotion}")
    plt.xlabel("Log Odds Score")
    plt.ylabel("Keyword")
    plt.tight_layout()
    plt.savefig(os.path.join(barplot_dir, f"{emotion}_keywords_barplot.png"), dpi=300)
    plt.close()