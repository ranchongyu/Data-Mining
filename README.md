# 🎯 GoEmotions Multi-Label Emotion Classification

A data mining project for multi-label emotion classification based on the **GoEmotions dataset** (by Google Research), supporting multiple modeling approaches including **TF-IDF + SVM**, **BERT Fine-tuning**, and **Hybrid TF-IDF + BERT**. Evaluation includes Micro/Macro/Sample F1 scores, Hamming Loss, and label-wise visualizations.

------

## 📦 Project Structure

```
GoEmotions
├── analysis/             # Visualization and correlation analysis
├── data/                 # Preprocessed and mapped datasets
├── eval/                 # Evaluation utilities (metrics, plots)
├── models/               # Model scripts: tfidf, bert, hybrid
├── preprocess/           # Text cleaning, label encoding, mapping
├── results/              # Output: .csv reports, .png plots
├── requirements.txt      # Full list of dependencies
├── README.md             # Project documentation
└── ...
```

------

## 📂 Dataset Preparation (GoEmotions)

We use the [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions) dataset which contains 58,000+ Reddit comments labeled with 27 emotion tags.

📥 Download the original data using:

```
dir -p datasets/goemotions
curl -o ./datasets/goemotions/goemotions_1.csv https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_1.csv
curl -o ./datasets/goemotions/goemotions_2.csv https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_2.csv
curl -o ./datasets/goemotions/goemotions_3.csv https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_3.csv
```

After downloading, run preprocessing:

```
python preprocess/prepare_data.py
```

This script performs:

- Label parsing & mapping (Ekman + Sentiment)
- Multi-hot encoding of emotion tags
- Clean text normalization
- Output as `mapped_goemotions_1.csv` in `data/split_encoded_train/`

------

## 🧠 Modeling Approaches

We implemented three types of classifiers:

| Model                  | Script                      | Features Used           |
| ---------------------- | --------------------------- | ----------------------- |
| **TF-IDF + SVM**       | `models/tfidf_baseline.py`  | TF-IDF only             |
| **BERT Fine-tune**     | `models/bert_multilabel.py` | BERT CLS token          |
| **Hybrid TF-IDF+BERT** | `models/tfidf_bert.py`      | TF-IDF + BERT embedding |



Each model:

- Loads `mapped_goemotions_1.csv`
- Splits train/test (80/20)
- Supports multi-label classification (`OneVsRestClassifier` for classical models, `Trainer` for BERT)
- Outputs:
  - `classification_report.csv`
  - `f1_score_plot.png`
  - Hamming Loss

------

## 📊 Evaluation Outputs

Evaluation results (stored in `results/`) include:

- `*_report.csv`: Precision / Recall / F1 by label
- `*_f1.png`: Sorted bar chart of label-wise F1 scores
- `bert_eval_summary.csv`: Overall metrics from Huggingface Trainer
- `bert_label_report.csv`: BERT per-label report
- `bert_f1_scores.png`: F1 chart for BERT

------

## 📈 Metrics Used

- **Micro-F1 / Macro-F1 / Samples-F1**
- **Hamming Loss**
- **Confusion Matrix**
- **Label-wise analysis**

------

## 🚀 How to Run

Ensure you’ve activated your virtual environment and installed dependencies:

```
pip install -r requirements.txt
```

Then, run models individually:

```
# TF-IDF baseline
python models/tfidf_baseline.py

# BERT fine-tuning (Trainer)
python models/bert_multilabel.py

# TF-IDF + BERT hybrid
python models/tfidf_bert.py
```

Outputs will be saved to `results/`.

------

## 🔧 Requirements

```
numpy>=1.19.5
pandas>=1.1.5
matplotlib>=3.3.4
seaborn>=0.11.1
scikit-learn>=0.24.2
scipy>=1.6.2
transformers==4.36.2
datasets>=2.10.1
accelerate>=0.21.0
torch>=1.10.0
tqdm>=4.62.3
absl-py>=0.13.0
```

------

## 📍 Author

Developed by **Ran Chongyu**
 University of Leeds – Data Mining Coursework
 Course: XJCO2121