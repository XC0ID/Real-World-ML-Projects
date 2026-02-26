# Review Sentiment Analysis

A real-world machine learning project that analyzes customer reviews and predicts sentiment using Natural Language Processing (NLP) and deep learning models.

---

# Project Overview

Customer reviews contain valuable insights about products and services. However, manually analyzing thousands of reviews is inefficient.
This project builds a machine learning pipeline that automatically classifies reviews into **positive** or **negative** sentiment.

The project demonstrates an **end-to-end ML workflow**, including:

• Data preprocessing
• Exploratory Data Analysis
• Feature engineering
• Model training
• Evaluation
• Performance comparison

The goal is to simulate how sentiment analysis is applied in real-world business scenarios such as product feedback analysis and customer satisfaction monitoring.

---

# Folder

```
Review-Sentiment/
│
├── data/
│   └── Appliances_Reviews.csv
|
├── LSTM_Model.ipynb
|
├── BERT_Model.ipynb
|
├── project_report.pdf
│
└── README.md
```

---

# Dataset

Dataset: Amazon Appliances Reviews

Features:

| Column     | Description     |
| ---------- | --------------- |
| reviewText | Customer review |
| summary    | Short summary   |
| overall    | Rating score    |

Sentiment labeling rule:

```
rating > 3 → Positive
rating ≤ 3 → Negative
```

---

# Data Processing Pipeline

1. Text Cleaning
2. Lowercasing
3. Removing special characters
4. Tokenization
5. Padding sequences
6. Train-test split

Example preprocessing:

```python
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text
```

---

# Models Implemented

## 1. LSTM Model

Architecture:

```
Embedding
   ↓
Bi-Directional LSTM
   ↓
Dropout
   ↓
Fully Connected Layer
```

Advantages:

• Fast training
• Good baseline performance
• Handles sequential text well

---

# 2. Transformer Model (BERT / MiniLM)

Architecture:

```
Tokenizer
   ↓
Pretrained Transformer Encoder
   ↓
Pooling
   ↓
Classifier
```

Advantages:

• Context aware
• Higher accuracy
• State-of-the-art NLP performance

---

# Training Configuration

| Parameter  | Value        |
| ---------- | ------------ |
| Batch Size | 32 / 64      |
| Epochs     | 2-5          |
| Optimizer  | AdamW        |
| Loss       | CrossEntropy |

---

# Model Evaluation

Metrics used:

• Accuracy
• Precision
• Recall
• F1 Score
• ROC-AUC
• Confusion Matrix

Example:

```
Accuracy : 0.91
Precision: 0.90
Recall   : 0.92
F1 Score : 0.91
ROC AUC  : 0.94
```

---

# Confusion Matrix

```
                Predicted
             Neg      Pos
Actual Neg   5300      700
Actual Pos    450     6100
```

---

# Technologies Used

Python
PyTorch
Transformers
Scikit-learn
Pandas
NumPy
Matplotlib
Seaborn

---

# Key Learnings

• Building NLP pipelines
• Text preprocessing techniques
• Training deep learning models
• Model evaluation techniques
• Performance optimization

---

# Future Improvements

• Hyperparameter tuning
• Larger dataset training
• Model deployment API
• Real-time prediction system
• Dashboard visualization

---

# Results

| Model      | Accuracy |
| ---------- | -------- |
| LSTM       | 0.87     |
| DistilBERT | 0.91     |
| MiniLM     | 0.92     |

---

