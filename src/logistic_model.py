
!pip uninstall -y transformers
!pip install transformers==4.40.2 accelerate datasets

!pip uninstall -y transformers accelerate datasets huggingface_hub
!pip install --upgrade pip
!pip install transformers accelerate datasets huggingface_hub

!pip install gensim
# =========================================
# LOGISTIC REGRESSION BASELINE
# =========================================

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ==========================
# 1. Load Dataset
# ==========================

train_df = pd.read_csv("train.tsv", sep="\t")
test_df = pd.read_csv("test.tsv", sep="\t")

print("Train size:", len(train_df))
print("Test size:", len(test_df))


# ==========================
# 2. Extract Intent
# ==========================

def extract_intent(parse_text):
    match = re.search(r'\[IN:([A-Z_]+)', parse_text)
    return match.group(1) if match else "UNKNOWN"

train_df['text'] = train_df['cs_query']
test_df['text'] = test_df['cs_query']

train_df['intent'] = train_df['cs_parse'].apply(extract_intent)
test_df['intent'] = test_df['cs_parse'].apply(extract_intent)


# ==========================
# 3. Label Encoding
# ==========================

label_encoder = LabelEncoder()

all_intents = pd.concat([
    train_df['intent'],
    test_df['intent']
])

label_encoder.fit(all_intents)

train_df['label'] = label_encoder.transform(train_df['intent'])
test_df['label'] = label_encoder.transform(test_df['intent'])

num_labels = len(label_encoder.classes_)
print("Total Intents:", num_labels)


# ==========================
# 4. TF-IDF + Logistic Regression
# ==========================

print("\nRunning Logistic Regression...\n")

tfidf = TfidfVectorizer(max_features=3000)

X_train = tfidf.fit_transform(train_df['text'])
X_test = tfidf.transform(test_df['text'])

y_train = train_df['label']
y_test = test_df['label']

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

y_pred = lr_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Logistic Regression Accuracy:", accuracy)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# ==========================
# 5. Confusion Matrix
# ==========================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, cmap="Blues")
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
