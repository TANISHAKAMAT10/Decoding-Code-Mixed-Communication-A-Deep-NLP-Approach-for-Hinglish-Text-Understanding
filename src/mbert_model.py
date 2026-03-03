
!pip uninstall -y transformers
!pip install transformers==4.40.2 accelerate datasets

!pip uninstall -y transformers accelerate datasets huggingface_hub
!pip install --upgrade pip
!pip install transformers accelerate datasets huggingface_hub

!pip install gensim

# ==========================
# mBERT MODEL
# ==========================
import pandas as pd
import re
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset as HFDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix

print("Using GPU:", torch.cuda.is_available())

# ==========================
# 1. Load Dataset
# ==========================

train_df = pd.read_csv("train.tsv", sep="\t")
val_df = pd.read_csv("validation.tsv", sep="\t")
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
all_intents = pd.concat([train_df['intent'], test_df['intent']])
label_encoder.fit(all_intents)

train_df['label'] = label_encoder.transform(train_df['intent'])
test_df['label'] = label_encoder.transform(test_df['intent'])

num_labels = len(label_encoder.classes_)
print("Total Intents:", num_labels)

# ==========================
# 4. Convert to HF Dataset
# ==========================

train_hf = HFDataset.from_pandas(train_df[['text','label']])
test_hf = HFDataset.from_pandas(test_df[['text','label']])

# ==========================
# 5. Tokenization
# ==========================

tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

def tokenize(batch):
    return tokenizer(
        batch['text'],
        padding=True,
        truncation=True,
        max_length=80
    )

train_hf = train_hf.map(tokenize, batched=True)
test_hf = test_hf.map(tokenize, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ==========================
# 6. Load Model
# ==========================

model = BertForSequenceClassification.from_pretrained(
    "bert-base-multilingual-cased",
    num_labels=num_labels
)

# ==========================
# 7. Metrics
# ==========================

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}

# ==========================
# 8. Training Arguments
# ==========================

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=6,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    fp16=torch.cuda.is_available()
)

# ==========================
# 9. Trainer
# ==========================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_hf,
    eval_dataset=test_hf,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# ==========================
# 10. Train & Evaluate
# ==========================

trainer.train()

results = trainer.evaluate()

print("\nFinal mBERT Accuracy:", results["eval_accuracy"])


# ==========================
# 11. Confusion Matrix
# ==========================

predictions = trainer.predict(test_hf)
y_true = predictions.label_ids
y_pred = np.argmax(predictions.predictions, axis=1)

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10,8))
sns.heatmap(cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - mBERT")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()