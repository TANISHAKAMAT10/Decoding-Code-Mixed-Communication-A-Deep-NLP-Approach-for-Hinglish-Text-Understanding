
!pip uninstall -y transformers
!pip install transformers==4.40.2 accelerate datasets

!pip uninstall -y transformers accelerate datasets huggingface_hub
!pip install --upgrade pip
!pip install transformers accelerate datasets huggingface_hub

!pip install gensim
# =========================================
# FASTTEXT + BiLSTM MODEL
# =========================================

import pandas as pd
import numpy as np
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from gensim.models import FastText
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


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
# 4. Train FastText Embeddings
# ==========================

train_tokens = [sentence.split() for sentence in train_df['text']]

ft_model = FastText(
    sentences=train_tokens,
    vector_size=50,
    window=3,
    min_count=1,
    workers=2
)

vocab = ft_model.wv.key_to_index
vocab_size = len(vocab)


# ==========================
# 5. Dataset Class
# ==========================

class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=15):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.texts[idx].split()
        ids = [self.vocab[word] if word in self.vocab else 0 for word in tokens]
        ids = ids[:self.max_len]
        ids += [0] * (self.max_len - len(ids))
        return torch.tensor(ids), torch.tensor(self.labels[idx])


train_dataset = TextDataset(
    train_df['text'].tolist(),
    train_df['label'].tolist(),
    vocab
)

test_dataset = TextDataset(
    test_df['text'].tolist(),
    test_df['label'].tolist(),
    vocab
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)


# ==========================
# 6. BiLSTM Model
# ==========================

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim,
                            bidirectional=True,
                            batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        return self.fc(hidden)


device = torch.device("cpu")

model = BiLSTM(vocab_size, 50, 64, num_labels).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# ==========================
# 7. Training
# ==========================

for epoch in range(2):
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss}")


# ==========================
# 8. Evaluation
# ==========================

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total

print("\nBiLSTM Accuracy:", accuracy)