# Decoding-Code-Mixed-Communication-A-Deep-NLP-Approach-for-Hinglish-Text-Understanding
This project provides a systematic comparative study and demonstrates the practical effectiveness of multilingual transformer architectures for low-resource, code-mixed intent classification, contributing toward more linguistically adaptive conversational AI systems

# Overview
This project implements and compares multiple NLP models for multi-class intent classification on the Hinglish-TOP dataset. The objective is to evaluate classical, sequential, and transformer-based architectures under a unified experimental pipeline for code-mixed Hinglish text.
The task involves classifying Hinglish queries into 64 predefined intent categories.

# Models Implemented
TF-IDF + Logistic Regression (Baseline)

FastText + BiLSTM

Multilingual BERT (mBERT)

XLM-RoBERTa

All models are trained and evaluated using the same preprocessing and label encoding framework to ensure fair comparison.

# Dataset
Hinglish-TOP Dataset
64 Intent Classes
~10,896 total samples
Code-mixed Hindi-English (Roman script)

Fields used:
cs_query → Input text
cs_parse → Intent extraction

# Results Summary
Model	Accuracy
Logistic Regression	74.2%
BiLSTM	~40%
mBERT	77.5% (Best)
XLM-R	71.8%

mBERT achieved the highest performance due to contextual multilingual representation learning.

# Technologies Used
Python
Scikit-learn
PyTorch
HuggingFace Transformers
Gensim
Pandas, NumPy

# How to Run
Install dependencies: pip install -r requirements.txt
Place dataset files (train.tsv, validation.tsv, test.tsv) in project directory.
Run: python main.py


# Key Contributions
Unified preprocessing and evaluation pipeline
Comparative analysis across 4 modeling paradigms
Evaluation on low-resource code-mixed dataset
Insight into transformer effectiveness for Hinglish NLP

# Future Work
Data augmentation for code-mixed text
Few-shot and zero-shot learning
Attention visualization and explainability
