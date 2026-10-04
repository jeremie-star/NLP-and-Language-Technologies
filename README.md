# Research-Informed Sequential Models for NLP: Gender-Based Violence Tweet Classification

**Formative Assignment 2 — Group 8**
Course: NLP and Language Technologies

> ⚠️ **Content warning:** This project works with tweets that describe sexual, physical, emotional and economic violence. The data, notebooks and example predictions contain distressing language.

---

## Overview

This project studies **five-class classification of gender-based violence (GBV) tweets** and asks a focused research question:

> **How effectively can different sequential and text-modelling approaches classify GBV tweets — and do more complex sequential models add anything beyond simple lexical cues?**

We compare two order-blind lexical baselines against three neural sequential models, each chosen for a different inductive bias:

| ID  | Approach                          | What it captures                             |
| --- | --------------------------------- | -------------------------------------------- |
| M0  | Majority-class & keyword rules    | Zero-learning reference points               |
| M1  | Word TF-IDF + Logistic Regression | Bag-of-ngrams (order-blind control)          |
| M2  | Character TF-IDF + Linear SVM     | Sub-word robustness without neural machinery |
| M3  | TextCNN                           | Local n-gram patterns                        |
| M4  | BiLSTM                            | Bidirectional sequential dependency          |
| M5  | BERTweet (fine-tuned)             | Pretrained contextual self-attention         |

## Dataset

- **Source:** [Zindi — Gender-Based Violence Tweet Classification Challenge](https://zindi.africa/competitions/gender-based-violence-tweet-classification-challenge)
- **Size:** 39,650 English tweets, collected from Twitter via _Twint_
- **Labels (5 classes):** `sexual_violence`, `physical_violence`, `emotional_violence`, `economic_violence`, `harmful_traditional_practice`
- **Imbalance:** ~82.3% of tweets are the majority class; the full imbalance ratio is **≈174:1** (rarest class has only 188 tweets total, ~28 in the test split)

## Repository structure

```
.
├── notebooks/
│   ├── EDA and Baselines.ipynb                       # M0–M2: EDA, split, lexical baselines
│   └── GBV_Tweet_Classification_—_Model_5_Transformer_(BERTweet).ipynb   # M5
├── results/
│   ├── results_model5_bertweet.csv                   # BERTweet full-data metrics
│   ├── results_model5_low_resource.csv               # BERTweet low-resource runs
│   └── test_predictions_m5_bertweet.csv              # Per-tweet predictions (for error analysis)
├── figures/
│   └── 08_learning_curves.png
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup and reproduction

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the Zindi dataset and place the split files under data/

# 4. Launch Jupyter and run the notebooks in order
jupyter notebook
```

Run `notebooks/EDA and Baselines.ipynb` first (it produces the EDA, the stratified 70/15/15 split and the M0–M2 baselines), then the BERTweet notebook. BERTweet training benefits from a GPU (e.g. Google Colab); the lexical baselines run on CPU.
