# GBV Tweet Classification Using Sequential and Text Models

This project investigates how different machine learning and neural language modelling approaches can classify tweets related to **Gender-Based Violence (GBV)**. "https://zindi.world/competitions/gender-based-violence-tweet-classification-challenge-2025"

The study compares traditional text-classification baselines with neural sequential models to understand how different representations of language perform on a real-world social media classification task.

## Problem

The goal is to automatically classify GBV-related tweets into five categories:

* **Sexual Violence**
* **Physical Violence**
* **Emotional Violence**
* **Economic Violence**
* **Harmful Traditional Practices**

This can support large-scale analysis of GBV-related discussions on social media, where manually reviewing large numbers of posts is difficult.

## Dataset

We use the **Gender-Based Violence Tweet Classification Challenge 2025** dataset.

The dataset contains **39,650 labelled tweets** across the five GBV categories. The data is highly imbalanced, with approximately **82.3% of tweets belonging to the majority class** and the smallest class containing only 188 examples.

Our experiments use a shared, stratified and duplicate-aware:

**70% training / 15% validation / 15% test split**

Because of the class imbalance, **macro-F1** is used as our primary evaluation metric, alongside accuracy and other supporting metrics.

## Models

We investigate five different approaches:

| Model   | Approach                          |
| ------- | --------------------------------- |
| Model 1 | Word TF-IDF + Logistic Regression |
| Model 2 | Character TF-IDF + Linear SVM     |
| Model 3 | TextCNN                           |
| Model 4 | BiLSTM                            |
| Model 5 | BERTweet                          |

The first two models provide strong traditional baselines, while TextCNN, BiLSTM and BERTweet allow us to investigate neural approaches to text and sequential modelling.

## Key Findings

The dataset contains strong lexical patterns, meaning that relatively simple models can already achieve very high performance.

The two completed traditional baselines achieved:

* **Word TF-IDF + Logistic Regression:** 0.9838 test macro-F1
* **Character TF-IDF + Linear SVM:** 0.9929 test macro-F1

Five-fold cross-validation showed that the difference between the two traditional models was relatively small compared with fold-to-fold variation.

The neural experiments further investigate local patterns, sequential dependencies, contextual representations, and performance under limited training data.

## Project Structure

```text
gbv_tweet_project/
│
├── data/
│   └── splits/
│       ├── train.csv
│       ├── val.csv
│       ├── test.csv
│       └── low_resource_protocol.json
│
├── notebooks/
│   ├── model_1_logistic_regression/
│   ├── model_2_linear_svm/
│   ├── model_3_textcnn/
│   ├── model_4_bilstm/
│   └── model_5_bertweet/
│
├── figures/
│
│   ├── learning_curves/
│  
│  
│
├── results/
│   ├── model_results/
│   └── predictions/
│
└── README.md
```

## Experimental Setup

The experiments use a shared dataset split and consistent evaluation protocol.

For the neural models, we investigate:

* class-weighted training
* validation-based model selection
* learning curves
* hyperparameter tuning
* confusion matrices
* error analysis
* low-resource training

BERTweet is used as a Twitter-specific pretrained Transformer model and is evaluated using subword tokenisation, class-weighted loss, learning-rate tuning and early stopping.

## Reproducibility

The experiments were developed primarily in **Google Colab** using Python and common machine learning libraries including:

* Python
* pandas
* NumPy
* scikit-learn
* PyTorch
* Hugging Face Transformers



## Team

This project was completed collaboratively as part of the **ALU Software Engineering NLP coursework**.

Each team member contributed to different parts of the dataset investigation, modelling, experimentation, evaluation, and analysis.

---
