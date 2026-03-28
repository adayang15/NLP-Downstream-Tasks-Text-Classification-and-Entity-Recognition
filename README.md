# NLP-Downstream-Tasks-Text-Classification-and-Entity-Recognition

## Project Overview
This project implements two classic NLP downstream tasks — **Text Classification** and **Named Entity Recognition (NER)** — using traditional machine learning methods. The text classification task benchmarks four classifiers with TF-IDF feature extraction on Chinese text, while the NER task employs a Conditional Random Field (CRF) model for sequence labeling. Both experiments were conducted as part of a Natural Language Processing course assignment at Beijing Forestry University.

## Objectives
- **Text Classification**: Apply TF-IDF feature extraction and benchmark four classical classifiers — Logistic Regression, kNN, Naive Bayes, and MLP — to evaluate performance on Chinese multi-class text classification
- **Named Entity Recognition**: Build a CRF-based sequence labeling model with handcrafted contextual features to identify and classify named entities in text
- **Comparative Evaluation**: Assess model performance on validation and test sets using accuracy, precision, recall, and F1 score across both tasks
- **Model Interpretability**: Extract top transition weights from the trained CRF model to interpret learned label dependency patterns

## Features
- **Data Preprocessing**: CoNLL-format parsing for NER data and tab-separated ingestion for classification data, with jieba-based Chinese word segmentation and sentence boundary detection
- **Feature Engineering**: TF-IDF vectorization for text classification; handcrafted word-level features including surface form, capitalization, digit flags, suffix patterns, and a ±2 context window for NER
- **Multi-Classifier Benchmarking**: Logistic Regression, K-Nearest Neighbors (kNN), Multinomial Naive Bayes, and Multilayer Perceptron (MLP) trained and evaluated on shared TF-IDF representations
- **CRF Sequence Labeling**: Conditional Random Field model trained with L-BFGS optimization and L1/L2 regularization, capturing global label dependencies across the full input sequence
