# 🧠 Cognitive Architect — Feedback Intelligence System

A **fully local AI system** that analyzes customer feedback using machine learning, visualizes insights through statistical graphs, and provides an intelligent chatbot interface — **without relying on any external APIs**.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.56+-red?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ Features

| Feature | Description |
|---|---|
| **ML Sentiment Classifier** | Trains both Logistic Regression & Naive Bayes, auto-selects the best |
| **Issue Category Detection** | Rule-based engine detecting Technical, Performance, Pricing, UI/UX issues |
| **Statistical Graphs** | Sentiment bar charts, issue distribution, pie charts, keyword frequency |
| **AI Chatbot** | Context-aware local chatbot that answers questions about your feedback data |
| **Live Inference** | Test any new feedback in real-time with confidence scores |
| **Improvement Engine** | Auto-generates actionable suggestions based on detected issues |
| **100% Local** | No OpenAI / external API dependency — proves real ML understanding |

---

## 🏗️ Architecture

```
User Feedback Dataset
        ↓
Preprocessing (NLTK — stemming, stopwords)
        ↓
Feature Extraction (TF-IDF with bigrams)
        ↓
ML Models (Logistic Regression + Naive Bayes)
        ↓
Best Model Auto-Selected → Predictions
        ↓
Visualization (Matplotlib graphs)
        ↓
Insight Engine (Keyword + Category Detection)
        ↓
Chatbot Layer (Local Intelligence + Memory)
```

---

## 📁 Project Structure

```
feedback-intelligence-chatbot/
│
├── app.py                    # Streamlit main application
├── requirements.txt          # Python dependencies
├── .gitignore
│
├── data/
│   └── feedback.csv          # Training dataset (110+ entries)
│
├── model/
│   ├── __init__.py
│   ├── train.py              # Dual-model training pipeline
│   ├── model.pkl             # Trained model (auto-generated)
│   ├── vectorizer.pkl        # TF-IDF vectorizer (auto-generated)
│   └── metrics.json          # Training metrics (auto-generated)
│
├── utils/
│   ├── __init__.py
│   ├── preprocess.py         # Text cleaning (NLTK)
│   ├── predictor.py          # Prediction + confidence scores
│   ├── issue_detector.py     # Rule-based issue categorization
│   ├── recommender.py        # Improvement suggestion engine
│   └── chatbot.py            # Local AI chatbot engine
│
└── assets/
    ├── index.html            # Reference UI design (Tailwind)
    └── style.css             # Custom Streamlit styles
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/feedback-intelligence-chatbot.git
cd feedback-intelligence-chatbot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model (optional — auto-trains on first run)
```bash
python model/train.py
```

### 4. Launch the dashboard
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 💬 Chatbot Queries

The AI chatbot understands questions like:

| Query | What it does |
|---|---|
| *"What are the main issues?"* | Shows issue category breakdown with visual bars |
| *"Show pricing complaints"* | Lists pricing-specific feedback with suggestions |
| *"What technical problems exist?"* | Surfaces technical bugs with examples |
| *"How is the overall sentiment?"* | Displays sentiment distribution with percentages |
| *"What should we improve?"* | Generates actionable improvement suggestions |
| *"Give me a summary report"* | Full analytics report with top concerns |
| *"What are common keywords?"* | Word frequency analysis |
| *"Show positive/negative feedback"* | Filtered feedback highlights |

---

## 📊 Model Performance

The system trains **two models** and automatically selects the best:

- **Logistic Regression** — linear classifier, great for text
- **Multinomial Naive Bayes** — probabilistic classifier, fast on sparse data

Both are evaluated using:
- Accuracy score
- Precision / Recall / F1-score (classification report)
- Stratified train/test split (80/20)

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Streamlit** — Interactive web dashboard
- **scikit-learn** — ML models (LogReg, NB)
- **NLTK** — Text preprocessing (stemming, stopwords)
- **Matplotlib** — Statistical visualizations
- **Pandas** — Data handling
- **TF-IDF** — Feature extraction with bigrams

---

## 🎯 Why This Project Matters

> *"I built a fully local AI system where feedback is analyzed using machine learning for sentiment classification. Then I extract insights using keyword-based categorization and visualize trends. On top of that, I implemented a chatbot layer that allows users to query the feedback data and receive intelligent summaries and improvement suggestions — all without any external API."*

This project demonstrates:
- ✅ ML pipeline design (preprocessing → features → training → evaluation)
- ✅ NLP fundamentals (TF-IDF, stemming, stopwords)
- ✅ Data visualization and analytics
- ✅ AI reasoning (rule-based chatbot with data queries)
- ✅ Full-stack system thinking
- ✅ Local-first architecture (no API dependency)

---

## 📄 License

MIT License — feel free to use and modify.

---

*Built with ❤️ as a fully local AI analytics system.*
