"""
Feedback Intelligence - Standalone ML Pipeline Demo
This script isolates the Machine Learning component (Feedback Analyzer).
It can be run independently in Python, Google Colab, or Jupyter.

What it does:
1. Loads feedback dataset.
2. Preprocesses text using NLTK (Tokenization, Stopwords).
3. Converts text to numerical vectors using Scikit-Learn TF-IDF.
4. Trains a Sentiment Model (Logistic Regression default).
5. Predicts sentiment for new examples.
"""

import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------------------------------------
# 1. SETUP & DATA DOWNLOADING
# -------------------------------------------------------------
print("=> Downloading NLTK requirements...")
nltk.download('stopwords', quiet=True)

# You can replace this with your actual feedback.csv if available. 
# Here we use a sample list if the file doesn't exist, to ensure it works anywhere.
DATA_PATH = "../data/feedback.csv"

if os.path.exists(DATA_PATH):
    print(f"=> Loading dataset from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
else:
    print("=> No local CSV found. Using hardcoded sample dataset.")
    data = {
        "text": [
            "The app is incredibly slow and crashes often.",
            "I love the new dark mode, it looks professional.",
            "Pricing for the premium tier is just too expensive.",
            "The dashboard navigation is confusing and hard to use.",
            "Customer support was very helpful and fast!"
        ],
        "sentiment": ["negative", "positive", "negative", "negative", "positive"],
        "issue": ["Performance", "UI/UX", "Pricing", "UI/UX", "General"]
    }
    df = pd.DataFrame(data)

# -------------------------------------------------------------
# 2. NLP PREPROCESSING
# -------------------------------------------------------------
print("=> Preprocessing text (Removing stopwords, tokenization, stemming)...")
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    import re
    # Lowercase and remove punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
    # Stemming & Stopword removal
    words = [stemmer.stem(word) for word in text.split() if word not in stop_words]
    return ' '.join(words)

# Apply cleaning to the dataset
X_raw = df['text'].apply(clean_text)

# We use the 'sentiment' column as the target label
if 'sentiment' in df.columns:
    y = df['sentiment']
elif 'label' in df.columns:
    y = df['label']
else:
    raise ValueError("Dataset missing sentiment label column!")

# -------------------------------------------------------------
# 3. FEATURE EXTRACTION (TF-IDF)
# -------------------------------------------------------------
print("=> Extracting TF-IDF Features...")
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(X_raw)

# -------------------------------------------------------------
# 4. MODEL TRAINING
# -------------------------------------------------------------
# Split dataset (only if we have enough data)
if len(df) > 10:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
else:
    X_train, X_test, y_train, y_test = X, X, y, y

print("=> Training Logistic Regression Model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -------------------------------------------------------------
# 5. EVALUATION
# -------------------------------------------------------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n[RESULTS] Model Accuracy: {acc * 100:.2f}%")
if len(set(y_test)) > 1:
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

# -------------------------------------------------------------
# 6. INFERENCE (TESTING NEW FEEDBACK)
# -------------------------------------------------------------
print("==================================================")
print("             TEST THE ANALYZER                  ")
print("==================================================")
test_inputs = [
    "The login page takes forever to load.",
    "This is the best tool I have ever used for my business.",
    "It's just okay, nothing special."
]

for text in test_inputs:
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]
    
    # Calculate Confidence
    probas = model.predict_proba(vec)[0]
    confidence = max(probas) * 100
    
    print(f"Feedback: '{text}'")
    print(f"-> Sentiment: {prediction.upper()} (Confidence: {confidence:.1f}%)\n")

print("=> Standalone Demo Complete! 🔥")
