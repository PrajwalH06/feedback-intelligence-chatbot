import pandas as pd
import pickle
import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Add parent dir to path so we can import utils
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocess import clean_text


def train():
    """
    Train both Logistic Regression and Naive Bayes models.
    Selects the best performing model and saves it.
    Also saves training metrics for the dashboard.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    print("=" * 50)
    print("  FEEDBACK INTELLIGENCE - MODEL TRAINING")
    print("=" * 50)

    # Load dataset
    csv_path = os.path.join(base_dir, "data", "feedback.csv")
    print(f"\n[1/5] Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"       Loaded {len(df)} feedback entries.")
    print(f"       Label distribution:\n{df['label'].value_counts().to_string()}")

    # Preprocess
    print("\n[2/5] Preprocessing text (stemming, stopword removal)...")
    df['cleaned'] = df['text'].apply(clean_text)

    # Feature extraction
    print("\n[3/5] Extracting TF-IDF features...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df['cleaned'])
    y = df['label']
    print(f"       Feature matrix shape: {X.shape}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"       Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

    # ---- Train Logistic Regression ----
    print("\n[4/5] Training models...")
    print("       > Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, C=1.0)
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_accuracy = accuracy_score(y_test, lr_pred)
    print(f"         Accuracy: {lr_accuracy:.4f}")

    # ---- Train Naive Bayes ----
    print("       > Multinomial Naive Bayes...")
    nb_model = MultinomialNB(alpha=0.5)
    nb_model.fit(X_train, y_train)
    nb_pred = nb_model.predict(X_test)
    nb_accuracy = accuracy_score(y_test, nb_pred)
    print(f"         Accuracy: {nb_accuracy:.4f}")

    # Select best model
    if lr_accuracy >= nb_accuracy:
        best_model = lr_model
        best_name = "Logistic Regression"
        best_accuracy = lr_accuracy
        best_pred = lr_pred
    else:
        best_model = nb_model
        best_name = "Multinomial Naive Bayes"
        best_accuracy = nb_accuracy
        best_pred = nb_pred

    print(f"\n       [BEST] Model: {best_name} ({best_accuracy:.4f})")

    # Classification report
    print(f"\n       Classification Report ({best_name}):")
    report = classification_report(y_test, best_pred)
    print(report)

    # Save model artifacts
    print("[5/5] Saving model artifacts...")
    model_dir = os.path.join(base_dir, "model")
    os.makedirs(model_dir, exist_ok=True)

    pickle.dump(best_model, open(os.path.join(model_dir, "model.pkl"), "wb"))
    pickle.dump(vectorizer, open(os.path.join(model_dir, "vectorizer.pkl"), "wb"))

    # Save training metrics for the dashboard
    metrics = {
        "best_model": best_name,
        "best_accuracy": round(best_accuracy * 100, 2),
        "lr_accuracy": round(lr_accuracy * 100, 2),
        "nb_accuracy": round(nb_accuracy * 100, 2),
        "total_samples": len(df),
        "train_samples": X_train.shape[0],
        "test_samples": X_test.shape[0],
        "features": X.shape[1],
        "label_distribution": df['label'].value_counts().to_dict()
    }
    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"       Saved: model.pkl, vectorizer.pkl, metrics.json")
    print("\n" + "=" * 50)
    print(f"  TRAINING COMPLETE -- {best_name}: {best_accuracy:.2%}")
    print("=" * 50)

    return metrics


if __name__ == "__main__":
    train()
