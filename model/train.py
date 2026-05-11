import pandas as pd
import pickle
import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report

# Add parent dir to path so we can import utils
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocess import clean_text


def train():
    """
    Train Logistic Regression, Naive Bayes, and SVM models.
    Uses cross-validation to select the best model and avoid overfitting.
    Also saves training metrics for the dashboard.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    print("=" * 50)
    print("  FEEDBACK INTELLIGENCE - MODEL TRAINING")
    print("=" * 50)

    # Load dataset
    csv_path = os.path.join(base_dir, "data", "training_data.csv")
    print(f"\n[1/6] Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    # Clean out any corrupt rows
    df = df[~((df['text'] == 'text') & (df['label'] == 'label'))]
    df = df.dropna(subset=['text', 'label'])
    df = df.drop_duplicates(subset=['text'], keep='first').reset_index(drop=True)
    print(f"       Loaded {len(df)} feedback entries.")
    print(f"       Label distribution:\n{df['label'].value_counts().to_string()}")

    # Preprocess
    print("\n[2/6] Preprocessing text (stemming, stopword removal)...")
    df['cleaned'] = df['text'].apply(clean_text)

    # Feature extraction — tuned for better generalization
    print("\n[3/6] Extracting TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 3),    # unigrams + bigrams + trigrams (captures "not good", "very bad app")
        min_df=1,              # keep ALL terms — important for short phrases
        max_df=0.95,           # ignore terms in 95%+ of docs (too common)
        sublinear_tf=True,     # apply log normalization to term frequencies
    )
    X = vectorizer.fit_transform(df['cleaned'])
    y = df['label']
    print(f"       Feature matrix shape: {X.shape}")

    # Guard: need at least 2 classes to train meaningfully
    n_classes = y.nunique()
    if n_classes < 2:
        print(f"\n       [WARNING] Only {n_classes} class(es) found: {y.unique().tolist()}")
        print(f"       Training a basic model, but predictions will be poor.")
        print(f"       Import more diverse labeled data for better accuracy.")
        # Train a simple model that at least doesn't crash
        lr = LogisticRegression(max_iter=1000, C=0.5, solver='lbfgs')
        lr.fit(X, y)
        model_dir = os.path.join(base_dir, "model")
        os.makedirs(model_dir, exist_ok=True)
        pickle.dump(lr, open(os.path.join(model_dir, "model.pkl"), "wb"))
        pickle.dump(vectorizer, open(os.path.join(model_dir, "vectorizer.pkl"), "wb"))
        metrics = {
            "best_model": "Logistic Regression (single-class fallback)",
            "best_accuracy": 0, "best_cv_accuracy": 0,
            "lr_accuracy": 0, "nb_accuracy": 0, "svm_accuracy": 0,
            "total_samples": len(df), "train_samples": len(df),
            "test_samples": 0, "features": X.shape[1],
            "label_distribution": y.value_counts().to_dict()
        }
        with open(os.path.join(model_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n  TRAINING COMPLETE (single-class fallback)")
        return metrics

    # Guard: need enough samples per class for stratified split
    min_class_count = y.value_counts().min()
    test_size = 0.2
    if min_class_count < 5:
        test_size = max(1, int(min_class_count * 0.3)) / len(df)
        test_size = min(test_size, 0.3)

    # Split — hold out for final evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    print(f"       Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

    # ── Cross-validation setup ──
    n_cv_splits = min(5, min_class_count)
    n_cv_splits = max(2, n_cv_splits)
    print(f"\n[4/6] Cross-validating models ({n_cv_splits}-fold)...")
    cv = StratifiedKFold(n_splits=n_cv_splits, shuffle=True, random_state=42)
    models = {}

    # ---- Logistic Regression (with regularization tuning) ----
    print("       > Logistic Regression (C=0.5, L2)...")
    lr_model = LogisticRegression(max_iter=1000, C=0.5, solver='lbfgs')
    lr_cv_scores = cross_val_score(lr_model, X_train, y_train, cv=cv, scoring='accuracy')
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_test_acc = accuracy_score(y_test, lr_pred)
    lr_cv_mean = lr_cv_scores.mean()
    print(f"         CV Accuracy: {lr_cv_mean:.4f} (+/- {lr_cv_scores.std():.4f})")
    print(f"         Test Accuracy: {lr_test_acc:.4f}")
    models['Logistic Regression'] = {
        'model': lr_model, 'cv_mean': lr_cv_mean, 'test_acc': lr_test_acc, 'pred': lr_pred
    }

    # ---- Multinomial Naive Bayes ----
    print("       > Multinomial Naive Bayes (alpha=1.0)...")
    nb_model = MultinomialNB(alpha=1.0)
    nb_cv_scores = cross_val_score(nb_model, X_train, y_train, cv=cv, scoring='accuracy')
    nb_model.fit(X_train, y_train)
    nb_pred = nb_model.predict(X_test)
    nb_test_acc = accuracy_score(y_test, nb_pred)
    nb_cv_mean = nb_cv_scores.mean()
    print(f"         CV Accuracy: {nb_cv_mean:.4f} (+/- {nb_cv_scores.std():.4f})")
    print(f"         Test Accuracy: {nb_test_acc:.4f}")
    models['Multinomial Naive Bayes'] = {
        'model': nb_model, 'cv_mean': nb_cv_mean, 'test_acc': nb_test_acc, 'pred': nb_pred
    }

    # ---- Linear SVM (calibrated for probability estimates) ----
    print("       > Linear SVM (C=0.5)...")
    try:
        svm_base = LinearSVC(C=0.5, max_iter=2000)
        min_class_count = min(y_train.value_counts())
        svm_cv_folds = min(3, min_class_count) if min_class_count >= 2 else 2
        svm_model = CalibratedClassifierCV(svm_base, cv=svm_cv_folds)
        svm_cv_scores = cross_val_score(svm_base, X_train, y_train, cv=cv, scoring='accuracy')
        svm_model.fit(X_train, y_train)
        svm_pred = svm_model.predict(X_test)
        svm_test_acc = accuracy_score(y_test, svm_pred)
        svm_cv_mean = svm_cv_scores.mean()
        print(f"         CV Accuracy: {svm_cv_mean:.4f} (+/- {svm_cv_scores.std():.4f})")
        print(f"         Test Accuracy: {svm_test_acc:.4f}")
        models['Linear SVM'] = {
            'model': svm_model, 'cv_mean': svm_cv_mean, 'test_acc': svm_test_acc, 'pred': svm_pred
        }
    except Exception as e:
        print(f"         Skipped (dataset too small for SVM calibration: {e})")

    # ── Select best model by CV score (more reliable than test accuracy alone) ──
    print("\n[5/6] Selecting best model...")
    best_name = max(models, key=lambda k: models[k]['cv_mean'])
    best_info = models[best_name]
    best_model = best_info['model']
    best_accuracy = best_info['test_acc']
    best_pred = best_info['pred']

    # Overfitting check
    overfit_gap = best_info['cv_mean'] - best_accuracy
    print(f"\n       [BEST] {best_name}")
    print(f"       CV Accuracy:   {best_info['cv_mean']:.4f}")
    print(f"       Test Accuracy: {best_accuracy:.4f}")
    if abs(overfit_gap) > 0.05:
        print(f"       [!] Overfit gap: {overfit_gap:+.4f} (CV vs Test)")
    else:
        print(f"       [OK] No overfitting detected (gap: {overfit_gap:+.4f})")

    # Classification report
    print(f"\n       Classification Report ({best_name}):")
    report = classification_report(y_test, best_pred, zero_division=0)
    print(report)

    # Save model artifacts
    print("[6/6] Saving model artifacts...")
    model_dir = os.path.join(base_dir, "model")
    os.makedirs(model_dir, exist_ok=True)

    pickle.dump(best_model, open(os.path.join(model_dir, "model.pkl"), "wb"))
    pickle.dump(vectorizer, open(os.path.join(model_dir, "vectorizer.pkl"), "wb"))

    # Save training metrics for the dashboard
    metrics = {
        "best_model": best_name,
        "best_accuracy": round(best_accuracy * 100, 2),
        "best_cv_accuracy": round(best_info['cv_mean'] * 100, 2),
        "lr_accuracy": round(models['Logistic Regression']['test_acc'] * 100, 2),
        "nb_accuracy": round(models['Multinomial Naive Bayes']['test_acc'] * 100, 2),
        "svm_accuracy": round(models['Linear SVM']['test_acc'] * 100, 2),
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
