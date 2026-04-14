import pickle
import os
import numpy as np
from utils.preprocess import clean_text

_model = None
_vectorizer = None


def _get_paths():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return (
        os.path.join(base, "model", "model.pkl"),
        os.path.join(base, "model", "vectorizer.pkl"),
    )


def load_models():
    """Load pickled model and vectorizer. Returns True on success."""
    global _model, _vectorizer
    model_path, vec_path = _get_paths()
    if os.path.exists(model_path) and os.path.exists(vec_path):
        with open(model_path, "rb") as f:
            _model = pickle.load(f)
        with open(vec_path, "rb") as f:
            _vectorizer = pickle.load(f)
        return True
    return False


def predict(text):
    """Predict sentiment label for a single text string."""
    if _model is None or _vectorizer is None:
        if not load_models():
            return "Unknown (Model not trained)"

    cleaned = clean_text(text)
    vec = _vectorizer.transform([cleaned])
    return _model.predict(vec)[0]


def predict_with_confidence(text):
    """Predict sentiment label and return confidence score."""
    if _model is None or _vectorizer is None:
        if not load_models():
            return "Unknown", 0.0

    cleaned = clean_text(text)
    vec = _vectorizer.transform([cleaned])
    label = _model.predict(vec)[0]

    # Get probability if available
    if hasattr(_model, "predict_proba"):
        proba = _model.predict_proba(vec)[0]
        confidence = float(np.max(proba))
    else:
        confidence = 0.95  # fallback

    return label, round(confidence, 3)
